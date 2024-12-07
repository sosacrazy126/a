import json
import logging
import shutil
import textwrap
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import islice, zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeAlias, Optional

from rich import print

from .dirs import get_logs_dir
from .message import Message, len_tokens, print_msg
from .prompts import get_prompt
from .reduce import limit_log, reduce_log
from .config import get_config
from .memory import MemoryManager, MemoryMode

PathLike: TypeAlias = str | Path

logger = logging.getLogger(__name__)

RoleLiteral = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class Log:
    messages: list[Message] = field(default_factory=list)

    def __getitem__(self, key):
        return self.messages[key]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Generator[Message, None, None]:
        yield from self.messages

    def replace(self, **kwargs) -> "Log":
        return replace(self, **kwargs)

    def append(self, msg: Message) -> "Log":
        return self.replace(messages=self.messages + [msg])

    def pop(self) -> "Log":
        return self.replace(messages=self.messages[:-1])

    @classmethod
    def read_jsonl(cls, path: PathLike, limit=None) -> "Log":
        gen = _gen_read_jsonl(path)
        if limit:
            gen = islice(gen, limit)  # type: ignore
        return Log(list(gen))

    def write_jsonl(self, path: PathLike) -> None:
        with open(path, "w") as file:
            for msg in self.messages:
                file.write(json.dumps(msg.to_dict()) + "\n")

    def print(self, show_hidden: bool = False):
        print_msg(self.messages, oneline=False, show_hidden=show_hidden)

    async def get_messages_for_model(
        self,
        memory_manager: Optional[MemoryManager] = None
    ) -> list[Message]:
        """Get messages formatted for the model, including memory context if available."""
        # Get messages, excluding those marked as quiet
        msgs = [msg for msg in self.messages if not msg.quiet]
        
        # Get memory context if available
        memory_context = []
        if (
            memory_manager 
            and memory_manager.mode != MemoryMode.DISABLED 
            and msgs
        ):
            try:
                # Get context based on the last message
                last_msg = msgs[-1].content
                
                # First try searching with the actual question
                memory_context = await memory_manager.get_relevant_context(
                    query=last_msg,
                    limit=5
                )
                
                # If no results and it's a question about personal info
                if not memory_context and any(q in last_msg.lower() for q in ["name", "who", "what", "when", "where", "why", "how"]):
                    # Get all memories and filter locally
                    all_messages = await memory_manager.get_all_messages()
                    
                    # Filter relevant messages based on content
                    for msg in all_messages:
                        content = msg.content.lower()
                        # Look for personal info patterns
                        if any(pattern in content for pattern in [
                            "my name is",
                            "i am",
                            "hi, i'm",
                            "call me",
                            "i work",
                            "i live",
                            "i like",
                            "i enjoy"
                        ]):
                            memory_context.append(msg)
                    
                    # Limit the number of context messages
                    memory_context = memory_context[:5]
                
                if memory_context:
                    # Add a system message to explain the context
                    memory_context.insert(0, Message(
                        role="system",
                        content="Here is some relevant context from previous conversations:",
                        timestamp=datetime.now()
                    ))
                    logger.info(f"Added {len(memory_context)} messages from memory context")
            except Exception as e:
                logger.warning(f"Failed to get memory context: {e}")
                memory_context = []
        
        # Combine memory context with current messages
        return memory_context + msgs


class LogManager:
    """Manages a conversation log."""

    def __init__(
        self,
        log: list[Message] | None = None,
        logdir: PathLike | None = None,
        branch: str | None = None,
    ):
        self.current_branch = branch or "main"

        if logdir:
            self.logdir = Path(logdir)
        else:
            # generate tmpfile
            fpath = TemporaryDirectory().name
            logger.warning(f"No logfile specified, using tmpfile at {fpath}")
            self.logdir = Path(fpath)
        self.name = self.logdir.name

        # Initialize memory manager
        config = get_config()
        self.memory_manager = None
        if config.memory_mode != MemoryMode.DISABLED:
            try:
                self.memory_manager = MemoryManager(
                    api_key=config.mem0_api_key,
                    mode=config.memory_mode,
                    fallback_threshold=config.memory_fallback_threshold,
                    context={
                        "conversation_id": self.name,
                        "branch": self.current_branch
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize memory manager: {e}")
                self.memory_manager = None

        # load branches from adjacent files
        self._branches = {self.current_branch: Log(log or [])}
        if self.logdir / "conversation.jsonl":
            _branch = "main"
            if _branch not in self._branches:
                self._branches[_branch] = Log.read_jsonl(
                    self.logdir / "conversation.jsonl"
                )
        for file in self.logdir.glob("branches/*.jsonl"):
            if file.name == self.logdir.name:
                continue
            _branch = file.stem
            if _branch not in self._branches:
                self._branches[_branch] = Log.read_jsonl(file)

    @property
    def log(self) -> Log:
        return self._branches[self.current_branch]

    @log.setter
    def log(self, value: Log | list[Message]) -> None:
        if isinstance(value, list):
            value = Log(value)
        self._branches[self.current_branch] = value

    @property
    def logfile(self) -> Path:
        if self.current_branch == "main":
            return get_logs_dir() / self.name / "conversation.jsonl"
        return self.logdir / "branches" / f"{self.current_branch}.jsonl"

    async def append(self, msg: Message) -> None:
        """Appends a message to the log, writes the log, prints the message, and stores in memory."""
        self.log = self.log.append(msg)
        self.write()
        
        # Store message in memory if available
        if (
            self.memory_manager 
            and self.memory_manager.mode != MemoryMode.DISABLED
            and msg.role in ("user", "assistant")  # Don't store system messages
        ):
            try:
                await self.memory_manager.store_message(msg)
            except Exception as e:
                logger.warning(f"Failed to store message in memory: {e}")
        
        if not msg.quiet:
            print_msg(msg, oneline=False)

    def write(self, branches=True) -> None:
        """
        Writes to the conversation log.
        """
        # create directory if it doesn't exist
        Path(self.logfile).parent.mkdir(parents=True, exist_ok=True)

        # write current branch
        self.log.write_jsonl(self.logfile)

        # write other branches
        # FIXME: wont write main branch if on a different branch
        if branches:
            branches_dir = self.logdir / "branches"
            branches_dir.mkdir(parents=True, exist_ok=True)
            for branch, log in self._branches.items():
                if branch == "main":
                    continue
                branch_path = branches_dir / f"{branch}.jsonl"
                log.write_jsonl(branch_path)

    def _save_backup_branch(self, type="edit") -> None:
        """backup the current log to a new branch, usually before editing/undoing"""
        branch_prefix = f"{self.current_branch}-{type}-"
        n = len([b for b in self._branches.keys() if b.startswith(branch_prefix)])
        self._branches[f"{branch_prefix}{n}"] = self.log
        self.write()

    async def edit(self, new_log: Log | list[Message]) -> None:
        """Edits the log."""
        if isinstance(new_log, list):
            new_log = Log(new_log)
        self._save_backup_branch(type="edit")
        
        # Clear memory for this conversation if available
        if (
            self.memory_manager 
            and self.memory_manager.mode != MemoryMode.DISABLED
        ):
            try:
                await self.memory_manager.clear()
                # Re-store all messages in the new log
                for msg in new_log:
                    if msg.role in ("user", "assistant"):
                        await self.memory_manager.store_message(msg)
            except Exception as e:
                logger.warning(f"Failed to update memory after edit: {e}")
        
        self.log = new_log
        self.write()

    async def undo(self, n: int = 1, quiet=False) -> None:
        """Removes the last message from the log."""
        undid = self.log[-1] if self.log else None
        if undid and undid.content.startswith("/undo"):
            self.log = self.log.pop()

        # don't save backup branch if undoing a command
        if self.log and not self.log[-1].content.startswith("/"):
            self._save_backup_branch(type="undo")

        peek = self.log[-1] if self.log else None
        if not peek:
            print("[yellow]Nothing to undo.[/]")
            return

        if not quiet:
            print("[yellow]Undoing messages:[/yellow]")
            
        # Track messages to remove from memory
        removed_msgs = []
        for _ in range(n):
            undid = self.log[-1]
            self.log = self.log.pop()
            removed_msgs.append(undid)
            if not quiet:
                print(
                    f"[red]  {undid.role}: {textwrap.shorten(undid.content.strip(), width=50, placeholder='...')}[/]",
                )
            peek = self.log[-1] if self.log else None
            
        # Remove messages from memory if available
        if (
            self.memory_manager 
            and self.memory_manager.mode != MemoryMode.DISABLED
            and removed_msgs
        ):
            try:
                for msg in removed_msgs:
                    if msg.role in ("user", "assistant"):
                        await self.memory_manager.remove_message(msg)
            except Exception as e:
                logger.warning(f"Failed to remove messages from memory: {e}")

    @classmethod
    def load(
        cls,
        logdir: PathLike,
        initial_msgs: list[Message] | None = None,
        branch: str = "main",
        create: bool = False,
        **kwargs,
    ) -> "LogManager":
        """Loads a conversation log."""
        if str(logdir).endswith(".jsonl"):
            logdir = Path(logdir).parent

        logsdir = get_logs_dir()
        if str(logsdir) not in str(logdir):
            # if the path was not fully specified, assume its a dir in logsdir
            logdir = logsdir / logdir
        else:
            logdir = Path(logdir)

        if branch == "main":
            logfile = logdir / "conversation.jsonl"
        else:
            logfile = logdir / f"branches/{branch}.jsonl"

        if not Path(logfile).exists():
            if create:
                logger.debug(f"Creating new logfile {logfile}")
                Path(logfile).parent.mkdir(parents=True, exist_ok=True)
                Log([]).write_jsonl(logfile)
            else:
                raise FileNotFoundError(f"Could not find logfile {logfile}")

        log = Log.read_jsonl(logfile)
        msgs = log.messages or initial_msgs or [get_prompt()]
        return cls(msgs, logdir=logdir, branch=branch, **kwargs)

    def branch(self, name: str) -> None:
        """Switches to a branch."""
        self.write()
        if name not in self._branches:
            logger.info(f"Creating a new branch '{name}'")
            self._branches[name] = self.log
        self.current_branch = name

    def diff(self, branch: str) -> str | None:
        """Prints the diff between the current branch and another branch."""
        if branch not in self._branches:
            logger.warning(f"Branch '{branch}' does not exist.")
            return None

        # walk the log forwards until we find a message that is different
        diff_i: int | None = None
        for i, (msg1, msg2) in enumerate(zip_longest(self.log, self._branches[branch])):
            diff_i = i
            if msg1 != msg2:
                break
        else:
            # no difference
            return None

        # output the continuing messages on the current branch as +
        # and the continuing messages on the other branch as -
        diff = []
        for msg in self.log[diff_i:]:
            diff.append(f"+ {msg.format()}")
        for msg in self._branches[branch][diff_i:]:
            diff.append(f"- {msg.format()}")

        if diff:
            return "\n".join(diff)
        else:
            return None

    def rename(self, name: str, keep_date=False) -> None:
        """
        Rename the conversation.
        Renames the folder containing the conversation and its branches.

        If keep_date is True, we will keep the date part of conversation folder name ("2021-08-01-some-name")
        If you want to keep the old log, use fork()
        """
        if keep_date:
            name = f"{self.logfile.parent.name[:10]}-{name}"

        logsdir = get_logs_dir()
        new_logdir = logsdir / name
        if new_logdir.exists():
            raise FileExistsError(f"Conversation {name} already exists.")
        self.name = name
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logdir.rename(logsdir / self.name)
        self.logdir = logsdir / self.name

    def fork(self, name: str) -> None:
        """
        Copy the conversation folder to a new name.
        """
        self.write()
        logsdir = get_logs_dir()
        shutil.copytree(self.logfile.parent, logsdir / name)
        self.logdir = logsdir / name
        self.write()

    def to_dict(self, branches=False) -> dict:
        """Returns a dict representation of the log."""
        d: dict[str, Any] = {
            "log": [msg.to_dict() for msg in self.log],
            "logfile": str(self.logfile),
        }
        if branches:
            d["branches"] = {
                branch: [msg.to_dict() for msg in msgs]
                for branch, msgs in self._branches.items()
            }
        return d

    async def get_messages_for_model(
        self,
        memory_manager: Optional[MemoryManager] = None
    ) -> list[Message]:
        """Get messages formatted for the model, including memory context if available."""
        # Get messages, excluding those marked as quiet
        msgs = [msg for msg in self.messages if not msg.quiet]
        
        # Get memory context if available
        memory_context = []
        if (
            memory_manager 
            and memory_manager.mode != MemoryMode.DISABLED 
            and msgs
        ):
            try:
                # Get context based on the last message
                last_msg = msgs[-1].content
                
                # First try searching with the actual question
                memory_context = await memory_manager.get_relevant_context(
                    query=last_msg,
                    limit=5
                )
                
                # If no results and it's a question about personal info
                if not memory_context and any(q in last_msg.lower() for q in ["name", "who", "what", "when", "where", "why", "how"]):
                    # Get all memories and filter locally
                    all_messages = await memory_manager.get_all_messages()
                    
                    # Filter relevant messages based on content
                    for msg in all_messages:
                        content = msg.content.lower()
                        # Look for personal info patterns
                        if any(pattern in content for pattern in [
                            "my name is",
                            "i am",
                            "hi, i'm",
                            "call me",
                            "i work",
                            "i live",
                            "i like",
                            "i enjoy"
                        ]):
                            memory_context.append(msg)
                    
                    # Limit the number of context messages
                    memory_context = memory_context[:5]
                
                if memory_context:
                    # Add a system message to explain the context
                    memory_context.insert(0, Message(
                        role="system",
                        content="Here is some relevant context from previous conversations:",
                        timestamp=datetime.now()
                    ))
                    logger.info(f"Added {len(memory_context)} messages from memory context")
            except Exception as e:
                logger.warning(f"Failed to get memory context: {e}")
                memory_context = []
        
        # Combine memory context with current messages
        return memory_context + msgs

    async def store_message(self, message: Message) -> None:
        """Store a message in memory."""
        if not self.memory_manager:
            return
            
        try:
            metadata = {
                "role": message.role,
                "timestamp": message.timestamp.isoformat(),
                "tags": ["conversation"]
            }
            await self.memory_manager.store(message.content, metadata)
        except Exception as e:
            logger.warning(f"Failed to store message in memory: {e}")
            
    async def get_relevant_context(self, query: str, limit: int = 5) -> list[Message]:
        """Get relevant context from memory."""
        if not self.memory_manager:
            return []
            
        try:
            results = await self.memory_manager.search(query, limit=limit)
            messages = []
            
            for result in results:
                try:
                    messages.append(Message(
                        role=result.metadata.get("role", "user"),
                        content=result.text,
                        timestamp=result.timestamp
                    ))
                except Exception as e:
                    logger.warning(f"Failed to convert memory result to message: {e}")
                    continue
                    
            return messages
            
        except Exception as e:
            logger.warning(f"Failed to get relevant context: {e}")
            return []


async def prepare_messages(
    msgs: list[Message],
    memory_manager: MemoryManager | None = None,
) -> list[Message]:
    """
    Prepares the messages before sending to the LLM.
    
    Args:
        msgs: List of messages to prepare
        memory_manager: Optional memory manager to get relevant context
    """
    from .tools._rag_context import _HAS_RAG, enhance_messages  # fmt: skip
    
    # Get memory context if available
    memory_context = []
    if (
        memory_manager 
        and memory_manager.mode != MemoryMode.DISABLED 
        and msgs
    ):
        try:
            # Get context based on the last message
            memory_context = await memory_manager.get_relevant_context(
                msgs[-1].content
            )
            if memory_context:
                logger.info(
                    f"Added {len(memory_context)} messages from memory context"
                )
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            memory_context = []
    
    # Combine memory context with current messages
    messages = memory_context + msgs
    
    # Enhance with RAG context if available
    if _HAS_RAG:
        messages = enhance_messages(messages)
        
    return messages


def _conversation_files() -> list[Path]:
    # NOTE: only returns the main conversation, not branches (to avoid duplicates)
    # returns the conversation files sorted by modified time (newest first)
    logsdir = get_logs_dir()
    return list(
        sorted(logsdir.glob("*/conversation.jsonl"), key=lambda f: -f.stat().st_mtime)
    )


@dataclass(frozen=True)
class ConversationMeta:
    """Metadata about a conversation."""

    name: str
    path: str
    created: float
    modified: float
    messages: int
    branches: int

    def format(self, metadata=False) -> str:
        """Format conversation metadata for display."""
        output = f"{self.name}"
        if metadata:
            output += f"\nMessages: {self.messages}"
            output += f"\nCreated:  {datetime.fromtimestamp(self.created)}"
            output += f"\nModified: {datetime.fromtimestamp(self.modified)}"
            if self.branches > 1:
                output += f"\n({self.branches} branches)"
        return output


def get_conversations() -> Generator[ConversationMeta, None, None]:
    """Returns all conversations, excluding ones used for testing, evals, etc."""
    for conv_fn in _conversation_files():
        log = Log.read_jsonl(conv_fn, limit=1)
        # TODO: can we avoid reading the entire file? maybe wont even be used, due to user convo filtering
        len_msgs = conv_fn.read_text().count("}\n{")
        assert len(log) <= 1
        modified = conv_fn.stat().st_mtime
        first_timestamp = log[0].timestamp.timestamp() if log else modified
        yield ConversationMeta(
            name=f"{conv_fn.parent.name}",
            path=str(conv_fn),
            created=first_timestamp,
            modified=modified,
            messages=len_msgs,
            branches=1 + len(list(conv_fn.parent.glob("branches/*.jsonl"))),
        )


def get_user_conversations() -> Generator[ConversationMeta, None, None]:
    """Returns all user conversations, excluding ones used for testing, evals, etc."""
    for conv in get_conversations():
        if any(conv.name.startswith(prefix) for prefix in ["tmp", "test-"]) or any(
            substr in conv.name for substr in ["gptme-evals-"]
        ):
            continue
        yield conv


def list_conversations(
    limit: int = 20,
    include_test: bool = False,
) -> list[ConversationMeta]:
    """
    List conversations with a limit.

    Args:
        limit: Maximum number of conversations to return
        include_test: Whether to include test conversations
    """
    conversation_iter = (
        get_conversations() if include_test else get_user_conversations()
    )
    return list(islice(conversation_iter, limit))


def _gen_read_jsonl(path: PathLike) -> Generator[Message, None, None]:
    with open(path) as file:
        for line in file.readlines():
            json_data = json.loads(line)
            files = [Path(f) for f in json_data.pop("files", [])]
            if "timestamp" in json_data:
                json_data["timestamp"] = datetime.fromisoformat(json_data["timestamp"])
            yield Message(**json_data, files=files)
