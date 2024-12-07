import logging
import re
import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from time import sleep
from typing import Literal, AsyncGenerator, Callable, Optional

from . import llm
from .export import export_chat_to_html
from .logmanager import LogManager, prepare_messages
from .message import (
    Message,
    len_tokens,
    msgs_to_toml,
    print_msg,
    toml_to_msgs,
)
from .llm.models import get_model
from .tools import ToolUse, execute_msg, loaded_tools
from .tools.base import ConfirmFunc
from .useredit import edit_text_with_editor
from .config import get_config, set_config_value, MemoryMode
from .memory import MemoryManager

logger = logging.getLogger(__name__)

Actions = Literal[
    "undo",
    "log",
    "tools",
    "edit",
    "rename",
    "fork",
    "summarize",
    "replay",
    "impersonate",
    "tokens",
    "export",
    "help",
    "exit",
    "memory",
]

action_descriptions: dict[Actions, str] = {
    "undo": "Undo the last action",
    "log": "Show the conversation log",
    "tools": "Show available tools",
    "edit": "Edit the conversation in your editor",
    "rename": "Rename the conversation",
    "fork": "Create a copy of the conversation with a new name",
    "summarize": "Summarize the conversation",
    "replay": "Re-execute codeblocks in the conversation, wont store output in log",
    "impersonate": "Impersonate the assistant",
    "tokens": "Show the number of tokens used",
    "export": "Export conversation as standalone HTML",
    "help": "Show this help message",
    "exit": "Exit the program",
    "memory": "Manage memory system settings and operations",
}
COMMANDS = list(action_descriptions.keys())


async def execute_cmd(msg: Message, log: LogManager, confirm: ConfirmFunc) -> bool:
    """Executes any user-command, returns True if command was executed."""
    assert msg.role == "user"

    # if message starts with / treat as command
    # when command has been run,
    if msg.content[:1] in ["/"]:
        async for resp in handle_cmd(msg.content, log, confirm):
            await log.append(resp)
        return True
    return False


async def handle_cmd(
    cmd: str,
    manager: LogManager,
    confirm: ConfirmFunc,
) -> AsyncGenerator[Message, None]:
    """Handles a command."""
    cmd = cmd.lstrip("/")
    logger.debug(f"Executing command: {cmd}")
    name, *args = re.split(r"[\n\s]", cmd)
    full_args = cmd.split(" ", 1)[1] if " " in cmd else ""
    match name:
        case "log":
            await manager.undo(1, quiet=True)
            manager.log.print(show_hidden="--hidden" in args)
        case "rename":
            await manager.undo(1, quiet=True)
            manager.write()
            # rename the conversation
            print("Renaming conversation")
            if args:
                new_name = args[0]
            else:
                print("(enter empty name to auto-generate)")
                new_name = input("New name: ").strip()
            await rename(manager, new_name, confirm)
        case "fork":
            # fork the conversation
            new_name = args[0] if args else input("New name: ")
            await manager.fork(new_name)
        case "summarize":
            msgs = await prepare_messages(manager.log.messages, memory_manager=manager.memory_manager)
            msgs = [m for m in msgs if not m.hide]
            summary = llm.summarize(msgs)
            print(f"Summary: {summary}")
        case "edit":
            # edit previous messages
            # first undo the '/edit' command itself
            await manager.undo(1, quiet=True)
            async for msg in edit(manager):
                yield msg
        case "undo":
            # undo the '/undo' command itself
            await manager.undo(1, quiet=True)
            # if int, undo n messages
            n = int(args[0]) if args and args[0].isdigit() else 1
            await manager.undo(n)
        case "exit":
            await manager.undo(1, quiet=True)
            manager.write()
            sys.exit(0)
        case "replay":
            await manager.undo(1, quiet=True)
            manager.write()
            print("Replaying conversation...")
            for msg in manager.log:
                if msg.role == "assistant":
                    for reply_msg in execute_msg(msg, confirm):
                        print_msg(reply_msg, oneline=False)
        case "impersonate":
            content = full_args if full_args else input("[impersonate] Assistant: ")
            msg = Message("assistant", content)
            yield msg
            # Convert sync generator to async
            for reply_msg in execute_msg(msg, confirm=lambda _: True):
                yield reply_msg
        case "tokens":
            await manager.undo(1, quiet=True)
            n_tokens = len_tokens(manager.log.messages)
            print(f"Tokens used: {n_tokens}")
            model = get_model()
            if model:
                print(f"Model: {model.model}")
                if model.price_input:
                    print(f"Cost (input): ${n_tokens * model.price_input / 1_000_000}")
        case "tools":
            await manager.undo(1, quiet=True)
            print("Available tools:")
            for tool in loaded_tools:
                print(
                    f"""
  # {tool.name}
    {tool.desc.rstrip(".")}
    tokens (example): {len_tokens(tool.examples)}"""
                )
        case "export":
            await manager.undo(1, quiet=True)
            manager.write()
            # Get output path from args or use default
            output_path = (
                Path(args[0]) if args else Path(f"{manager.logfile.parent.name}.html")
            )
            # Export the chat
            export_chat_to_html(manager.name, manager.log, output_path)
            print(f"Exported conversation to {output_path}")
        case "memory":
            await manager.undo(1, quiet=True)
            if not args:
                print("Memory system commands:")
                print("  search <query>  - Search memory for specific content")
                print("  status         - Show memory system status")
                print("  clear          - Clear memory for current conversation")
                print("  enable         - Enable memory system")
                print("  disable        - Disable memory system")
                return
            
            subcmd, *subargs = args
            match subcmd:
                case "search":
                    if not subargs:
                        print("Please provide a search query")
                        return
                    query = " ".join(subargs)
                    if hasattr(manager, "memory_manager") and manager.memory_manager:
                        results = await manager.memory_manager.search(query)
                        if results:
                            print(f"\nFound {len(results)} relevant memories:")
                            for result in results:
                                tags = " ".join(f"#{tag}" for tag in result.metadata.get("tags", []))
                                print(f"Score: {result.score:.2f} - {result.text} {tags}")
                        else:
                            print("No relevant memories found")
                    else:
                        print("Memory system not initialized")
                    
                case "status":
                    config = get_config()
                    print(f"Memory System Status:")
                    print(f"  Mode: {config.memory_mode}")
                    print(f"  API Key: {'configured' if config.mem0_api_key else 'not configured'}")
                    if hasattr(manager, "memory_manager") and manager.memory_manager:
                        mm = manager.memory_manager
                        print(f"  Failure Count: {mm._failure_count}")
                    else:
                        print("  Memory Manager: Not initialized")
                    
                case "enable":
                    config = get_config()
                    if not config.mem0_api_key:
                        print("Memory system requires API key to be configured")
                        return
                    
                    # Initialize memory manager with string mode
                    manager.memory_manager = MemoryManager(
                        api_key=config.mem0_api_key,
                        mode="full",  # Use string instead of enum
                        context={
                            "user_id": "default_user",
                            "conversation_id": manager.name
                        }
                    )
                    print("Memory system enabled")
                    
                case "disable":
                    manager.memory_manager = None
                    print("Memory system disabled")
                    
                case "clear":
                    if not hasattr(manager, "memory_manager"):
                        print("Memory system not initialized")
                        return
                    mm = getattr(manager, "memory_manager")
                    if mm.mode == "disabled":
                        print("Memory system is disabled")
                        return
                    try:
                        success = await mm.clear()
                        if success:
                            print("Memory cleared successfully")
                        else:
                            print("Failed to clear memory")
                    except Exception as e:
                        print(f"Failed to clear memory: {e}")
                    
                case _:
                    print(f"Unknown memory subcommand: {subcmd}")
                    print("Use /memory without arguments to see available commands")
        case _:
            # the case for python, shell, and other block_types supported by tools
            tooluse = ToolUse(name, [], full_args)
            if tooluse.is_runnable:
                # Convert sync generator to async
                for msg in tooluse.execute(confirm):
                    yield msg
            else:
                if manager.log[-1].content.strip() == "/help":
                    # undo the '/help' command itself
                    await manager.undo(1, quiet=True)
                    manager.write()
                    help()
                else:
                    print("Unknown command")


async def edit(manager: LogManager) -> AsyncGenerator[Message, None]:  # pragma: no cover
    # generate editable toml of all messages
    t = msgs_to_toml(reversed(manager.log))  # type: ignore
    res = None
    while not res:
        t = edit_text_with_editor(t, "toml")
        try:
            res = toml_to_msgs(t)
        except Exception as e:
            print(f"\nFailed to parse TOML: {e}")
            try:
                sleep(1)
            except KeyboardInterrupt:
                yield Message("system", "Interrupted")
                return
    await manager.edit(list(reversed(res)))
    print("Applied edited messages, write /log to see the result")


async def rename(manager: LogManager, new_name: str, confirm: ConfirmFunc) -> None:
    if new_name in ["", "auto"]:
        msgs = await prepare_messages(manager.log.messages, memory_manager=manager.memory_manager)[1:]  # skip system message
        new_name = llm.generate_name(msgs)
        assert " " not in new_name, f"Invalid name: {new_name}"
        print(f"Generated name: {new_name}")
        if not confirm("Confirm?"):
            print("Aborting")
            return
        manager.rename(new_name, keep_date=True)
    else:
        manager.rename(new_name, keep_date=False)
    print(f"Renamed conversation to {manager.logfile.parent}")


def _gen_help(incl_langtags: bool = True) -> Generator[str, None, None]:
    yield "Available commands:"
    max_cmdlen = max(len(cmd) for cmd in COMMANDS)
    for cmd, desc in action_descriptions.items():
        yield f"  /{cmd.ljust(max_cmdlen)}  {desc}"

    if incl_langtags:
        yield ""
        yield "To execute code with supported tools, use the following syntax:"
        yield "  /<langtag> <code>"
        yield ""
        yield "Example:"
        yield "  /sh echo hello"
        yield "  /python print('hello')"
        yield ""
        yield "Supported langtags:"
        for tool in loaded_tools:
            if tool.block_types:
                yield f"  - {tool.block_types[0]}" + (
                    f"  (alias: {', '.join(tool.block_types[1:])})"
                    if len(tool.block_types) > 1
                    else ""
                )


def help():
    for line in _gen_help():
        print(line)
