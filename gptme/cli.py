import importlib.metadata
import logging
import os
import signal
import sys
import asyncio
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Literal

import click
from pick import pick

from .chat import chat
from .commands import _gen_help
from .constants import MULTIPROMPT_SEPARATOR
from .dirs import get_logs_dir
from .init import init_logging
from .interrupt import handle_keyboard_interrupt, set_interruptible
from .logmanager import ConversationMeta, get_user_conversations
from .message import Message
from .prompts import get_prompt
from .tools import all_tools, init_tools
from .util import epoch_to_age, generate_name
from .util.readline import add_history

logger = logging.getLogger(__name__)


script_path = Path(os.path.realpath(__file__))
commands_help = "\n".join(_gen_help(incl_langtags=False))
available_tool_names = ", ".join([tool.name for tool in all_tools if tool.available])


docstring = f"""
gptme is a chat-CLI for LLMs, empowering them with tools to run shell commands, execute code, read and manipulate files, and more.

If PROMPTS are provided, a new conversation will be started with it.
PROMPTS can be chained with the '{MULTIPROMPT_SEPARATOR}' separator.

The interface provides user commands that can be used to interact with the system.

\b
{commands_help}"""


@click.command(help=docstring)
@click.argument(
    "prompts",
    default=None,
    required=False,
    nargs=-1,
)
@click.option(
    "-n",
    "--name",
    default="random",
    help="Name of conversation. Defaults to generating a random name.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Model to use, e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet-20240620. If only provider given, a default is used.",
)
@click.option(
    "-w",
    "--workspace",
    default=None,
    help="Path to workspace directory. Pass '@log' to create a workspace in the log directory.",
)
@click.option(
    "-r",
    "--resume",
    is_flag=True,
    help="Load last conversation",
)
@click.option(
    "-y",
    "--no-confirm",
    is_flag=True,
    help="Skips all confirmation prompts.",
)
@click.option(
    "-n",
    "--non-interactive",
    "interactive",
    default=True,
    flag_value=False,
    help="Force non-interactive mode. Implies --no-confirm.",
)
@click.option(
    "--system",
    "prompt_system",
    default="full",
    help="System prompt. Can be 'full', 'short', or something custom.",
)
@click.option(
    "-t",
    "--tools",
    "tool_allowlist",
    default=None,
    multiple=True,
    help=f"Comma-separated list of tools to allow. Available: {available_tool_names}.",
)
@click.option(
    "--no-stream",
    "stream",
    default=True,
    flag_value=False,
    help="Don't stream responses",
)
@click.option(
    "--show-hidden",
    is_flag=True,
    help="Show hidden system messages.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show verbose output.",
)
@click.option(
    "--version",
    is_flag=True,
    help="Show version and configuration information",
)
def main(
    prompts: list[str],
    prompt_system: str,
    name: str,
    model: str | None,
    tool_allowlist: list[str] | None,
    stream: bool,
    verbose: bool,
    no_confirm: bool,
    interactive: bool,
    show_hidden: bool,
    version: bool,
    resume: bool,
    workspace: str | None,
):
    """Main entrypoint for the CLI."""
    if version:
        # print version
        print(f"gptme {importlib.metadata.version('gptme')}")

        # print dirs
        print(f"Logs dir: {get_logs_dir()}")

        exit(0)

    if "PYTEST_CURRENT_TEST" in os.environ:
        interactive = False

    # init logging
    init_logging(verbose)

    if not interactive:
        no_confirm = True

    if no_confirm:
        logger.warning("Skipping all confirmation prompts.")

    if tool_allowlist:
        # split comma-separated values
        tool_allowlist = [tool for tools in tool_allowlist for tool in tools.split(",")]

    # early init tools to generate system prompt
    init_tools(tool_allowlist)

    # get initial system prompt
    initial_msgs = [get_prompt(prompt_system, interactive=interactive)]

    # if stdin is not a tty, we might be getting piped input, which we should include in the prompt
    was_piped = False
    if not sys.stdin.isatty():
        # fetch prompt from stdin
        prompt_stdin = _read_stdin()
        if prompt_stdin:
            # TODO: also append if existing convo loaded/resumed
            initial_msgs += [Message("system", f"```stdin\n{prompt_stdin}\n```")]
            was_piped = True

            # Attempt to switch to interactive mode
            sys.stdin.close()
            try:
                sys.stdin = open("/dev/tty")
            except OSError:
                # if we can't open /dev/tty, we're probably in a CI environment, so we should just continue
                logger.warning(
                    "Failed to switch to interactive mode, continuing in non-interactive mode"
                )

    # add prompts to readline history
    for prompt in prompts:
        if prompt and len(prompt) > 1000:
            # skip adding long prompts to history (slows down startup, unlikely to be useful)
            continue
        add_history(prompt)

    # join prompts, grouped by `-` if present, since that's the separator for "chained"/multiple-round prompts
    sep = "\n\n" + MULTIPROMPT_SEPARATOR
    prompts = [p.strip() for p in "\n\n".join(prompts).split(sep) if p]
    # TODO: referenced file paths in multiprompts should be read when run, not when parsed
    prompt_msgs = [Message("user", p) for p in prompts]

    if resume:
        logdir = get_logdir_resume()
    # don't run pick in tests/non-interactive mode, or if the user specifies a name
    elif (
        interactive
        and name == "random"
        and not prompt_msgs
        and not was_piped
        and sys.stdin.isatty()
    ):
        logdir = pick_log()
    else:
        logdir = get_logdir(name)

    if workspace == "@log":
        workspace_path: Path | None = logdir / "workspace"
        assert workspace_path  # mypy not smart enough to see its not None
        workspace_path.mkdir(parents=True, exist_ok=True)
    else:
        workspace_path = Path(workspace) if workspace else None

    # register a handler for Ctrl-C
    set_interruptible()  # prepare, user should be able to Ctrl+C until user prompt ready
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    # Run the async chat function
    asyncio.run(chat(
        prompt_msgs,
        initial_msgs,
        logdir,
        model,
        stream,
        no_confirm,
        interactive,
        show_hidden,
        workspace_path,
        tool_allowlist,
    ))


def get_name(name: str) -> str:
    """
    Returns a name for the new conversation.
    """
    if name == "random":
        name = generate_name()
    return name


def get_logdir(name: str) -> Path:
    """
    Returns a path to a new conversation directory.
    """
    name = get_name(name)
    logdir = get_logs_dir() / f"{datetime.now().strftime('%Y-%m-%d')}-{name}"
    if logdir.exists():
        raise FileExistsError(f"Conversation {name} already exists.")
    return logdir


def get_logdir_resume() -> Path:
    """
    Returns a path to the most recent conversation directory.
    """
    conversations = list(get_user_conversations())
    if not conversations:
        raise FileNotFoundError("No conversations found.")
    return Path(conversations[0].path).parent


def pick_log() -> Path:
    """
    Shows a picker for selecting a conversation.
    """
    conversations = list(islice(get_user_conversations(), 100))
    if not conversations:
        return get_logdir("random")

    # format conversations for picker
    options = []
    # Add "New Conversation" option at the top
    options.append("* New Conversation")
    
    # Add existing conversations
    for conv in conversations:
        age = epoch_to_age(conv.modified)
        options.append(f"{conv.name} ({age})")

    try:
        # show picker
        option, index = pick(options, "Select a conversation (or press q to start a new one):", indicator=">")
        
        # If "New Conversation" was selected (index 0)
        if index == 0:
            return get_logdir("random")
            
        # get selected conversation (subtract 1 from index due to "New Conversation" option)
        conv = conversations[index - 1]
        return Path(conv.path).parent
    except (KeyboardInterrupt, EOFError, ValueError):
        # Handle 'q' press by creating a new conversation
        # ValueError is raised by pick when 'q' is pressed
        return get_logdir("random")


def _read_stdin() -> str | None:
    """
    Reads from stdin if available.
    """
    try:
        stdin = sys.stdin.read()
        if stdin:
            return stdin.strip()
    except:
        pass
    return None


if __name__ == "__main__":
    main()
