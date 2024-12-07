"""
Serve web UI and API for the application.

See here for instructions how to serve matplotlib figures:
 - https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html
"""

import atexit
import io
import logging
from collections.abc import Generator
from contextlib import redirect_stdout
from datetime import datetime
from importlib import resources
from itertools import islice

from quart import Blueprint, current_app, request, jsonify, Response
from quart_cors import cors

from ..commands import execute_cmd
from ..dirs import get_logs_dir
from ..llm import _stream
from ..logmanager import LogManager, get_user_conversations, prepare_messages
from ..message import Message
from ..llm.models import get_model
from ..tools import execute_msg
from ..tools.base import ToolUse

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)
cors(api)


@api.route("/api")
async def api_root():
    return jsonify({"message": "Hello World!"})


@api.route("/api/conversations")
async def api_conversations():
    limit = int(request.args.get("limit", 100))
    conversations = list(islice(get_user_conversations(), limit))
    return jsonify(conversations)


@api.route("/api/conversations/<path:logfile>")
async def api_conversation(logfile: str):
    """Get a conversation."""
    log = LogManager.load(logfile)
    return jsonify(log.to_dict(branches=True))


@api.route("/api/conversations/<path:logfile>", methods=["PUT"])
async def api_conversation_put(logfile: str):
    """Create or update a conversation."""
    msgs = []
    req_json = await request.get_json()
    if req_json and "messages" in req_json:
        for msg in req_json["messages"]:
            timestamp: datetime = datetime.fromisoformat(msg["timestamp"])
            msgs.append(Message(msg["role"], msg["content"], timestamp=timestamp))

    logdir = get_logs_dir() / logfile
    if logdir.exists():
        raise ValueError(f"Conversation already exists: {logdir.name}")
    logdir.mkdir(parents=True)
    log = LogManager(msgs, logdir=logdir)
    await log.write()
    return {"status": "ok"}


@api.route(
    "/api/conversations/<path:logfile>",
    methods=["POST"],
)
async def api_conversation_post(logfile: str):
    """Post a message to the conversation."""
    req_json = await request.get_json()
    branch = (req_json or {}).get("branch", "main")
    log = LogManager.load(logfile, branch=branch)
    assert req_json
    assert "role" in req_json
    assert "content" in req_json
    msg = Message(
        req_json["role"], req_json["content"], files=req_json.get("files", [])
    )
    await log.append(msg)
    return {"status": "ok"}


# TODO: add support for confirmation
def confirm_func(msg: str) -> bool:
    return True


# generate response
@api.route("/api/conversations/<path:logfile>/generate", methods=["POST"])
async def api_conversation_generate(logfile: str):
    # get model or use server default
    req_json = await request.get_json() or {}
    stream = req_json.get("stream", False)  # Default to no streaming (backward compat)
    model = req_json.get("model", get_model().model)

    # load conversation
    manager = LogManager.load(logfile, branch=req_json.get("branch", "main"))

    # performs reduction/context trimming, if necessary
    msgs = await prepare_messages(manager.log.messages, memory_manager=manager.memory_manager)

    if not msgs:
        logger.error("No messages to process")
        return jsonify({"error": "No messages to process"})

    if not stream:
        # Non-streaming response
        try:
            # Get complete response
            output = "".join(_stream(msgs, model))

            # Store the message
            msg = Message("assistant", output)
            msg = msg.replace(quiet=True)
            await manager.append(msg)

            # Execute any tools
            reply_msgs = list(execute_msg(msg, confirm_func))
            for reply_msg in reply_msgs:
                await manager.append(reply_msg)

            # Return all messages
            response = [{"role": "assistant", "content": output, "stored": True}]
            response.extend(
                {"role": msg.role, "content": msg.content, "stored": True}
                for msg in reply_msgs
            )
            return jsonify(response)

        except Exception as e:
            logger.exception("Error during generation")
            return jsonify({"error": str(e)})

    # Streaming response
    async def generate() -> Generator[str, None, None]:
        # Start with an empty message
        output = ""
        try:
            logger.info(f"Starting generation for conversation {logfile}")

            # Prepare messages for the model
            if not msgs:
                logger.error("No messages to process")
                yield f"data: {jsonify({'error': 'No messages to process'}).get_data(as_text=True)}\n\n"
                return

            # if prompt is a user-command, execute it
            last_msg = manager.log[-1]
            if last_msg.role == "user" and last_msg.content.startswith("/"):
                f = io.StringIO()
                print("Begin capturing stdout, to pass along command output.")
                with redirect_stdout(f):
                    resp = await execute_cmd(manager.log[-1], manager, confirm_func)
                print("Done capturing stdout.")
                output = f.getvalue().strip()
                if resp and output:
                    print(f"Replying with command output: {output}")
                    await manager.write()
                    yield f"data: {jsonify({'role': 'system', 'content': output, 'stored': False}).get_data(as_text=True)}\n\n"
                    return

            # Stream tokens from the model
            logger.debug(f"Starting token stream with model {model}")
            for char in (char for chunk in _stream(msgs, model) for char in chunk):
                output += char
                # Send each token as a JSON event
                yield f"data: {jsonify({'role': 'assistant', 'content': char, 'stored': False}).get_data(as_text=True)}\n\n"

                # Check for complete tool uses
                tooluses = list(ToolUse.iter_from_content(output))
                if tooluses and any(tooluse.is_runnable for tooluse in tooluses):
                    logger.debug("Found runnable tool use, breaking stream")
                    break

            # Store the complete message
            logger.debug(f"Storing complete message: {output[:100]}...")
            msg = Message("assistant", output)
            msg = msg.replace(quiet=True)
            await manager.append(msg)
            yield f"data: {jsonify({'role': 'assistant', 'content': output, 'stored': True}).get_data(as_text=True)}\n\n"

            # Execute any tools and stream their output
            for reply_msg in execute_msg(msg, confirm_func):
                logger.debug(
                    f"Tool output: {reply_msg.role} - {reply_msg.content[:100]}..."
                )
                await manager.append(reply_msg)
                yield f"data: {jsonify({'role': reply_msg.role, 'content': reply_msg.content, 'stored': True}).get_data(as_text=True)}\n\n"

        except GeneratorExit:
            logger.info("Client disconnected during generation, interrupting")
            if output:
                output += "\n\n[interrupted]"
                msg = Message("assistant", output)
                msg = msg.replace(quiet=True)
                await manager.append(msg)
            raise
        except Exception as e:
            logger.exception("Error during generation")
            yield f"data: {jsonify({'error': str(e)}).get_data(as_text=True)}\n\n"
        finally:
            logger.info("Generation completed")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


gptme_path_ctx = resources.as_file(resources.files("gptme"))
root_path = gptme_path_ctx.__enter__()
static_path = root_path / "server" / "static"
media_path = root_path.parent / "media"
atexit.register(gptme_path_ctx.__exit__, None, None, None)


# serve index.html from the root
@api.route("/")
async def root():
    return await current_app.send_static_file("index.html")


# serve computer interface
@api.route("/computer")
async def computer():
    return await current_app.send_static_file("computer.html")


# serve chat interface (for embedding in computer view)
@api.route("/chat")
async def chat():
    return await current_app.send_static_file("index.html")


@api.route("/favicon.png")
async def favicon():
    return await current_app.send_from_directory(media_path, "logo.png")
