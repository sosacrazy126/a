import logging
import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS
from quart import Quart
from quart_cors import cors

from ..api import api
from ..init import init_logging
from ..llm.models import init_model

logger = logging.getLogger(__name__)

app = Quart(__name__)
cors(app)

app.register_blueprint(api)

def main():
    """Main entrypoint for the server."""
    init_logging(verbose=True)
    init_model()

    # get port from env or use default
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")

    # run server
    app.run(host=host, port=port)

if __name__ == "__main__":
    main()
