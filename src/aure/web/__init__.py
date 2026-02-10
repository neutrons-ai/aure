"""
Flask web application for visualizing AuRE workflow results.

Usage:
    from aure.web import create_app
    app = create_app("/path/to/output")
    app.run(port=5000)

Or via the CLI:
    aure serve ./output
"""

from flask import Flask
from .routes import bp


def create_app(output_dir: str) -> Flask:
    """
    Create the Flask application.

    Args:
        output_dir: Path to the workflow output directory.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    app.config["OUTPUT_DIR"] = output_dir
    app.register_blueprint(bp)
    return app
