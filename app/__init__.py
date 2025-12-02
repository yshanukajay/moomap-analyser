# Flask app factory
from flask import Flask
from .config import Config
from .db import init_db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize MongoDB
    init_db(app)

    # Register API routes
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Register error handlers
    from .errors import register_error_handlers
    register_error_handlers(app)

    return app
