"""
__init__.py

Application factory and extension initialization for the Flask eCommerce app.
Sets up SQLAlchemy, Flask-Login, Flask-Migrate, and blueprint registration.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config

# Initialize extensions (not bound to app yet)
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'main_bp.login'  # Redirects to login page when not authenticated
migrate = Migrate()


def create_app():
    """
    Flask application factory.
    Creates and configures the app instance, registers blueprints,
    and initializes all Flask extensions and context processors.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions with the app
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    # Register context processors
    from .context_processors import inject_current_year
    app.context_processor(inject_current_year)

    with app.app_context():
        # Ensure models are registered before migrations or blueprints
        from . import models
        from . import routes

        # Register main blueprint for all routes
        app.register_blueprint(routes.main_bp)

    return app
