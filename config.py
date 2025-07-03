"""
config.py

Central configuration file for the Flask eCommerce application.
Loads environment variables using dotenv and provides the main Config class
used to configure the app (e.g., secret keys, database URI, Stripe keys).
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define base directory path
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """
    Configuration class for Flask app settings.
    """

    # Secret key for CSRF protection and session management
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_key")
    print("\n\n\n\nBase directory:", basedir)
    # Handle PostgreSQL URL compatibility (for Render/Heroku)
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Use provided DATABASE_URL or fallback to local SQLite
    SQLALCHEMY_DATABASE_URI = (
        f'sqlite:///{os.path.join(basedir, "instance", "ecommerce.db")}'
    )

    # Disable modification tracking to save system resources
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Stripe API credentials
    STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY", "")
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
