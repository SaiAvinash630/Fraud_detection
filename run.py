"""
run.py

Entry point for the Flask eCommerce application.
This script initializes the app using the application factory pattern
and runs the development server.
"""

from app import create_app, db  # Import the app factory
from app.feature_engineer import FeatureEngineer  # Import the feature engineer module

# Create the Flask application instance
app = create_app()
# Run the development server (debug mode is on for development only)
if __name__ == "__main__":
    app.run(debug=True)
