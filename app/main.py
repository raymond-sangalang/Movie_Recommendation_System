""" main.py - Entry point for flask and executes the web application. """
from flask import Flask
from routes import application_routes



def create_app():
    # Initialize Flask object with the main thread
    # and create the blueprints of the application

    app = Flask(__name__)
    app.register_blueprint(application_routes)
    return app


if __name__ == "__main__" :

    # Initialize and start running the application
    app = create_app()
    app.run(debug=True)
