""" main.py - Entry point for flask and executes the web application. """
from flask import Flask
from routes import application_routes
from reviews_db import init_db



def create_app():
    # Initialize Flask object with the main thread,
    # initializes a database, and 
    # create the blueprints of the application.

    app = Flask(__name__)
    init_db()   
    app.register_blueprint(application_routes)
    
    return app


if __name__ == "__main__" :

    # Initialize and start running the application
    app = create_app()
    app.run(debug=True)

