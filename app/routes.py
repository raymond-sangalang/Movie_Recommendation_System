""" routes.py - Initializes user routes as well as rendering templates. """
from flask import Blueprint, render_template, request
from recommender_utils import load_model_and_data, get_similar_movies


# Instantiate Blueprint object for appliction 
application_routes = Blueprint('application_routes', __name__)

# Load model and data once at startup
model_data = load_model_and_data()



@application_routes.route('/')
def home():
    # Creates the home page from index.html 
    return render_template("index.html", movies= sorted( list(model_data["movie_names"].values()) ))  


@application_routes.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get("movie")
    return render_template("recommend.html", 
                            movie= request.form.get("movie"),  
                            recommendations= get_similar_movies(selected_movie, model_data)   
                        )
