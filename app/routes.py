""" routes.py - Initializes user routes as well as rendering templates. """
from flask import Blueprint, render_template, request
from recommender_utils import load_model_and_data, get_similar_movies, get_user_recommendations


# Instantiate Blueprint object for appliction 
application_routes = Blueprint('application_routes', __name__)

# Load model and data once at startup
model_data = load_model_and_data()



@application_routes.route('/')
def home():
    # Creates the home page from index.html 
    return render_template("index.html", 
                            movies= sorted( list(model_data["movie_names"].values()) )
                        )  


@application_routes.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get("movie")

    return render_template("recommend.html", 
                            movie= selected_movie,  
                            recommendations= get_similar_movies(selected_movie, model_data)   
                        )


@application_routes.route('/user')
def user_select():
    """ User selection page — loads all user IDs from the dataset. """
    user_ids = sorted(model_data["train_set"].user_to_index.keys())

    return render_template('user_select.html', user_ids=user_ids)



@application_routes.route('/user_recommend', methods=['POST'])
def user_recommend():
    #
    selected_userID = int(request.form.get("user_id"))

    return render_template("user_recommend.html",
                           user_id= selected_userID,
                           recommendations= get_user_recommendations(selected_userID, model_data)  
                        )