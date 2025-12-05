""" routes.py - Initializes user routes as well as rendering templates. """
from flask import Blueprint, render_template, request, jsonify
from recommender_utils import load_model_and_data, get_similar_movies, get_user_recommendations, fetch_poster
from reviews_db import add_review, get_movie_reviews, get_average_rating, user_has_reviewed


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


store_reviews = {}       # { movie_id: [ { "review": str, "rating": int } ] }



@application_routes.route("/api/reviews/<int:movie_id>", methods=["GET", "POST"])
def reviews_api(movie_id):
    """ Using JSON API for movie reviews, such that:
        - GET:  returns all of a movies reviews and average rating
        - POST: appends a new review                                """

    if request.method == "POST":
        data = request.get_json() or {}

        user_id = data.get("user_id")
        rating = data.get("rating")
        review_text = data.get("review") or ""
        review_text = review_text.strip()

        # Condition to validate the data
        if not user_id or rating is None:
            return jsonify({"error": "user_id and rating are required"}), 400

        elif user_has_reviewed(int(user_id), movie_id):
            return jsonify({"error": "You have already submitted a review for this movie"}), 400


        try:
            # Datatype validation and check range in rating
            rating = float(rating)
           
            if rating < 1 or rating > 5:
                return jsonify({"error": "rating must be between 1 and 5"}), 400

        except ValueError:
            return jsonify({"error": "rating must be a number"}), 400

        # Store in the DataBase and return successful  
        add_review(user_id= int(user_id), movie_id= movie_id, rating= rating, review_text= review_text)
        return jsonify({"status": "ok"}), 201

    else:

        # request.method == "GET"
        # return all reviews and the average rating
        return jsonify({
            "movieId":          movie_id,
            "average_rating":   get_average_rating(movie_id),
            "reviews":          get_movie_reviews(movie_id)
        })



@application_routes.route("/movie/<int:movie_id>")
def movie_details(movie_id):

    movie_names = model_data["movie_names"]  # {movieId: title}
    train_set = model_data["train_set"]      # RatingsLoader ***
    
    # obtain the movie given by its ID
    title = movie_names.get(movie_id, "Unknown movie")
    overview = None   # TMDb usage
    genres = None

    user_ids = sorted(train_set.user_to_index.keys())
    default_user_id = user_ids[0] if user_ids  else 1       # use for authentication

    return render_template(
        "movie_details.html",
        movie_id= movie_id,
        movie_title= title,
        poster_url= fetch_poster(title),
        overview= overview,
        genres= genres,
        user_id= default_user_id,
        user_ids= user_ids,
    )
