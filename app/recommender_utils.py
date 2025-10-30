""" recommender_utils.py - File will handle loading the trained model, embeddings, and logic for recommendations """
import os
import torch
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# 
import sys
import os
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
if target_dir not in sys.path:
    sys.path.insert(0, target_dir) 

from MovieRatingsModel import MovieRatingsModel

# Load API key for movie personalization from .env
load_dotenv()  
TMDB_API_KEY = os.getenv("TMDB_API_KEY")



def load_model_and_data():
    """Loads model, embeddings, and movie titles."""

    # obtain the movies and ratings data and initialize the dataframes
    movies_df = pd.read_csv('../data/movies.csv')
    ratings_df = pd.read_csv('../data/ratings.csv')

    # Get the count of elements in both dataframes
    num_users = len(ratings_df.userId.unique())
    num_movies = len(ratings_df.movieId.unique())

    # Creating the model object and use pytorch to load the latest checkpoint
    #  and set the mode of the model prepared for evaluation.
    model = MovieRatingsModel(num_users, num_movies, num_factors=32)
    checkpoint = torch.load('../saved/movie_ratings_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    

    embeddings = model.movie_factors.weight.data.cpu().numpy()
    movie_names = movies_df.set_index('movieId')['title'].to_dict()
    return {"model": model, "embeddings": embeddings, "movie_names": movie_names}



def fetch_poster(title):
    """ Fetch the poster url from TMDb website. """
    # Use request and response objects to communicate with a website, 
    #  and return the movie poster path if data is present
   
    title_name = title.split('(')[0].strip() 
    url = "https://api.themoviedb.org/3/search/movie?" + f"api_key={TMDB_API_KEY}&query={title_name}"

    try:
        response = requests.get(url)
        data = response.json()

        if data["results"]:
            return f"https://image.tmdb.org/t/p/w200{data["results"][0]["poster_path"]}"     
    except Exception:
        pass
    return "/static/images/movies.png"  # default movie image



def get_similar_movies(movie_title, model_data, top_n=5):
    """ Find top-N similar movies with posters. """
    
    movie_names = model_data["movie_names"]
    embeddings = model_data["embeddings"]

    # Obtain the unique movie id by searching for the given movie title
    movie_id = next((_movieID for _movieID, _movieTitle in movie_names.items() if _movieTitle == movie_title), None)
    if not movie_id:   
        return []

    movie_index = list(movie_names.keys()).index(movie_id)
    similarities = cosine_similarity([embeddings[movie_index]], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]

    similar_movies = []

    for index in top_indices:
        title = movie_names[list(movie_names.keys())[index]]
        poster_url = fetch_poster(title)
        similar_movies.append({"title": title, "poster": poster_url})

    return similar_movies
