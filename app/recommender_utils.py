""" recommender_utils.py - File will handle loading the trained model, embeddings, and logic for recommendations """
import os
import sys
import torch
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Adding model directory to sys.path
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
if target_dir not in sys.path:
    sys.path.insert(0, target_dir) 

from MovieRatingsModel import MovieRatingsModel
from RatingsLoader import RatingsLoader

# Load (TMDB) API key for movie personalization from .env
load_dotenv()  
TMDB_API_KEY = os.getenv("TMDB_API_KEY")



def load_model_and_data():
    """ Loads trained model, embeddings, movie titles, and mappings. """

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

    # Use RatingsLoader to keep mappings consistent
    train_set = RatingsLoader(ratings_df)
    
    embeddings = model.movie_factors.weight.data.cpu().numpy()
    movie_names = movies_df.set_index('movieId')['title'].to_dict()
    return {
        "model": model, 
        "embeddings": embeddings, 
        "movie_names": movie_names,
        "train_set": train_set
    }



def fetch_poster(title):
    """ Fetch the poster url from TMDb website. """
    # Use request and response objects to communicate with a website, 
    #  and return the movie poster path if data is present
   
    title_name = title.split('(')[0].strip() 
    url = "https://api.themoviedb.org/3/search/movie?" + f"api_key={TMDB_API_KEY}&query={title_name}"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w200{poster_path}"


            return f"https://image.tmdb.org/t/p/w200{data["results"][0]["poster_path"]}"     
    except Exception:
        pass
    return "/static/images/movies.png"  # default movie image



def get_similar_movies(movie_title, model_data, top_number=5):
    """ Find top_number similar movies with posters. """
    
    movie_names = model_data["movie_names"]
    embeddings = model_data["embeddings"]

    # Obtain the unique movie id by searching for the given movie title
    movie_id = next((_movieID for _movieID, _movieTitle in movie_names.items() if _movieTitle == movie_title), None)
    if movie_id is None:   
        return []

    movie_index = list(model_data["train_set"].movie_to_index.values())[
        list(model_data["train_set"].movie_to_index.keys()).index(movie_id)
    ]
    similarities = cosine_similarity([embeddings[movie_index]], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_number + 1]


    similar_movies = []
    for index in top_indices:
        title = movie_names[list(movie_names.keys())[index]]
        poster_url = fetch_poster(title)
        similar_movies.append({"title": title, "poster": poster_url})
    return similar_movies





def get_user_recommendations(user_id, model_data, top_number=5):
    """ Given a user_id, return personalized movie recommendations based on
         the trained model. Ensures indices are correct and prevents IndexErrors. """
         
    model = model_data["model"]
    movie_names = model_data["movie_names"]
    train_set = model_data["train_set"]

    # Check if user id exists
    if user_id not in train_set.user_to_index:
        return []

    user_index = train_set.user_to_index[user_id]
    movie_indices = list(train_set.movie_to_index.values())
    num_movies = len(movie_indices)

    user_tensor = torch.LongTensor([user_index] * num_movies)
    movie_tensor = torch.LongTensor(movie_indices)
    data = torch.stack((user_tensor, movie_tensor), dim=1)

    # Predict ratings after transforming the tensors shape
    with torch.no_grad():
        predictions = model(data).cpu().numpy()

    # Top recommendations
    top_indices = np.argsort(predictions)[::-1][:top_number]
    reverse_index_to_movie = {v: k for k, v in train_set.movie_to_index.items()}

    recommendations = []
    for i in top_indices:
        movie_id = reverse_index_to_movie[movie_tensor[i].item()]
        title = movie_names.get(movie_id, "Unknown")
        poster_url = fetch_poster(title)
        recommendations.append({"title": title, "poster": poster_url})

    return recommendations

