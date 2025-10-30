import torch
import pandas as pd
from MovieRatingsModel import MovieRatingsModel, evaluate_model
from RatingsLoader import RatingsLoader


if __name__ == "__main__":
	
	path_to_model = "movie_ratings_model.pth"
	movies_df = pd.read_csv('../data/movies.csv')
	#
	ratings_df = pd.read_csv('../data/ratings.csv')
	num_users = len(ratings_df.userId.unique())
	num_movies = len(ratings_df.movieId.unique())
	#
	model = MovieRatingsModel(num_users= num_users, num_items= num_movies, num_factors= 32)

	# Load trained weights - load and set to eval mode
	model.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))
	model.eval()
	print(f"Loaded model from file path: {path_to_model}")

	# Mapping dataset for evaluation
	train_set = RatingsLoader(ratings_df)
	#
	movie_names = movies_df.set_index('movieId')['title'].to_dict()

	# Evaluate the loaded model
	evaluate_model(model, ratings_df, train_set, movie_names, cuda=False)
