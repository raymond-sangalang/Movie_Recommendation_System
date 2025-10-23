import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd



class RatingsLoader(Dataset):
    """ PyTorch Dataset for user-movie ratings.
        Takes in a DataFrame and creates continuous IDs for users and movies. """
   
    def __init__(self, ratings_df):
        
        self.ratings = ratings_df.copy()   # avoid mutating or corrupting the original

        # Extract and map all user IDs and movie IDs to indices
        unique_users = self.ratings["userId"].unique()
        unique_movies = self.ratings["movieId"].unique()

    
        # Continuous ID for users and movies
        self.user_to_index = {_user: index for index, _user in enumerate(unique_users)}   
        self.index_to_user = {index: _user for index, _user in self.user_to_index.items()}

        self.movie_to_index = {_movie: index for index, _movie in enumerate(unique_movies, start=0)}
        self.index_to_movie = {index: _movie for _movie, index in self.movie_to_index.items()}


        # Mappings to dataset
        self.ratings["movieId"] = self.ratings["movieId"].map(self.movie_to_index)       
        self.ratings["userId"] = self.ratings["userId"].map(self.user_to_index)           

        # Create Features and the labels
        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values

        # Transforms the data to tensors 
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)



    def getIndexByUserID(self, user_id):
        return self.user_to_index[user_id]
    def getIndexByMovieID(self, movie_id):
        return self.movie_to_index[movie_id]

    def getUserIDByIndex(self, index):
        return self.index_to_user[index]
    def getMovieByIndex(self, index):
        return self.index_to_movie[index]


    def __getitem__(self, index):
        # return (self.x[index], self.y[index])
        user_id, movie_id = self.x[index]
        rating = self.y[index]
        return (
            torch.tensor(self.x[index], dtype= torch.long),
            rating if torch.is_tensor(rating) else torch.tensor(rating, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.ratings)
