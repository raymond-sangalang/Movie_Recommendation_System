import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import Embedding, Parameter, MSELoss
from tqdm import trange   
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from RatingsLoader import RatingsLoader #, movies_df
import os



# Initialize Model - Matrix Factorization
class MovieRatingsModel(torch.nn.Module):
    def __init__(self, num_users, num_movies, num_factors=32):
        super().__init__()

        self.user_factors = Embedding(num_users, num_factors)  # factors
        self.movie_factors = Embedding(num_movies, num_factors)
        self.user_bias = Embedding(num_users, 1)              # bias terms
        self.movie_bias = Embedding(num_movies, 1)
        self.global_bias = Parameter(torch.zeros(1))

        # Initialize the weights uniformly
        for param in self.parameters():
            torch.nn.init.uniform_(param, 0, 0.05)

    def forward(self, data):
        users, movies = data[:, 0], data[:, 1]
        dot = (self.user_factors(users) * self.movie_factors(movies)).sum(1)
        return dot + self.user_bias(users).squeeze() + self.movie_bias(movies).squeeze() + self.global_bias





def saveModel(model, optimizer, epoch, path_to_checkpoint, latest_checkpoint):
    # add update and save checkpoints
    filepath_checkpoint = os.path.join(path_to_checkpoint, f"epoch{epoch+1}.pth")

    torch.save({                  # every checkpoint
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

        }, filepath_checkpoint
    )

    torch.save({                  # the most current model
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

        }, latest_checkpoint
    )
    print(f"Saved Epoch checkpoint #{epoch+1}\n\tlocation: {filepath_checkpoint}\n")




# Training Function 
def trainModel(model, ratings_df, period= 5, path_to_checkpoint= "../saved/"):

    os.makedirs(path_to_checkpoint, exist_ok=True)

    # Training Configurations
    start_epochs = 0
    num_epochs = 64                                     
    batch_size = 128  # 1024                                   
    lr = 1e-3                                           # Learning rate
    reg_strength = 1e-5                                 # Regularization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))    # Set mixed precision


    loss_fn = MSELoss()                                      # Mean Squared Error (MSE) loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ADAM optimizier

    # Load Data
    train_set = RatingsLoader(ratings_df)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    # check if a saved model exists
    latest_checkpoint = os.path.join(path_to_checkpoint, "movie_ratings_model.pth")
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epochs = checkpoint['epoch'] + 1
        print(f"Resuming previous models training at epoch# {start_epochs}.\n")


    # Train Model
    for epoch in trange(start_epochs, num_epochs, desc="Training Model"):
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                
                outputs = model(x)
                mse_loss = loss_fn(outputs.squeeze(), y.float())
                reg_loss = reg_strength * (
                    model.user_factors.weight.norm(2) + model.movie_factors.weight.norm(2)
                )
                loss = mse_loss + reg_loss

            scaler.scale(loss).backward()   # scaled backward pass
            scaler.step(optimizer)          # step optimizer
            scaler.update()

            losses.append(loss.item())

        print(f"Epoch {epoch+1:03d} | Loss: {sum(losses)/len(losses):.4f}")
        # Save checkpoints periodically
        if (epoch + 1) % period == 0:
            saveModel(model, optimizer, epoch, path_to_checkpoint, latest_checkpoint)

    print("\nTraining is complete.")
    return train_set



# Evaluation Function
def evaluate_model(model, ratings_df, train_set, movie_names, cuda=False, test_size=0.2, show_plots=True):
    """ Evaluate the trained model on a test split using RMSE, show sample predictions,
        and visualize prediction errors.                                                """
   
    # Split data into two data frames
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    test_df = test_df.copy()

    # Map the desired test IDs to internal indices
    test_df["userId"] = test_df["userId"].map(train_set.user_to_index)
    test_df["movieId"] = test_df["movieId"].map(train_set.movie_to_index)
    test_df = test_df.dropna()

    # Convert to tensors
    x_test = torch.tensor(test_df[["userId", "movieId"]].values, dtype=torch.long)
    y_test = torch.tensor(test_df["rating"].values, dtype=torch.float32)

    if torch.cuda.is_available():
        model = model.cuda()
        x_test, y_test = x_test.cuda(), y_test.cuda()
        

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(x_test).squeeze()

    preds_np = preds.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_test_np, preds_np))

    print(f"\nRMSE on test set: {rmse:.4f}\n" 
        +"\nSample predictions (Actual vs Predicted):\n")


    # Sample Predictions 
    for index in np.random.choice(len(test_df), 5, replace=False):  

        # obtain the random sample from the test dataframe
        user_id = int(test_df.iloc[index]["userId"])
        movie_id = int(test_df.iloc[index]["movieId"])

        # Utilize the MovieRatingsModel's inherited methods 
        actual = y_test_np[index]
        predicted = preds_np[index]

        # Utilize the RatingsLoader class and obtain
        real_user_id = train_set.index_to_user.get(user_id, user_id)
        real_movie_id = train_set.index_to_movie.get(movie_id, movie_id)
        movie_title = movie_names.get(int(real_movie_id), "Unknown")

        print(f"User {real_user_id:<4} | Movie: {movie_title:<40} | Actual: {actual:.1f} | Predicted: {predicted:.2f}")

    # Illustrate histogram plots 
    if show_plots:

        errors = preds_np - y_test_np

        # Histogram of prediction errors
        # - this will provide the performance of the model
        #   such that the denser (more frequent or populated) values are to Zero, the accuracy of the model would be higher.
        #   And vice versa
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=30, edgecolor='black')
        plt.title("Distribution of Prediction Errors")          # 
        plt.xlabel("Error in Prediction := (Predicted - Actual)")     # BIAS
        plt.ylabel("Frequency (1 / Period)")
        plt.grid(alpha=0.3)
        plt.show()

        # Scatter plot: Actual vs Predicted
        #
        #
        #
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test_np, preds_np, alpha=0.4)
        plt.title("Actual vs Predicted Ratings")
        plt.xlabel("Actual Rating")
        plt.ylabel("Predicted Rating")
        plt.plot([0, 5], [0, 5], color='blue', linestyle='--')  
        plt.grid(alpha=0.3)
        plt.show() 
 
    return rmse



def viewMovieClusters(kmeans, ratings_df, movie_names, num_clusters=10):

    print("\nMovie Clusters (Top ten movies by rating count for every cluster):\n")
    for cluster in range(num_clusters):
       
        print(f"\nCluster #{cluster + 1}\n{'-'*10}")
        movie_cluster = []
        # Find movie indices belonging to the current cluster
        # and obtain ratings associated to movies
        for movie_index in np.where(kmeans.labels_ == cluster)[0]:
          
            movieid = train_set.getMovieByIndex(movie_index)
            num_ratings = len(ratings_df.loc[ratings_df['movieId'] == movieid])
            movie_cluster.append((movie_names[movieid], num_ratings))

        # Sort movies by rating count in descending order, then print the top 10 movies in cluster
        for movie_title, _ in sorted(movie_cluster, key=lambda tup: tup[1], reverse=True)[:10]:
            print(f"\t{movie_title}")




if __name__ == "__main__":
    
    # convert the datasets to a pandas dataframe
    movies_df = pd.read_csv('../data/movies.csv')
    ratings_df = pd.read_csv('../data/ratings.csv')

    num_users = len(ratings_df.userId.unique())
    num_movies = len(ratings_df.movieId.unique())

    # Initialize model
    model = MovieRatingsModel(num_users, num_movies, num_factors=32)
    
    # Check if machine has a GPU to use with model
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()


    # Train the model
    train_set = trainModel(model, ratings_df= ratings_df)


    # Save the model
    torch.save(model.state_dict(), "movie_ratings_model.pth")

    trained_movie_embeddings = model.movie_factors.weight.data.cpu().numpy()
    print(f'Number of unique movie factor weights: {len(trained_movie_embeddings)}')
 
    # Fit the clusters based on the movie weights
    kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)
    movie_names = movies_df.set_index('movieId')['title'].to_dict()       # Movie ID to movie name mapping

    #  Using the evaluation_model, rsme, and plots
    rmse = evaluate_model(model, ratings_df, train_set, movie_names, cuda=cuda)


    # 
    viewMovieClusters(kmeans, ratings_df, movie_names, num_clusters=10)