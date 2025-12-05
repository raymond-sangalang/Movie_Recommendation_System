# Movie Recommendation System

A Flask-powered web application that recommends movies based on similarity and personalized predictions using a trained model built with PyTorch.

---

## The system includes:

+ Movie similarity recommendations
+ Personalized user-based predictions
+ Star rating display
+ Dark mode toggle
+ User profile sidebar
+ Machine learning model for rating predictions
+ Modern UI with navigation bar & responsive design



## Interactive Features

### 1. Movie-Based Recommendations

* Select a movie from a dropdown list
* Get similar movies based on embedding cosine similarity
* TMDb poster integration
* Star-rating visual indicator (1–5 stars)

### 2. User-Based Personalized Recommendations

* Choose a user ID to view personalized movie predictions
* Sidebar displays user's profile info (ID, stats)
* Utilized trained PyTorch matrix factorization model to predict ratings

### 3. Toggle Dark/Light Mode

* Toggle dark mode using the navigation bar button

### 4. Navigation Bar

* Quick access to Home and User Recommendation pages
* Responsive and styled to match both themes

### 5. Movie Card User Interface

* Hovering animation
* Poster, title, star rating, numeric rating


---

# Project Structure

```
movie_recommendation/
├── app
│    ├── main.py
│    ├── recommender_utils.py
│    ├── reviews.db
│    ├── reviews_db.py
│    ├── routes.py
│    ├── static
│    │     ├── css
│    │     │     └── style.css
│    │     ├── images
│    │     │     ├── movie_seats.png
│    │     │     └── movies.png
│    │     └── js
│    │           └── reviews.js
│    └── templates
│          ├── index.html
│          ├── layout.html
│          ├── movie_details.html
│          ├── recommend.html
│          ├── user_recommend.html
│          └── user_select.html
├── data
│     ├── movies.csv
│     └── ratings.csv
├── .env
├── model
│     ├── EvaluateSavedModel.py
│     ├── movie_ratings_model.pth
│     ├── MovieRatingsModel.py
│     └── RatingsLoader.py
├── package.json
├── package-lock.json
├── README.md
├── requirements.txt
└── saved
      └── movie_ratings_model.pth

```

---

# Dependencies and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/raymond-sangalang/Movie_Recommendation_System.git
cd Movie_Recommendation_System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add TMDb API Key

Create a `.env` file in the `/app` directory:

```
TMDB_API_KEY=your_api_key_here
```

Get a key from: [https://www.themoviedb.org/](https://www.themoviedb.org/)

### 5. Run the App

```bash
cd app
python main.py
```

Then open:

➡ [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

# Machine Learning Model

### The system uses Matrix Factorization with PyTorch:

* Learns latent factors for users & movies
* Predicts user ratings with bias terms
* Trained on the MovieLens dataset (`ratings.csv`)
* Saved to `saved/movie_ratings_model.pth`

### Training is handled in:

```
MovieRatingsModel.py
```

### You can retrain using:

```python
python MovieRatingsModel.py
```

---


# Key Technologies

| Component | Technology                     |
| --------- | ------------------------------ |
| Backend   | Python, Flask                  |
| ML Model  | PyTorch (Matrix Factorization) |
| Frontend  | HTML, CSS, JavaScript, Jinja2  |
| Styling   | CSS, Flexbox                   |
| Data      | MovieLens ratings & metadata   |
| API       | TMDb movie poster API          |

---

# How the Recommendation System Works

#### Movie Similarity

* Computes cosine similarity between movie embeddings
* Displays top 5 most similar movies

#### User Recommendations

* Model predicts rating for each movie
* Top-N highest predicted scores returned
* Shown visually with star ratings

---

# Evaluation

`EvaluateSavedModel.py` computes:

* RMSE on test split
* Prediction scatter plots
* Error distribution histograms

---

# Acknowledgments

#### MovieLens dataset.
* Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM TiiS 5(4), 1–19. https://doi.org/10.1145/2827872

#### TMDb API Attributions. 
* This product uses the TMDB API but is not endorsed or certified by TMDB.



