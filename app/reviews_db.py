import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "reviews.db")


def get_connection():
    # Using sqlite3 to implement dictionary rows
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    return conn


def init_db():
    """ Create the reviews table and set the columns 
        associated with their datatypes, restrictions, behaviors """
   
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            userId      INTEGER NOT NULL,
            movieId     INTEGER NOT NULL,
            rating      REAL NOT NULL,
            review      TEXT,
            timestamp   TEXT NOT NULL,
            UNIQUE(userId, movieId)
        ) 
        """
    )

    conn.commit()
    conn.close()


def add_review(user_id, movie_id, rating, review_text):
    # add_review:

    conn = get_connection()
    cur = conn.cursor()

    ts = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO reviews (userId, movieId, rating, review, timestamp) 
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, movie_id, rating, review_text, ts)
    )

    conn.commit()
    conn.close()


def get_movie_reviews(movie_id):
    # get_movie_reviews: obtaining the movie_id
    # retrieve the rows containing column movieId equal to given movie_id
    
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, userId, movieId, rating, review, timestamp 
        FROM reviews 
        WHERE movieId = ? 
        ORDER BY timestamp DESC
        """,
        (movie_id,)
    )
    rows = cur.fetchall()

    conn.close()
    return [dict(row) for row in rows]


def get_average_rating(movie_id):
    # get_average_rating: Using aggregate function in query to obtain the AVG from the ratings column.

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT AVG(rating) AS avg_rating 
        FROM reviews 
        WHERE movieId = ?
        """, 
        (movie_id,)
    )
    row = cur.fetchone()
    
    conn.close()
    return float(row["avg_rating"]) if (row and row["avg_rating"] is not None)  else None
 

def user_has_reviewed(user_id, movie_id):
    """ user_has_reviewed: Return True if this user already has a review for this movie. """
    
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 1 FROM reviews 
        WHERE userId = ? AND movieId = ? 
        LIMIT 1
        """,
        (user_id, movie_id),
    )
    row = cur.fetchone()

    conn.close()
    return row is not None