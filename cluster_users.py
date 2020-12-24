import pandas as pd
import json
import pickle


def avg(x):
    if len(x) == 0: return None
    return sum(x)/len(x)

# Import movie csv into pandas
movies = pd.read_csv('./movies.csv', dtype={'movieId': 'Int64'})
movies['genres'] = movies['genres'].str.split('|')
# Import ratings csv into pandas
ratings = pd.read_csv('./ratings.csv', dtype={'userId': 'Int64', 'movieId': 'Int64', 'timestamp': 'Int64'})

# Create or import list of genres that appear in movies
try:
    with open('genres.json', 'r', encoding='utf-8') as file:
        genres = json.load(file)
        print('genres loaded from JSON')
except FileNotFoundError:
    genres = []
    for index, row in movies.iterrows():
        for genre in row['genres']:
            if genre not in genres: genres.append(genre)
    with open('genres.json', 'w', encoding='utf-8') as file:
        json.dump(genres, file)

# Create or import dict with average genre rating
try:
    with open('avg_ratings_per_genre.json', 'r', encoding='utf-8') as file:
        avg_ratings_per_genre = json.load(file)
        print('avg_ratings_per_genre loaded from JSON')
except FileNotFoundError:
    avg_ratings_per_genre = {}
    for key in genres: avg_ratings_per_genre[key] = []

    for i, rating in ratings.iterrows():
        for genre in movies[movies['movieId'] == rating['movieId']]['genres'].values[0]:
            avg_ratings_per_genre[genre].append(rating['rating'])

    for genre in avg_ratings_per_genre:
        avg_ratings_per_genre[genre] = avg(avg_ratings_per_genre[genre])

    with open('avg_ratings_per_genre.json', 'w', encoding='utf-8') as file:
        json.dump(avg_ratings_per_genre, file)


class User():
    def __init__(self, userID):
        self.userID = userID

        # Get user ratings per genre
        self.genre_ratings = {}
        for key in genres: self.genre_ratings[key] = []
        for i, rating in ratings[ratings['userId'] == self.userID].iterrows():
            for genre in movies[movies['movieId'] == rating['movieId']]['genres'].values[0]:
                self.genre_ratings[genre].append(rating['rating'])

        # Get user average rating per genre
        self.genre_avgs = {}
        for genre in self.genre_ratings:
            self.genre_avgs[genre] = avg(self.genre_ratings[genre])

        # Pre-process for K-Means
        self.genre_avgs_prefilled = self.genre_avgs.copy()
        for genre in self.genre_avgs_prefilled:
            if self.genre_avgs_prefilled[genre] is None:
                # Fill missing ratings with mean of all ratings
                self.genre_avgs_prefilled[genre] = avg_ratings_per_genre[genre]


u = User(2)
print(u.genre_avgs)