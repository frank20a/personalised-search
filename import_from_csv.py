from elasticsearch import Elasticsearch, helpers
import csv
import re


def import_movies():
    es = Elasticsearch()

    with open('bin/movies.csv', encoding='utf-8') as datafile:
        datareader = csv.DictReader(datafile, delimiter=',', quotechar='"')
        movies = []
        for row in datareader:
            movie = {'_index': 'movies', '_type': 'movie', '_id': int(row['movieId']),
                     'genres': row['genres'].split('|') if row['genres'] != '(no genres listed)' else []}

            regex = re.search(r'(.*) \((.*)\)', row['title'], re.M)
            if regex:
                movie['title'] = regex.group(1)
                movie['year'] = regex.group(2)
            else:
                movie['title'] = row['title']
                movie['year'] = ''

            movies.append(movie)

    helpers.bulk(es, movies)


def import_ratings():
    es = Elasticsearch()

    with open('bin/ratings.csv', encoding='utf-8') as datafile:
        datareader = csv.DictReader(datafile, delimiter=',', quotechar='"')
        ratings = []
        for n, row in enumerate(datareader):
            rating = {'_index': 'ratings', '_type': 'rating', '_id': int(n), 'userID': int(row['userId']),
                      'movieID': int(row['movieId']), 'rating': float(row['rating']),
                      'timestamp': int(row['timestamp'])}

            ratings.append(rating)

    #print(ratings[0])
    helpers.bulk(es, ratings)


if __name__ == '__main__':
    import_movies()
    import_ratings()
