from elasticsearch import Elasticsearch
from typing import *
from user import User, ratings, load_users
from cluster_users import cluster

inf = float('inf')

es = Elasticsearch()


def avg(A):
    if len(A) == 0: return 0
    return sum(A) / len(A)


def movie_score_avg(m, max_BM25, max_usr, max_avg, w_BM25=1, w_usr=1, w_avg=1):
    n_BM25 = w_BM25 * m['BM25_score'] / max_BM25
    n_avg = w_avg * m['avg_score'] / max_avg
    n_usr = 0 if m['usr_score'] is None else (w_usr * m['usr_score'] / max_usr)
    w = w_BM25 + w_avg + (0 if m['usr_score'] is None else w_usr)
    return (n_BM25 + n_avg + n_usr) / w


def get_usr_ratings(userID):
    return es.search(index='ratings', body={
        'query': {
            'match': {
                'userID': {
                    'query': userID,
                    'minimum_should_match': '100%'
                }
            }
        },
        'size': 10000
    })['hits']['hits']


def get_usr_rating_from_elastic(movieID, userID):
    for rate in get_usr_ratings(userID):
        if rate['_source']['movieID'] == movieID: return rate['_source']['rating']
    return None


def get_usr_rating(movie, user):
    # users have their ratings saved on the User class of user.py
    try:
        return float(user.movie_ratings[movie['id']]), False
    except KeyError:
        key = max(user.Cluster_score, key=lambda n: float(user.Cluster_score[n]) if n in movie['genres'] else -inf)
        return user.Cluster_score[key], True


def get_avg_rating_from_elastic(movieID):
    res = es.search(index='ratings', body={
        'query': {
            'match': {
                'movieID': {
                    'query': movieID,
                    'minimum_should_match': '100%'
                }
            }
        },
        'size': 10000
    })

    ratings = [rate['_source']['rating'] for rate in res['hits']['hits']]
    return avg(ratings)


def get_avg_rating(movie):
    # ratings is a pandas doc from user.py file
    return avg(ratings[ratings['movieId'] == movie['id']]['rating'])


def search_BM25(q: str, size: int = 10000) -> Tuple[list, float]:
    res = es.search(index='movies', body={
        'query': {
            'multi_match': {
                'query': q,
                'type': 'best_fields',
                'fields': ['title', 'genres', 'year'],
                'operator': 'AND',
                'fuzziness': 'AUTO'
            }
        },
        'size': size
    })

    A = []
    for n, hit in enumerate(res['hits']['hits']):
        A.append(hit['_source'])
        A[n]['id'] = int(hit['_id'])
        A[n]['BM25_score'] = hit['_score']
    return A, res['hits']['max_score']


def personalized_search(query: str, user: User, limit: int = 10):
    res, max_BM25 = search_BM25(query)
    user.recalculate_cluster_score()

    max_usr = max_avg = -inf
    for movie in res:
        movie['usr_score'], movie['usr_score_from_cluster'] = get_usr_rating(movie, user)
        if movie['usr_score'] > max_usr: max_usr = movie['usr_score']

        movie['avg_score'] = get_avg_rating(movie)
        if movie['avg_score'] > max_avg: max_avg = movie['avg_score']

    for movie in res:
        movie['normalized_score'] = movie_score_avg(movie, max_BM25, max_usr, max_avg,
                                                    w_usr=2 if movie['usr_score_from_cluster'] else 5)

    return sorted(res, key=lambda m: m['normalized_score'], reverse=True)[:min(len(res), limit)]


if __name__ == '__main__':
    users = load_users()
    cluster(users)
    for i in personalized_search('Star', users[219]):
        print(i)
