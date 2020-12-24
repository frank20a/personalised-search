from elasticsearch import Elasticsearch, helpers
import json
from typing import *
from functools import partial

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


def get_usr_rating(movieID, userID):
    for rate in get_usr_ratings(userID):
        if rate['_source']['movieID'] == movieID: return rate['_source']['rating']
    return None


def get_avg_rating(movieID):
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


def search_BM25(q: str, size: int = 10000) -> Tuple[list, float]:
    res = es.search(index='movies', body={
        'query': {
            'multi_match': {
                'query': q,
                'type': 'best_fields',
                'fields': ['title', 'genres', 'year']
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


def personalized_search(query, userID, limit=10):
    res, max_BM25 = search_BM25(query)
    max_usr = max_avg = -float('inf')
    for movie in res:
        movie['usr_score'] = get_usr_rating(movie['id'], userID)
        if movie['usr_score'] is not None and movie['usr_score'] > max_usr: max_usr = movie['usr_score']

        movie['avg_score'] = get_avg_rating(movie['id'])
        if movie['avg_score'] > max_avg: max_avg = movie['avg_score']

    for movie in res:
        movie['normalized_score'] = movie_score_avg(movie, max_BM25, max_usr, max_avg)

    return sorted(res, key=lambda m: m['normalized_score'], reverse=True)[:min(len(res), limit)]


if __name__ == '__main__':
    for i in personalized_search('Star Wars', 220):
        print(i)
