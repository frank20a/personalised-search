from elasticsearch import Elasticsearch, helpers
import json
from typing import *
from functools import partial

es = Elasticsearch()


def avg(A): return sum(A) / len(A)


def movie_score_avg(max_BM25, max_usr, max_avg, m):
    if m['usr_score'] is None:
        return (m['BM25_score'] / max_BM25 + m['avg_score'] / max_avg) / 2
    else:
        return (m['BM25_score'] / max_BM25 + m['usr_score'] / max_usr + m['avg_score'] / max_avg) / 3


def get_usr_rating(movieID, userID):
    res = es.search(index='ratings', body={
        'query': {
            'match': {
                'userID': {
                    'query': userID,
                    'minimum_should_match': '100%'
                }
            }
        },
        'size': 10000
    })

    for rate in res['hits']['hits']:
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


def search_title_BM25(q: str, size: int = 10000) -> Tuple[list, float]:
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
        A[n]['id'] = hit['_id']
        A[n]['BM25_score'] = hit['_score']
    return A, res['hits']['max_score']


def personalized_search(query, userID):
    res, max_BM25 = search_title_BM25(query)
    max_usr = max_avg = -float('inf')
    for movie in res:
        print(get_usr_rating(movie['id'], userID))
        movie['usr_score'] = get_usr_rating(movie['id'], userID)
        if movie['usr_score'] is not None and movie['usr_score'] > max_usr: max_usr = movie['usr_score']

        movie['avg_score'] = get_avg_rating(movie['id'])
        if movie['avg_score'] > max_avg: max_avg = movie['avg_score']

    for movie in res:
        movie['normalized_score'] = movie_score_avg(max_BM25, max_usr, max_avg, movie)

    return sorted(res, key=lambda m: m['normalized_score'], reverse=True)


print(get_usr_rating(1, 154))
for i in personalized_search('Toy', 154):
    print(i)
