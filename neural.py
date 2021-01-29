import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import re
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from os import getcwd, path
import pickle

vectors = None
try:
    with open('bin/vocabulary.pickle', 'rb') as file:
        vectors = pickle.load(file)
        print('vocabulary loaded from pickle')
except FileNotFoundError:
    raise Exception('Vocabulary not found')



# ========== Code for creating vocabulary pickle ==========
# def load_vectors():
#     # Load word vectors from GloVe
#     global vectors
#     try:
#         vectors = KeyedVectors.load(path.join(getcwd(), 'bin/glove.wv'))
#     except FileNotFoundError:
#         tmp_file = get_tmpfile(path.join(getcwd(), 'bin/temp_word2vec.txt'))
#         _ = glove2word2vec(datapath(path.join(getcwd(), 'bin/glove.6B.100d.txt')), tmp_file)
#         vectors = KeyedVectors.load_word2vec_format(tmp_file)
#         print(type(vectors))
#         vectors.save(path.join(getcwd(), 'bin/glove.wv'))
#
#         print('word vectors loaded from text and save to word2vec .wv')


def pre_process(s: str):
    return re.sub(r'[^A-Za-z0-9 ]+', '', re.sub(r'-', ' ', re.sub(r"\([^()]*\)", "", s))).lower()


def get_title_vector(title):
    vector = np.zeros(100)
    for word in pre_process(title).split():
        try:
            vector += vectors[word]
        except KeyError:
            #print('skipping', word, 'is unknown word :(')
            pass
    return list(vector)


# ========= DEPRECATED =========
# def get_title_vector(title, wv):
#     vector = []
#     for word in pre_process(title).split():
#         vector.append(wv[word])
#     return list(vector)
#
#
# def get_title_tokens(movie_title, vocab_mul: float = 3):
#     # # ==== find the length of the biggest sentence
#     # most_words_sentence = max(movies['title'].tolist(), key=lambda n: len(pre_process(n).split(' ')))
#     # print(len(pre_process(most_words_sentence).split(' ')), pre_process(most_words_sentence))     # => 15
#     # # ==== find size of vocabulary
#     # voc = []
#     # for title in movies['title'].tolist():
#     #     for word in pre_process(title).split(' '):
#     #         if word not in voc: voc.append(word)
#     # print(len(voc))     # => 8762
#     # tokenize
#     vocab_size = int(8762 * vocab_mul)
#     max_sen_length = 15
#     encoded_docs = [one_hot(pre_process(d), vocab_size, lower=True) for d in movie_title]
#     padded_docs = pad_sequences(encoded_docs, maxlen=max_sen_length, padding='post')
#
#     return padded_docs, vocab_size, max_sen_length
#
#
# def get_word2vec_model(movies=None):
#     try:
#         model = Word2Vec.load('./bin/Word2Vec.model')
#     except FileNotFoundError:
#         data = []
#         for n, title in movies['title'].iteritems():
#             data.append(pre_process(title).split())
#         model = Word2Vec(data, min_count=1, size=100, window=5, workers=4)
#         model.save('./bin/Word2Vec.model')
#
#     return model
#
#
# def get_word2vec_vectors(movies=None):
#     try:
#         vectors = KeyedVectors.load('./bin/Word2Vec.wordvectors')
#     except FileNotFoundError:
#         model = get_word2vec_model(movies)
#         vectors = model.wv
#         vectors.save('./bin/Word2Vec.wordvectors')
#
#     return vectors

class FrankNet(KerasRegressor):
    def __init__(self, userID):
        super().__init__(build_fn=self.get_model, epochs=50, batch_size=5)
        self.userID = userID

    @staticmethod
    def get_model():
        model = keras.Sequential()
        model.add(layers.Dense(80, input_dim=119, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(18, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        print('\n\n')
        model.summary()

        return model

    def fit(self, x, y, *args, **kwargs):
        try:
            self.model = load_model('bin/user_models/m' + str(self.userID) + '.h5')
            print('model loaded from h5')
        except OSError:
            super().fit(x, y, *args, **kwargs)
            self.model.save('bin/user_models/m' + str(self.userID) + '.h5')
            print('\n', '=' * 60, '\n')


# ========== Code for creating vocabulary pickle ==========
# if __name__ == '__main__':
#     from user import load_users, movies, genres
#     load_vectors()
#     vocabulary = {}
#     for title in movies['title']:
#         for word in pre_process(title).split():
#             try:
#                 if word not in vocabulary: vocabulary[word] = vectors[word]
#             except KeyError:
#                 print('skipping', word, 'is unknown word :(')
#     with open('bin/vocabulary.pickle', 'wb') as file:
#         pickle.dump(vocabulary, file)
#         print('dumped vocab in pickle')

if __name__ == '__main__':
    from user import load_users, movies, genres

    users = load_users()
    user = 352
    # print(users[352].movie_ratings)
    # print(movies[movies['movieId'] == 1])

    IN = []
    OUT = []

    # one-hot embed movie genres
    for movieID in users[user].movie_ratings:
        temp = []
        for genre in genres[:-1]:
            temp.append(1 if genre in movies[movies['movieId'] == movieID]['genres'].to_list()[0] else 0)
        OUT.append([users[user].movie_ratings[movieID]])
        IN.append(get_title_vector(movies[movies['movieId'] == movieID]['title'].to_list()[0]) + temp)

    # Split data for validation
    x_train, x_test, y_train, y_test = train_test_split(IN, OUT)

    estimator = FrankNet(user)
    estimator.fit(x_train, y_train, verbose=2)
    print('\n', '='*60, '\n')

    predictions = estimator.predict(x_test)
    MAE = 0
    MSE = 0
    for i in range(len(x_test)):
        MAE += abs(predictions[i] - y_test[i][0])
        MSE += (predictions[i] - y_test[i][0])**2
    MAE /= len(x_test)
    MSE /= len(x_test)
    print('MAE =', MAE)
    print('MSE =', MSE)

    print('\n', '='*60, '\n')