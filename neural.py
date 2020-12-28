import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from user import load_users, movies, genres
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def pre_process(s: str):
    return re.sub(r"\([^()]*\)", "", s)


def get_title_tokens(movie_title, vocab_mul: float = 3):
    # # ==== find the length of the biggest sentence
    # most_words_sentence = max(movies['title'].tolist(), key=lambda n: len(pre_process(n).split(' ')))
    # print(len(pre_process(most_words_sentence).split(' ')), pre_process(most_words_sentence))     # => 15
    # # ==== find size of vocabulary
    # voc = []
    # for title in movies['title'].tolist():
    #     for word in pre_process(title).split(' '):
    #         if word not in voc: voc.append(word)
    # print(len(voc))     # => 8762
    # tokenize
    vocab_size = int(8762 * vocab_mul)
    max_sen_length = 15
    encoded_docs = [one_hot(pre_process(d), vocab_size, lower=True) for d in movie_title]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_sen_length, padding='post')

    return padded_docs, vocab_size, max_sen_length


class FrankNet(KerasRegressor):
    def __init__(self, vocab_size, input_len):
        super().__init__(build_fn=lambda: self.get_model(vocab_size, input_len), epochs=100, batch_size=5)

    @staticmethod
    def get_model(vocab_size, input_len):
        model = keras.Sequential()
        model.add(layers.Embedding(vocab_size, 8, input_length=input_len))
        model.add(layers.Flatten())
        model.add(layers.Dense(60, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(6, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        print('\n\n')
        model.summary()

        return model


if __name__ == '__main__':
    users = load_users()
    # print(users[352].movie_ratings)
    # print(movies[movies['movieId'] == 1])

    GENRES_ONE_HOT = []
    IN = []
    OUT = []

    # one-hot embed movie genres
    for movieID in users[352].movie_ratings:
        temp = {}
        for key in genres[:-1]:
            temp[key] = 1 if key in movies[movies['movieId'] == movieID]['genres'].to_list()[0] else 0
        GENRES_ONE_HOT.append(list(temp.values()))
        OUT.append(users[352].movie_ratings[movieID])

    # get tokenized titles
    for movieID in users[352].movie_ratings:
        IN.append(movies[movies['movieId'] == movieID]['title'].to_list()[0])
    IN, vocab_size, max_sen_length = get_title_tokens(IN)

    # model = FrankNet(vocab_size, max_sen_length)
    estimator = FrankNet(vocab_size, max_sen_length)
    estimator.fit(IN, np.array(OUT))
    while True:
        token, _, _ = get_title_tokens([input("Predict for user 353: ")])
        print(estimator.predict(token))
