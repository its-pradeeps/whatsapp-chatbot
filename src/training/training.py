"""
This module is used to create the trained data for the chat script using the intents.
"""

import json
import pickle

import nltk

nltk.download('punkt')
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')

import pandas as pd

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from keras.layers import Input, Dense, Embedding, LSTM, Dropout
from keras.models import Model


class TrainData:
    """
    Class to Generate the training data
    """

    def __init__(self, intents):
        """

        :param intents: Training data
        """
        self.intents = intents
        self.embed_dim = 32
        self.lstm_dim = 64

    def get_tag_response(self):
        """

        :return:
        """
        tags = []
        xy = []
        for intent in self.intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for sentence in intent['patterns']:
                xy.append((sentence, tag))
        return tags, xy

    def get_dataframe(self):
        """

        :return:
        """
        tags, xy = self.get_tag_response()
        df = pd.DataFrame(xy, columns=['sentence', 'tag'])
        tag_map = dict()
        for index, tag in enumerate(tags):
            tag_map[tag] = index
        df['tag'] = df['tag'].map(tag_map)
        df = df.sample(frac=1)
        return df

    def get_stemmed_dataframe(self):
        """

        :return:
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        df = self.get_dataframe()
        df['stemmed_sentence'] = df['sentence'].apply(lambda sentence:
                                                      ' '.join(stemmer.stem(token)
                                                               for token in
                                                               nltk.word_tokenize(sentence) if
                                                               token not in stop_words))
        return df

    def get_training_xy(self):
        """

        :return:
        """
        tags, _ = self.get_tag_response()
        df = self.get_stemmed_dataframe()
        tags_cat = df['tag']
        labels = to_categorical(tags_cat, len(tags))

        inputs = df['stemmed_sentence']
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(inputs)
        sequences = tokenizer.texts_to_sequences(inputs)
        max_length = max(len(x) for x in sequences)
        paded_seq = pad_sequences(sequences, maxlen=max_length, padding='post')
        vocab = len(tokenizer.word_index) + 1

        with open('../tokenizer.pickle', 'wb') as outfile:
            pickle.dump(tokenizer, outfile)
        print('Tokenizer saved successfully')

        with open('../max_seq_length', 'wb') as outfile:
            pickle.dump(max_length, outfile)

        return paded_seq, labels, max_length, vocab

    def train_data(self, embed_dim=32, lstm_dim=64):
        """

        :param embed_dim:
        :param lstm_dim:
        :return:
        """
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        print(f'{embed_dim}\n{lstm_dim}')
        paded_seq, labels, max_length, vocab = self.get_training_xy()

        # Get the number of categories from the labels shape
        num_categories = labels.shape[1]
        print(f"Number of intent categories: {num_categories}")

        i = Input(shape=(max_length,))
        x = Embedding(vocab, self.embed_dim)(i)
        x = Dropout(0.5)(x)
        x = LSTM(self.lstm_dim)(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(num_categories, activation='softmax')(x)  # Use dynamic number of categories

        model = Model(i, x)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        r = model.fit(paded_seq, labels, epochs=100)

        model_name = "../ret_chatbot.h5"
        model.save(model_name)
        print(f"{model_name} saved successfully")

        return r


if __name__ == "__main__":
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    train_data_obj = TrainData(intents)
    history = train_data_obj.train_data(embed_dim=42, lstm_dim=120)
