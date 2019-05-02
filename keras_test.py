import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.preprocessing.text import Tokenizer



max_words = 1000
batch_size = 32
epochs = 5



from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# print(newsgroups_train["data"][0])


print("Preparing the Tokenizer...")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(newsgroups_train["data"])

print('Vectorizing sequence data...')
x_train = tokenizer.texts_to_matrix(newsgroups_train["data"], mode='binary')
x_test = tokenizer.texts_to_matrix(newsgroups_test["data"], mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print(x_train)

num_classes = np.max(newsgroups_train["target"]) + 1
print(num_classes, 'classes')

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(newsgroups_train["target"], num_classes)
y_test = keras.utils.to_categorical(newsgroups_test["target"], num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print(y_train)



print('Building model sequentially 1...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


print('Building model sequentially 2...')
model = Sequential([
          Dense(512, input_shape=(max_words,)),
          Activation('relu'),
          Dropout(0.5),
          Dense(num_classes),
          Activation('softmax')
        ])

print(model.layers)

print(model.to_yaml())

print('Building model functionally...')
a = Input(shape=(max_words,))
b = Dense(512)(a)
b = Activation('relu')(b)
b = Dropout(0.5)(b)
b = Dense(num_classes)(b)
b = Activation('softmax')(b)
model = Model(inputs=a, outputs=b)

from keras.models import model_from_yaml

yaml_string = model.to_yaml()

# model = model_from_yaml(yaml_string)



