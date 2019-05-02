import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!') # создаем объект из TF
sess = tf.InteractiveSession() # создаем сессию
print(sess.run(hello)) #сессия "выполняет" объект