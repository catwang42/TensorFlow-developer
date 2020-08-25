import tensorflow as tf 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

tokenizer = Tokenizer()

data = open('/tmp/irish-lyrics-eof.txt').read()

corpus = data.lower().split('\n')

tokenizer.fit_on_text(corpus)
totoal_words = len(tokenizer.word_index) +1 

input_sequences = []

#build ngram tokens 
for line in corpurs:
	token_list = tokenizer.texts_to_sequence([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

#pad the input sequence up to the max length 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequence = np.array(pad_sequences(input_sequence, maxlen = max_sequence_len, padding= 'pre'))

#build training, validation and labels 
xs , labels = input_sequence[:,:-1], input_sequence[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=500, verbose=1)



