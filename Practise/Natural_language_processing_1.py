import tensorflow as tf 
import tensorflow_datasets as tfds 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences


#import the data 
imdb, info = tfds.load("imdb_reviews/subwords8k", with_inf= True, as_supervised= True)
train_data , c = imdb['train'] , imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
	training_sentences.append(s.numpy().decode('utf8'))
	training_labels.append(l.numpy())

for s, l in test_data:
	testing_sentence.append(s.numpy().decode('utf8'))
	training_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


#traing the tokenizer 
vocab_size = 10000
max_length = 120  # can try different value 100 
#padding_type = 'post'
trunc_type = 'post'
cov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token =cov_tok)
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(sequences, maxlen = max_length)

#model 
embedding_dim = 16 #try different 32, 64 

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
	tf.keras.layers.Flattern(), 
	tf.keras.layers.Dense(6, activation='relu'), #try different , not improving result 
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics =['accuracy'])

model.summary()

#training 
num_epochs = 10
model.fit(training_padded, training_labels_final, epochs = num_epochs, validation_data = (testing_padded, testing_labels_final))

#Investgation 
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)






