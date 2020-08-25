import tensorflow as tf 
import tensorflow_datasets as tfds 

#import the data 
imdb, info = tfds.load("imdb_reviews/subwords8k", with_inf= True, as_supervised= True)
train_data , test_data = imdb['train'] , imdb['test']


#prep the data 

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

#Text(shape=(None,), dtype=tf.int64, encoder=<SubwordTextEncoder vocab_size=8185>)
tokenizer = info.features['text'].encoder

#build model 
embedding_dim = 64

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(6, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model.summary()

#training 
num_epochs = 10 

model.complie(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs = num_epochs, validation_data = test_dataset)


#visulisation 
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

