import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

# Read dataset from the directory
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    'data/trainData', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'data/trainData', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

# Visualize data
# for text_batch, label_batch in train_ds.take(1):
#   for i in range(5):
#     #print(label_batch[i].numpy(), text_batch.numpy()[i])

# Configure dataset for training
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Vocabulary size and number of words in a sequence.
vocab_size = 1000
sequence_length = 100

# TEXT PROCESSING (TEXT VECTORIZATION)
vectorize_layer = TextVectorization(
    #standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# BUILD YOUR VOCABULARY
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

############# Use for testing only. Not for production use #############
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])
########################################################################

# Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

model.summary()

# out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
# out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

# for index, word in enumerate(vocab):
#  if index == 0:
#    continue  # skip 0, it's padding.
#  vec = weights[index]
#  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
#  out_m.write(word + "\n")
# out_v.close()
# out_m.close()