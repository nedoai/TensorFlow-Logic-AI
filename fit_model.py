import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

training_data = []
with open(r"Logic_v1/logic.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            premise, conclusion = line.split("|")
            training_data.append((premise.strip(), conclusion.strip()))

inputs = [data[0] for data in training_data]
outputs = [data[1] for data in training_data]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + outputs)

input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)

max_seq_length = max(max(map(len, input_sequences)), max(map(len, output_sequences)))
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_seq_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
num_heads = 4
num_transformer_layers = 4

input_layer = tf.keras.layers.Input(shape=(max_seq_length,))
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(input_layer)
transformer_output = embedding_layer

for _ in range(num_transformer_layers):
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(query=transformer_output, value=transformer_output)
    add_attention = tf.keras.layers.Add()([attention_output, transformer_output])
    normalization_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention)
    feed_forward = tf.keras.layers.Dense(units=embedding_dim, activation='relu')(normalization_output)
    transformer_output = tf.keras.layers.Add()([feed_forward, normalization_output])

output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')(transformer_output)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(input_sequences, np.array(output_sequences), epochs=75)

input_text = "В округе ограбили магазин"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
output_sequence = model.predict(input_sequence)

decoded_output = tokenizer.sequences_to_texts([np.argmax(output_sequence[0], axis=-1)])[0]
print("Вывод:", decoded_output)

model.save("transformer_logic.tf")
