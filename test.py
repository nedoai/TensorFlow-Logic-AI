import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model

model = load_model(r"Logic_v1/transformer_logic.tf")

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

# Преобразование текстовых данных в последовательности чисел
input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)

# Подгонка последовательностей до одинаковой длины
max_seq_length = max(max(map(len, input_sequences)), max(map(len, output_sequences)))

# Пример взаимодействия с моделью
input_text = "На полу разбитая игрушка"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
output_sequence = model.predict(input_sequence)

# Декодирование вывода модели в текст
decoded_output = tokenizer.sequences_to_texts([np.argmax(output_sequence[0], axis=-1)])[0]
print("Вывод:", decoded_output)