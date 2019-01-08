from keras_attention.attention_decoder import AttentionDecoder
from keras import Sequential
from keras.layers import LSTM
from data_generator import get_pair, one_hot_decode
import numpy as np


n_timesteps_in= 5
n_timesteps_out = 2
input_dim = 30
encoded_dim = 100
n_epochs = 5000

model = Sequential()
model.add(LSTM(encoded_dim, input_shape=(n_timesteps_in, input_dim), return_sequences=True))
model.add(AttentionDecoder(encoded_dim, input_dim))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])


for epoch in range(n_epochs):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, input_dim)
    model.fit(X, y, epochs=1, verbose=2)

total = 100
correct = 0
for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, input_dim)
    yhat = model.predict(X, verbose=0)
    if np.array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct+=1

print(float(correct)/float(total))