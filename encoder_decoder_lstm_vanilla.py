from keras import Sequential
from keras.layers import TimeDistributed, Dense, LSTM, RepeatVector
from keras.losses import categorical_crossentropy
from data_generator import get_pair, one_hot_decode
import numpy as np

'''
The vanila lstm approach to predict the output sequence y from X is not enough
because it has a fixed length vector for representation the whole sequence
we need an attention mechanism that tells the LSTM where to attend
'''


n_timesteps_in= 5
n_timesteps_out = 2
input_dim = 30
encoded_dim = 100
n_epochs = 5000

model = Sequential()
model.add(LSTM(encoded_dim, input_shape=(n_timesteps_in, input_dim)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(encoded_dim, return_sequences=True))
model.add(TimeDistributed(Dense(input_dim, activation="softmax")))
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