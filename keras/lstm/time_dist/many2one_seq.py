"""
MANY to ONE
https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
"""

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
# The input for LSTMs must be three dimensional.
# We can reshape the 2D sequence into a 3D sequence with 1 sample, 5 time steps, and 1 feature.
#1 sample, 5 time steps, and 1 feature
X = seq.reshape(1, length, 1)
print("X:", X, "\n", "X.shape:", X.shape)
"""
X: [[
  [0. ]
  [0.2]
  [0.4]
  [0.6]
  [0.8]
]] 
 X.shape: (1, 5, 1) """

# We will define the output as 1 sample with 5 features.
y = seq.reshape(len(seq), 1)
print("y:", y, "\n", "y.shape:", y.shape)
"""
y: [
 [0. ]
 [0.2]
 [0.4]
 [0.6]
 [0.8]] 
 y.shape: (5, 1)"""

# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500

# We will define the model as having one input with 5 time steps.
# The first hidden layer will be an LSTM with 5 units.
# The output layer is a fully-connected layer with 5 neurons.
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=0)

# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)