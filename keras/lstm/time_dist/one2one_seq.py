"""
ONE to ONE
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
# We can reshape the 2D sequence into a 3D sequence with 5 samples, 1 time step, and 1 feature.
X = seq.reshape(len(seq), 1, 1)
print("X:", X, "\n", "X.shape:", X.shape)
"""
X: [
 [[0. ]]
 [[0.2]]
 [[0.4]]
 [[0.6]]
 [[0.8]]
] 
 X.shape: (5, 1, 1) """

# We will define the output as 5 samples with 1 feature.
y = seq.reshape(len(seq), 1)
print("y:", y, "\n", "y.shape:", y.shape)
"""
y: [
 [0. ]
 [0.2]
 [0.4]
 [0.6]
 [0.8]] 
 y.shape: (5, 1) """

# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000

# The network model as having 1 input with 1 time step.
# The first hidden layer will be an LSTM with 5 units.
# The output layer with be a fully-connected layer with 1 output.
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=0)

# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)