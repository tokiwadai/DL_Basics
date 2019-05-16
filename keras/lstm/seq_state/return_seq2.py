"""
https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/

Running the example returns a sequence of 3 values,
one hidden state output for each input
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

# define model
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1, return_sequences=True)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
print("data:", data, "\n", "data.shape:", data.shape)

# make and show prediction
output = model.predict(data)
print("output", output, "\n", "output.shape", output.shape)
"""
[[[-0.00189824]
  [-0.00565225]
  [-0.01119041]]]
"""