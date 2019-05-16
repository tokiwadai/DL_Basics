"""
https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/

Running the example, we can see now why the LSTM output tensor
    and hidden state output tensor are declared separably.
The layer returns the hidden state for each input time step, then separately,
    the hidden state output for the last time step and the cell state for the last input time step.
This can be confirmed by seeing that the last value in the returned sequences (first array)
    matches the value in the hidden state (second array).
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))

# make and show prediction
print(model.predict(data))