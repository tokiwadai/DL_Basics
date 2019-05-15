"""
https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/

Running the example returns 3 arrays:
The LSTM hidden state output for the last time step.
The LSTM hidden state output for the last time step (again).
The LSTM cell state for the last time step.
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))

# make and show prediction
print(model.predict(data))