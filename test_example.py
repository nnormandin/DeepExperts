from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt
from numpy.random import randn
from utils import Indicator, ExpertLayer

# parameters
n_breaks = 20
single_output = True
n_epochs = 500

# sample data
x = randn(1000, 50)
y = randn(1000, 1)

# architecture
input_layer = Input((x.shape[1],))
l1 = Dense(64, activation = 'relu')(input_layer)

# local expert layer
output = ExpertLayer(l1, n_breaks)

# fit model
mod = Model(input_layer, output)
mod.compile(optimizer = 'adam', loss = 'binary_crossentropy')
mod.fit(x, Indicator(y, n_breaks, stack = True), epochs = n_epochs)

# demonstrate test CDF plot
test_pred = mod.predict(randn(1, 50)).ravel()
plt.plot(test_pred)
plt.show()