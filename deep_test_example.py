from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt
from numpy.random import randn
from utils import Indicator, DeepExpertLayer

# parameters
n_breaks = 10
single_output = True
n_epochs = 40

# sample data
x = randn(1000, 50)
y = randn(1000, 1)

# architecture
input_layer = Input((x.shape[1],))
l1 = Dense(64, activation = 'relu')(input_layer)

# local expert layers
output = DeepExpertLayer(l1, n_breaks, depth = 3, widths = [64, 64])

# fit model
mod = Model(input_layer, output)
print(mod.summary())
mod.compile(optimizer = 'adam', loss = 'binary_crossentropy')
mod.fit(x, Indicator(y, n_breaks, stack = True), epochs = n_epochs)

# demonstrate test CDF plot
test_pred = mod.predict(randn(1, 50)).ravel()
plt.plot(test_pred)
plt.show()