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
x_test = randn(1000, 50)
y_test = randn(1000, 1)

# generate binary y vectors
y_vecs, y_test, breaks = Indicator(y, n_breaks, stack = True, y_test = y_test)

# architecture
input_layer = Input((x.shape[1],), name = 'input')
l1 = Dense(64, activation = 'relu')(input_layer)

# local expert layers
output = DeepExpertLayer(l1, n_breaks, depth = 1, widths = [64]*5)

# fit model
mod = Model(input_layer, output)
print(mod.summary())
mod.compile(optimizer = 'adam', loss = 'binary_crossentropy')
mod.fit(x, y_vecs, epochs = n_epochs, validation_data = (x_test, y_test),
		shuffle = True)

# demonstrate test CDF plot
test_pred = mod.predict(randn(1, 50)).ravel()
plt.plot(test_pred)
plt.show()

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(mod).create(prog='dot', format='svg'))