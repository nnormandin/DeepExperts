from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from utils import Indicator, DeepExpertLayer

# parameters
n_breaks = 10
single_output = True
n_epochs = 50
opt = 'adam'
#opt = Adam(lr = 0.001)

# import boston housing data
(x, y), (x_test, y_test) = boston_housing.load_data()
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_vecs, y_testvecs, breaks = Indicator(y, n_breaks, stack = True, y_test = y_test)

# architecture
input_layer = Input((x.shape[1],), name = 'input')
l1 = Dense(64, activation = 'relu')(input_layer)
l1 = Dropout(.5)(l1)
l1 = Dense(64, activation = 'relu')(l1)
l1 = Dropout(.75)(l1)
l1 = Dense(32, activation = 'relu')(l1)

# local expert layers
output = DeepExpertLayer(l1, n_breaks, depth = 3, widths = [32, 16, 16])

# fit model
mod = Model(input_layer, output)
print(mod.summary())
mod.compile(optimizer = opt, loss = 'binary_crossentropy')
mod.fit(x, y_vecs,
		validation_data = (x_test, y_testvecs),
		epochs = n_epochs, batch_size = 16)

# demonstrate test CDF plot
test_preds = mod.predict(x_test)
plt.plot(test_preds[0])
plt.show()


SVG(model_to_dot(mod).create(prog='dot', format='svg'))