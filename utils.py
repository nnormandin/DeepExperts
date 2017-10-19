import numpy as np
from keras.layers import Dense, concatenate


def Indicator(y, n = 10, stack = False):
	breaks = np.linspace(y.min(), y.max(), num = n+2)[1:-1]
	out = [np.where(y > i, 0, 1) for i in breaks]
	if stack:
		return np.hstack(out)
	else:
		return out

def ExpertLayer(input_layer, n = 10):
	output_layers = [Dense(1, activation = 'sigmoid',
						name = 'le_{}'.format(i))(input_layer) for i in range(n)]
	out_concat = concatenate(output_layers)
	return out_concat

