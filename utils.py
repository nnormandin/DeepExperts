import numpy as np
from keras.layers import Dense, concatenate

def Indicator(y, n = 10, breaks = None, y_test = None, stack = False):
	if not breaks:
		breaks = np.linspace(y.min(), y.max(), num = n+2)[1:-1]
	out = [np.where(y > i, 0, 1) for i in breaks]
	if y_test is not None:
		out2 = [np.where(y_test > i, 0, 1) for i in breaks]
		
	if stack and (y_test is not None):
		return np.hstack(out), np.hstack(out2), breaks
	elif y_test:
		return out, out2, breaks
	elif stack:
		return np.hstack(out), breaks
	else:
		return out, breaks
	
def ExpertLayer(input_layer, n = 10):
	output_layers = [Dense(1, activation = 'sigmoid',
						name = 'le_{}'.format(i))(input_layer) for i in range(n)]
	out_concat = concatenate(output_layers)
	return out_concat

def BuildLayer(input_list, width = 64, activ = 'relu', layer_num = 0):
	return [Dense(width, activation = activ)(i) for i in input_list]

def DeepExpertLayer(input_layer, n = 10, depth = 3, activ = 'relu',
					widths = [64, 64]):

	# if depth is 1, just return ExpertLayer
	if depth == 1:
		return ExpertLayer(input_layer, n)
	else:
		for d in range(depth):
			# first layer
			if d == 0:
				layer = [Dense(widths[d], activation = activ)(input_layer)
				for i in range(n)]
			# last layer
			elif d == depth-1:
				layer = BuildLayer(layer, width = 1, activ = 'sigmoid')
				return concatenate(layer, name = 'combined_output')
			# middle layers
			else:
				layer = BuildLayer(layer, widths[d], activ = activ)