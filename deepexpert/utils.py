import numpy as np

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
	
