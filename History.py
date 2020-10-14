import numpy as np

class History:
	def __init__(self, shape):
		# Takes in number of buffer frames, size in rows, and size in columns
		# In this case it's 84 x 84 x 4 (buffer frames)
		self.buffer = np.zeros(shape, dtype=np.float32)

	def value(self):
		return self.buffer

	def append(self, state):
		self.buffer[:-1] = self.buffer[1:]
		self.buffer[-1] = state

	def reset(self):
		self.buffer.fill(0)