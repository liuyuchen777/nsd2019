import numpy as np

class Maxpool2d(object):
	def __init__(self, kernel_size):
		self.kernel_size = kernel_size
	def forward(self,x):
		height = x.shape[2]
		width  = x.shape[3]
		batch_size = x.shape[0]
		channels = x.shape[1]
		out_x = np.zeros((batch_size, channels, np.ceil((height/self.kernel_size[0])).astype(np.int), np.ceil((width/self.kernel_size[1])).astype(np.int)))
		self.reg = np.zeros((batch_size, channels, np.ceil((height/self.kernel_size[0])).astype(np.int), np.ceil((width/self.kernel_size[1])).astype(np.int)),dtype = np.int) #to rember which pixel we choose
		self.origin_size = x.shape
		idx = 0
		jdx = 0
		for i in range(0,height,self.kernel_size[0]):
			jdx = 0
			for j in range(0,width,self.kernel_size[1]):
				sliding_window = x[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]]
				out_x[:,:,idx,jdx] = sliding_window.max(axis = (2,3))
				self.reg[:,:,idx,jdx] = sliding_window.reshape(batch_size,channels,-1).argmax(axis = 2 ).astype(np.int)
				jdx += 1
			idx += 1
		return out_x

	def backprop(self, delta_loss):
		delta_w = np.zeros(self.origin_size)
		height = self.origin_size[2]
		width = self.origin_size[3]
		idx = 0
		for i in range(0,height,self.kernel_size[0]):
			jdx = 0
			for j in range(0,width,self.kernel_size[1]):
				sliding_window = np.zeros(delta_w[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]].reshape(-1).shape)
				sliding_window[self.reg[:,:,idx,jdx]] = delta_loss[:,:,idx,jdx]
				sliding_window = sliding_window.reshape(delta_w[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]].shape)
				delta_w[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]] = sliding_window
				jdx += 1
			idx += 1
		return delta_w

