from Layers.Convolution2d import Conv2d
from Layers.Maxpool2d import Maxpool2d
from Layers.Linear import Linear
from Method.Data import Load_file
import numpy as np

class CNN(object):
	def __init__(self):
		self.layer1 = Conv2d(1,16,kernel_size = (5,5), stride = (1,1), padding = (2,2))
		self.layer2 = Maxpool2d(kernel_size = (2,2))
		self.layer3 = Conv2d(16,36,kernel_size = (5,5), stride = (1,1), padding = (2,2))
		self.layer4 = Maxpool2d(kernel_size = (2,2))
		self.layer5 = Linear(7*7*36,128)
		self.layer6 = Linear(128,10)
	def forward(self,x):
		x = self.layer1.forward(x)
		x = self.layer2.forward(x)
		x = self.layer3.forward(x)
		x = self.layer4.forward(x).reshape(-1,7*7*36)
		x = self.layer5.forward(x)
		x = self.layer6.forward(x)
		return x
	def backprop(self,delta_loss, lr = 0.1):
		delta_loss = self.layer6.backprop(delta_loss,lr = lr)
		delta_loss = np.asarray(self.layer5.backprop(delta_loss,lr = lr)).reshape(-1,36,7,7)
		delta_loss = self.layer4.backprop(delta_loss)
		delta_loss = self.layer3.backprop(delta_loss,lr = lr)
		delta_loss = self.layer2.backprop(delta_loss)
		delta_loss = self.layer1.backprop(delta_loss,lr = lr)

def softmax(X):
    exps = np.exp(X - np.max(X,axis = 1).reshape(-1,1))
    return exps / np.sum(exps,axis = 1).reshape(-1,1)

train_x,train_y = Load_file(dataset = 'MNIST', mode = 'train')
test_x,test_y = Load_file(dataset = 'MNIST', mode = 'test')
train_y_onehot = []
for i in range(len(train_y)):
	tmp = np.zeros(10)
	tmp[train_y[i]] = 1
	train_y_onehot.append(tmp)
train_y_onehot = np.asarray(train_y_onehot)
model = CNN()
test_x = test_x[:100]
test_y = test_y[:100]

for i in range(1000):
	batch_size = 32
	lr = 1e-8
	x = train_x[i*batch_size:(i+1)*batch_size]
	y = train_y_onehot[i*batch_size:(i+1)*batch_size]

	out = model.forward(x)
	grad = softmax(out)
	#print(grad)
	#print(y)
	grad -= y
	#print(grad)
	grad = grad/10
	model.backprop(grad,lr = lr)

	out = model.forward(test_x).argmax(axis = 1).reshape(-1,1)
	print('Iteraion ',i,' : test accuracy : ',(test_y == out).astype(np.int).sum() ,'%')

