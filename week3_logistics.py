import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate(sample_size, num_classes, diff):
	mean = np.random.randn(2)
	cov = np.eye(2)
	samples_per_class = int(sample_size / num_classes)
	X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
	Y0 = np.zeros(samples_per_class)
	for ci, d in enumerate(diff):
		X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
		Y1 = (ci + 1) * np.ones(samples_per_class)
		X0 = np.concatenate((X0, X1))
		Y0 = np.concatenate((Y0, Y1))
	# print(X0, Y0)
	return X0, Y0

def sigmoid(z):
	s=1/(1+np.exp(-z))
	return s

def initialize_with_zero(dim):
	w=np.ones((1,dim))
	b=0
	return w,b

def logistics_regression(X,Y,lr,max_iter):
	global buffer_w,buffer_b,buffer_batch
	w,b=initialize_with_zero(2)
	m = X.shape[0]
	pick=1
	for i in range(max_iter):
		Z = np.dot(X,w.T) + b
		Y_hat = sigmoid(Z)

		error=Y_hat-Y
		# print(error.shape)
		dw=np.dot(error.T,X)/m
		db=np.sum(np.dot(error.T,np.ones(shape=(X.shape[0],1))))
		loss=-np.sum(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))/m

		w=w-lr*dw
		b=b-lr*db

		if pick%100==0:
			print("iter:{0},\nw:{1},\nb:{2},\nloss:{3}".format(pick,dw,db,loss))
			buffer_w.append(w)
			buffer_b.append(b)
			buffer_batch.append(pick)
		pick+=1
	# print(w.shape,b.shape)
	return w,b

def update(i):
	global buffer_w,buffer_b,buffer_batch,line,x,ax
	buffer=buffer_batch[i]
	label = 'buffer_batch {0}'.format(buffer)
	print(label)
	w=buffer_w[i]
	b=buffer_b[i]
	print(w,b)
	line.set_ydata((-b-w[0][0]*x)/w[0][1])
	ax.set_xlabel(label)
	return line, ax

def main():
	global buffer_w,buffer_b,buffer_batch,line,x,ax
	buffer_w=[]
	buffer_b=[]
	buffer_batch=[]

	num_classes = 2
	X, Y = generate(500, num_classes, [2.0])
	Y=np.expand_dims(Y,axis = 1)

	w, b = logistics_regression(X, Y, 0.001, 1500)

	fig, ax = plt.subplots()
	fig.set_tight_layout(True)
	x = np.arange(-2, 6, 0.1)
	colors = ['r' if i == 0 else 'b' for i in Y]
	ax.scatter(X[:, 0], X[:, 1], c = colors)
	line, = ax.plot(x,x, 'y', linewidth = 2)

	anim = FuncAnimation(fig, update, frames =len(buffer_batch), interval = 200)

	if len(sys.argv) > 1 and sys.argv[1] == 'save':
		anim.save('logistics.gif', dpi = 80, writer = 'logistics')
	else:
		# Plt.show()会一直循环动画
		plt.show()

if __name__=="__main__":
	main()
