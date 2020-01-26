import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def general_data():
	X=(np.random.randint(0,100,size=100))*random.random()
	w=random.randint(0,10)+random.random()
	b=random.randint(0,5)+random.random()
	noise=np.random.normal(0,7,[X.shape[0]])
	Y=w*X+b+noise
	return  X,Y,w,b

def liner_regression(X,Y,lr,max_iter):
	global buffer_w,buffer_b,buffer_batch
	batch=0
	w,b=0,0

	for i in range(max_iter):
		Y_hat= w * X + b
		dw=np.dot((Y_hat-Y).T,X)/X.shape[0]
		db=np.sum(Y_hat-Y)/X.shape[0]
		loss = 0.5 * np.dot((Y_hat-Y).T,(Y_hat-Y))/X.shape[0]

		w=w-lr*dw
		b=b-lr*db

		if batch%100==0:
			print("batch:{0},\nw:{1},b:{2},\nloss:{3}".format(batch,w,b,loss))
			buffer_w.append(w)
			buffer_b.append(b)
			buffer_batch.append(batch)

		batch+=1
		plt.show()
	return w,b


def update(i):
	global buffer_w, buffer_b, buffer_batch,line,ax,x
	buffer=buffer_batch[i]
	label="batch{0}".format(buffer)
	print(label)
	w=buffer_w[i]
	b=buffer_b[i]
	print(w,b)
	line.set_ydata(w*x+b)
	ax.set_xlabel(label)
	return line, ax

def show_scatter(X,Y,w,b):
	plt.figure()
	plt.scatter(X,Y,c="b")
	x=np.linspace(-1,np.argmax(X),100)
	y=w*x+b
	plt.plot(x,y,lw=3,c="r")
	plt.show()

def main():
	global line, ax, buffer_batch, buffer_b, buffer_w, x
	buffer_w = []
	buffer_b = []
	buffer_batch = []

	X,Y,w,b=general_data()
	pred_w,pred_b=liner_regression(X,Y,0.001,1000)
	print("T_w:{0},T_b:{1},\nPred_w:{2},Pred_b:{3}".format(w, b, pred_w, pred_b))

	fig, ax = plt.subplots()
	fig.set_tight_layout(True)
	x = np.arange(-2, 10, 0.1)
	ax.scatter(X, Y, c = "b")
	line, = ax.plot(x, x, 'y', linewidth = 2)

	anim = FuncAnimation(fig, update, frames = len(buffer_batch), interval = 200)
	plt.show()

if __name__=="__main__":
	main()