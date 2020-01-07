import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

def my_show(img,size=(3,3)):
	plt.figure(figsize = size)
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.show()

def image_crop(img):  # you code here
	heigh = int(input("你想要截取的图片高度："))
	width = int(input("你想要截取的图片宽度："))
	fx, fy = map(int,input("你想从哪里开始截取 fx fy：").split())
	row, col, channel = img.shape
	print(img.shape,heigh,width,fx,fy)
	if fx >= row:
		fx = input("你的截取高度大于图像尺寸")
	if fy > col:
		fy = input("你的截取宽度大于图像尺寸")
	imagec = img[fx:(heigh+fx),fy:(width+fy)]
	my_show(imagec)




def color_shift(img):  # you code here
	img_mask=[]
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low_red =np.array([0,43,46])
	up_red = np.array([10,255,255])

	low_blue = np.array([100,43,46])
	up_blue = np.array([124,255,255])

	plt.subplot(121)
	mask_red = cv2.inRange(hsv, low_red, up_red)
	erode_red = cv2.erode(mask_red, None, iterations = 1)
	dilate_red = cv2.dilate(erode_red, None, iterations = 1)
	plt.imshow(dilate_red)
	plt.subplot(122)
	mask_blue = cv2.inRange(hsv, low_blue, up_blue)
	erode_blue = cv2.erode(mask_blue, None, iterations = 1)
	dilate_blue = cv2.dilate(erode_blue, None, iterations = 1)
	plt.imshow(dilate_blue)
	plt.show()


	choice = input("你想更换图片中的哪种背景？，1：红色，2：蓝色:")
	B,G,R = map(int, input("你想更换成哪种颜色？输入颜色B G R值：").split())
	color=[B,G,R]
	if int(choice) == 1:
		mask_inv = cv2.bitwise_not(dilate_red)
		img1 = cv2.bitwise_and(img, img, mask = mask_inv)
		bg = img.copy()
		rows, cols, channels = img.shape
		bg[:rows, :cols, :] =color
		img2 = cv2.bitwise_and(bg, bg, mask = dilate_red)
		img_color = cv2.add(img1, img2)
		my_show(img_color)
	elif int(choice) == 2:
		mask_inv = cv2.bitwise_not(dilate_blue)
		img1 = cv2.bitwise_and(img, img, mask = mask_inv)
		bg = img.copy()
		rows, cols, channels = img.shape
		bg[:rows, :cols, :] = color
		img2 = cv2.bitwise_and(bg, bg, mask = dilate_blue)
		img_color = cv2.add(img1, img2)
		my_show(img_color)

	key = cv2.waitKey(0)
	if key == 27:
		cv2.destroyAllWindows()
	return


def rotation(img):  # you code here
	row, col, channel = img.shape
	rot=int(input("你想要旋转多少度？"))
	M = cv2.getRotationMatrix2D((row / 2, col / 2), rot, 1)
	Rotation_img = cv2.warpAffine(img, M, (row, col))
	my_show(Rotation_img)

def perspective_transform(img):  # you code here
	height, width, channel = img.shape
	random_margin = 60
	x1 = random.randint(-random_margin, random_margin)
	y1 = random.randint(-random_margin, random_margin)
	x2 = random.randint(width - random_margin - 1, width - 1)
	y2 = random.randint(-random_margin, random_margin)
	x3 = random.randint(width - random_margin - 1, width - 1)
	y3 = random.randint(height - random_margin - 1, height - 1)
	x4 = random.randint(-random_margin, random_margin)
	y4 = random.randint(height - random_margin - 1, height - 1)

	dx1 = random.randint(-random_margin, random_margin)
	dy1 = random.randint(-random_margin, random_margin)
	dx2 = random.randint(width - random_margin - 1, width - 1)
	dy2 = random.randint(-random_margin, random_margin)
	dx3 = random.randint(width - random_margin - 1, width - 1)
	dy3 = random.randint(height - random_margin - 1, height - 1)
	dx4 = random.randint(-random_margin, random_margin)
	dy4 = random.randint(height - random_margin - 1, height - 1)
	pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	img_pt = cv2.warpPerspective(img, M, (height, width))
	my_show(img_pt)
	return

def main():
	picture=input("变换图片名称：")
	print(picture)
	img_orig=cv2.imread(str(picture),1)
	choic=input("你想做哪种图片变换，1：截图，2：变色，3：旋转，4：透视,5:退出:")
	if int(choic)==1:
		image_crop(img_orig)
	elif int(choic)==2:
		color_shift(img_orig)
	elif int(choic)==3:
		rotation(img_orig)
	elif int(choic)==4:
		perspective_transform(img_orig)
	else:
		exit()
	return



if __name__=="__main__":
	main()



