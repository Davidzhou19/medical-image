import cv2
import matplotlib.pyplot as plt
import numpy as np

def mediumBlur(img,kernel):
	h, w, c = img.shape
	#print(img.shape,kernel)
	img1=np.zeros((h,w,c),np.uint8)
	for chann in range(c):
		for kh in range(1,h):
			for kw in range(1,w):
				temp=np.zeros(kernel*kernel,np.uint8)
				s=0
				for k in range(-1,kernel-1):
					for i in range (-1,kernel-1):
						kkh=kh+i
						kkw=kw+k
						if kkh>=h:
							kkh=h-1
						if  kkw>=w:
							kkw=w-1
						temp[s]=img[kkh,kkw,chann]
						s+=1
				temp.sort()
				medi=temp[(int((kernel*kernel)/2))]
				img1[kh,kw,chann]=medi
		# print(img1)
		# plt.imshow(img1)
		# plt.show()
	return img1
def my_show(img):
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def main():
	name=input("please input which picture:")
	ker=int(input("kernel:"))
	img=cv2.imread(name,1)
	plt.figure(figsize = (10,10),dpi = 100)
	plt.subplot(121)
	my_show(img)
	plt.subplot(122)
	md_img=mediumBlur(img,ker)
	my_show(md_img)
	plt.show()

if __name__=="__main__":
	main()
