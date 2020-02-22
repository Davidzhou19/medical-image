import cv2
import numpy as np

def sift_kp(image):
	gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift=cv2.xfeatures2d_SIFT.create()
	kps,features=sift.detectAndCompute(image,None)
	kp_image=cv2.drawKeypoints(gray_image,kps,None)
	return kp_image,kps,features

def get_match(des1,des2):
	bf=cv2.BFMatcher()
	matches=bf.knnMatch(des1,des2,k=2)
	match=[]
	for m,n in matches:
		if m.distance<0.75*n.distance:
			match.append(m)
	return match

def siftImage(img1,img2):
	km1,kp1,des1=sift_kp(img1)
	km2,kp2,des2=sift_kp(img2)
	KM=np.concatenate((km1,km2),axis=1)
	# cv2.imshow("KM", KM)
	siftmatch=get_match(des1,des2)
	# print("siftmatch",siftmatch)
	if len(siftmatch)>4:
		ptsA=np.float32([kp1[m.queryIdx].pt for m in siftmatch]) #pt是坐标点tuple（x,y)
		ptsB=np.float32([kp2[m.trainIdx].pt for m in siftmatch])
		ransacReprojThreshold=5
		H,status=cv2.findHomography(ptsB,ptsA,cv2.RANSAC,ransacReprojThreshold);
		imgOut=cv2.warpPerspective(img2,H,(img2.shape[1]*2,img2.shape[0]))

	# ptsAA=([kp1[m.queryIdx].pt for m in siftmatch])
	# ptsBB=([kp2[m.trainIdx].pt for m in siftmatch])
	# print(ptsAA[0][0],ptsBB[0][0])
	# h1,w1,_=km1.shape
	# for j in range(len(ptsAA)):
	# 	KMM=cv2.line(KM,(int(ptsAA[j][0]),int(ptsAA[j][1])),(w1+int(ptsBB[j][0]),int(ptsBB[j][1])),(0,0,255),1)
	# cv2.imshow("KM", KMM)
		draw_params = dict(matchColor = (0, 255, 0),
						   singlePointColor = None,
						   matchesMask=status.ravel().tolist(),
						   flags = 2)
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,siftmatch,None,**draw_params)
		cv2.imshow("img3",img3)

	return imgOut,H,status

def mix(img_R,img_O):#将两张图片混合
	h1,w1=img_O.shape[:2]
	h2,w2=img_R.shape[:2]
	padding=w2-w1
	print("h1:",h1,"w1:",w1)
	print("h2:", h2, "w2:", w2)
	print("padding",padding)
	img_padded=np.zeros(shape=(h2,w2,3),dtype=np.uint8)
	print(img_padded.shape)
	img_padded[:h1,:w1,:]=img_O[:,:,:]
	img_padded[:, w1:w2, :] = img_R[:, w1:w2, :]
	return img_padded

def main():
	img1 = cv2.imread("t3.jpg")
	img2 = cv2.imread("t4.jpg")
	print(img1.shape, img2.shape)

	result, H_max, machesMask = siftImage(img1, img2)
	res=mix(result,img1)

	cv2.imshow("allImg", res)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=="__main__":
	main()
