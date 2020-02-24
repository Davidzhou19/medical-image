import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import random

def get_data():
    sampleNo=200
    mu=np.array([[1,5]])
    sigma=np.array([[2,0],[0,3]])
    R=cholesky(sigma)
    s1=np.dot(np.random.randn(sampleNo,2),R)+mu
    # plt.scatter(s1[:,0],s1[:,1])

    mu = np.array([[6, 0]])
    sigma = np.array([[2, 1], [1, 2]])
    R = cholesky(sigma)
    s2 = np.dot(np.random.randn(sampleNo, 2), R) + mu
    # plt.scatter(s2[:, 0], s2[:, 1])

    mu = np.array([[8, 10]])
    sigma = np.array([[2, 0], [0, 2]])
    R = cholesky(sigma)
    s3 = np.dot(np.random.randn(sampleNo, 2), R) + mu
    # plt.scatter(s3[:, 0], s3[:, 1])
    # plt.show()

    X=np.vstack((s1,s2,s3))
    return X

def distance(pointA,pointB):
    dist=(pointA-pointB)*(pointA-pointB).T
    return dist[0,0]

# def randCent(data,K):
# # 	n=np.shape(data)[1]
# # 	centroids=np.mat(np.zeros((k,n)))
# # 	for j in range(n):
# # 		minJ=np.min(data[:,j])
# # 		rangeJ=np.max(data[:,j])-minJ
# # 		centroids[:,j]=minJ*np.mat(np.ones((k,1)))+np.random.rand(k,1)*rangeJ
# # 	return centroids

def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            sum_all += d[j]
        sum_all *= random.random()
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers

def nearest(point, cluster_centers):
    min_dist =np.inf
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        d = distance(point, cluster_centers[i, ])
        if min_dist > d:
            min_dist = d
            # print(min_dist)
    return min_dist

def kmeans(data,k,centroids):
    m,n=np.shape(data)
    print("m,n:",m,n)
    subcenter=np.mat(np.zeros((m,2)))
    change=True
    while change==True:
        change=False
        for i in range(m):
            minDist=np.inf
            minIndex=0
            for j in range(k):
                dist=distance(data[i,:],centroids[j,:])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if subcenter[i,0]!=minIndex:
                change=True
                subcenter[i,]=np.mat([minIndex,minDist])
        for j in range(k):
            sum_all=np.mat(np.zeros((1,n)))
            r=0
            for i in range(m):
                if subcenter[i,0]==j:
                    sum_all+=data[i,]
                    r+=1
        return subcenter


if __name__=="__main__":
    k=3
    data=get_data()
    centeroids=get_centroids(data,k) #随机中心
    subCenter=kmeans(data,k,centeroids)

    for i in range(len(data)):
        if subCenter[:,0][i]==2:
            plt.scatter(data[:,0][i],data[:,1][i],c="red")
        elif subCenter[:,0][i]==1:
            plt.scatter(data[:, 0][i], data[:, 1][i], c = "green")
        elif subCenter[:,0][i]==0:
            plt.scatter(data[:, 0][i], data[:, 1][i], c = "black")
    plt.show()



