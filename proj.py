import pandas as pd
import numpy as np
import math
import numpy.matlib as M
import cvxopt
from cvxopt import solvers
from cvxopt import matrix
import scipy.io as sio
import scipy.sparse
from sklearn import preprocessing
import os 
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.misc as sim
from skimage import io

def computeCentroids(X,idx,k):
	m=X.shape[0]
	n=X.shape[1]
	centroids=np.zeros((k,n))
	itr=np.zeros((k,1))

	for i in range(0,len(idx)):
		l=int(idx[i,0])
		centroids[l,:]=centroids[l,:]+X[i,:];
		itr[l]=itr[l]+1;
	#print(itr)
	#print(centroids)
	centroids=np.divide(centroids,itr)
	return centroids;	


def findClosestCentroids(X,centroids):
	K=centroids.shape[0]
	idex=np.zeros((X.shape[0],1))
	for i in range(0,X.shape[0]):
		min=1000000000
		for j in range(0,K):
			if(pow(LA.norm(X[i,:]-centroids[j,:]),2)<min):
				min=pow(LA.norm(X[i,:]-centroids[j,:]),2);
				idex[i,0]=j;

	return idex;    	

def kMeansInitCentroids(X,k):
	c=np.zeros((k,X.shape[1]));
	X1=np.unique(X,axis=0);
	r=np.random.permutation(X1.shape[0]);
	for i in range (0,k):
		c[i]=X1[i];
	return c;

def runkMeans(X,initial_centroids,max_iters):
    m=X.shape[0]
    n=X.shape[1]
    K=initial_centroids.shape[0]
    centroids=initial_centroids
    previous_centroids=centroids
    idx=np.zeros((m,1))

    for i in range(0,max_iters):
        idx=findClosestCentroids(X,centroids);
        centroids=computeCentroids(X,idx,K)
        print("Iteration number:",i+1);
	#uncomment to see how the centroids reach stable values from initial state        
	#a=[]
        #b=[]
        #c=[]
        #d=[]
        #for i in range(0,centroids.shape[0]):
        #	a.append(centroids[i,0])
        #	b.append(centroids[i,1])

        #for i in range(0,X.shape[0]):
        #	c.append(X[i,0])
        #	d.append(X[i,1])

        #plt.plot(c,d,'bs',a,b,'ro')
        #plt.axis([0,np.amax(X),0,np.amax(X)])
        #plt.show()


    return centroids,idx;

print('Testing kNN on test data')

mat=sio.loadmat('ex7data2.mat');
d1=mat['X'];
#print(d1)
#d1=d1.reshape((,2));
k=3;
#initial centroids can be made random also 
initial_centroids=np.array([[3,3],[6,2],[8,5]]);

idx=findClosestCentroids(d1,initial_centroids);
print('Closest centroids for the first 3 examples:',idx[2,0]);
input("Press enter to continue");
#os.system("pause");

centroids=computeCentroids(d1,idx,k);
print('Centroids computed after initial finding of closest centroids:')
print(centroids);

a=[]
b=[]
for i in range(0,k):
    a.append(centroids[i,0])
    b.append(centroids[i,1])

c=[]
d=[]
for i in range(0,d1.shape[0]):
	c.append(d1[i,0])
	d.append(d1[i,1])

plt.plot(c,d,'bs',a,b,'ro') 
plt.axis([0,10,0,10])
plt.show()

centroids,idx=runkMeans(d1,initial_centroids,10);
a=[]
b=[]
for i in range(0,k):
    a.append(centroids[i,0])
    b.append(centroids[i,1])

c=[]
d=[]
for i in range(0,d1.shape[0]):
	c.append(d1[i,0])
	d.append(d1[i,1])

plt.plot(c,d,'bs',a,b,'ro') 
plt.axis([0,10,0,10])
plt.show()


input("Press enter to continue");
#os.system("pause");



pic=scipy.misc.imread('bird_small.png')

image1=io.imread('bird_small.png')
io.imshow(image1)
io.show()

pic=pic/255
pic_size=pic.shape
pic=np.reshape(pic,(pic_size[0]*pic_size[1],3))
#print(pic.shape)

KK=16;
max_iter=50;

init_centroid=kMeansInitCentroids(pic,KK);

centroids,idx=runkMeans(pic,init_centroid,max_iter);

idx=findClosestCentroids(pic,centroids);

recovered_pic=np.zeros((pic.shape[0],3))


for i in range(0,idx.shape[0]):
	l=int(idx[i,0]);
	recovered_pic[i]=centroids[l]

recovered_pic=np.reshape(recovered_pic,(pic_size[0],pic_size[1],3));

sim.imsave('bircom2.png',recovered_pic);
image2=io.imread('birdcom2.png')
io.imshow(image2)
io.show()
