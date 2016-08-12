import urllib
import gzip
import struct
from array import array
import numpy as np
import os


"""
	Downloads MNIST data set and saves the data set as pickled numpy arrays
	
	Script adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
	and python documentation
"""


def gzip_extract(filenameIn,filenameOut):
	f_in=gzip.open(filenameIn,'rb')
	f_out=open(filenameOut,'wb')
	f_out.write(f_in.read())
	f_in.close()
	f_out.close()

def unpack_mnist(filenameIm,filenameLabels):
	flbl = open(filenameLabels, 'rb')
	magic_nr, size = struct.unpack(">II", flbl.read(8))
	lbl = array("b", flbl.read())
	flbl.close()

	fimg = open(filenameIm, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img=array("B",fimg.read())
	fimg.close()

	ind = [ k for k in xrange(size)]
	N=len(ind)
	images =  np.zeros((rows,cols,N),dtype=np.uint8)
	labels = np.zeros((N, 1),dtype=np.uint8)



	for i in xrange(N):
		images[:, :,i] = np.array(img[ i*rows*cols : (i+1)*rows*cols]).reshape(rows,cols)
		labels[i] = lbl[i]

	return images,labels


train_img_url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label_url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

test_img_url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label_url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


train_img_gz="train-images-idx3-ubyte.gz"
train_label_gz="train-labels-idx1-ubyte.gz"
test_img_gz="t10k-images-idx3-ubyte.gz"
test_label_gz="t10k-labels-idx1-ubyte.gz"

trainIm_file="train-images-idx3-ubyte"
trainLabel_file="train-labels-idx1-ubyte"
testIm_file="t10k-images-idx3-ubyte"
testLabel_file="t10k-labels-idx1-ubyte"

print "Downloading MNIST from http://yann.lecun.com/exdb/mnist/index.html ..."

print "Downloading training set...",
urllib.urlretrieve(train_img_url,train_img_gz)
urllib.urlcleanup()
urllib.urlretrieve(train_label_url,train_label_gz)
urllib.urlcleanup()
print "Done!"

print "Extracting ubyte files from gz files..."
print "-"+train_img_gz
print "-"+train_label_gz
gzip_extract(train_img_gz,trainIm_file)
gzip_extract(train_label_gz,trainLabel_file)
print "...Done!"

print "Extracting training data into numpy arrays...",
trainIm,trainLabel=unpack_mnist(trainIm_file,trainLabel_file)
print "Done!"

print "Pickling numpy arrays...",
trainIm.dump("trainIm.pkl");
trainLabel.dump("trainLabel.pkl")
print "Done!"

print " "

print "Downloading test set...",
urllib.urlretrieve(test_img_url,test_img_gz)
urllib.urlcleanup()
urllib.urlretrieve(test_label_url,test_label_gz)
urllib.urlcleanup()
print "Done!"

print "Extracting ubyte files from gz files..."
print "-"+test_img_gz
print "-"+test_label_gz
gzip_extract(test_img_gz,testIm_file)
gzip_extract(test_label_gz,testLabel_file)
print "...Done!"

print "Extracting test data into numpy arrays...",
testIm,testLabel=unpack_mnist(testIm_file,testLabel_file)
print "Done!"

print "Pickling numpy arrays...",
testIm.dump("testIm.pkl");
testLabel.dump("testLabel.pkl")
print "Done!"

print "Cleaning up...",
os.remove(train_img_gz)
os.remove(train_label_gz)
os.remove(test_img_gz)
os.remove(test_label_gz)

os.remove(trainIm_file)
os.remove(trainLabel_file)
os.remove(testIm_file)
os.remove(testLabel_file)
print "Done!"
