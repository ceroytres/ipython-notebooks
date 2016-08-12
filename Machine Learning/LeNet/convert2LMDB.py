# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 23:03:45 2016

@author: dmare
"""

import lmdb
import caffe
import numpy as np


train_X=np.load('trainIm.pkl')
train_y=np.load('trainLabel.pkl')

map_size = 10*train_X.nbytes

env = lmdb.open('mnist_train_lmdb',map_size=map_size)
N=train_X.shape[2]

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = train_X.shape[0]
        datum.width = train_X.shape[1]
        datum.data = train_X[:,:,i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(train_y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
#env.close()        
        
test_X=np.load('testIm.pkl')
test_y=np.load('testLabel.pkl')

map_size = 10*test_X.nbytes

env = lmdb.open('mnist_test_lmdb',map_size=map_size)
N=test_X.shape[2]

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = test_X.shape[0]
        datum.width = test_X.shape[1]
        datum.data = test_X[:,:,i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(test_y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
#env.close()        


#lmdb_env = lmdb.open('mnist_test_lmdb')
#lmdb_txn = lmdb_env.begin()
#lmdb_cursor = lmdb_txn.cursor()
#datum = caffe.proto.caffe_pb2.Datum()
#
#for key, value in lmdb_cursor:
#    datum.ParseFromString(value)
#    label = datum.label
#    data = caffe.io.datum_to_array(datum)
#    break
#lmdb_env.close()