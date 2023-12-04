#!/usr/bin/env python

import sys
import numpy as np
import os
import scipy.io as scp
from EEGFunctions import LBP_extractor   #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt

from LBPDLmodels import LBPLSTM
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
K.set_image_data_format('channels_last')

#directories with the dataset. You could change it based on the directory of your pc.
working_dir = '/home/dyp/iEEGSeizure2/dataset/'
#working_dir = 'E:/OneDrive - The University of Hong Kong/dataset/Epilepsy EEG Dataset/'

slen = 1 

Patient = 10
# 每个patient的seizure 数量，注意这里由于Sub13有两种seizure，这里先只尝试serzure_type=3
# 由于subj11的seizure7太长，所以舍弃
Patient_Seizure = [2,6,6,4,8,5] 

Acc_sum = np.zeros((int(Patient_Seizure[Patient-10])))

if Patient == 13:
	seizure_type = 3
elif Patient == 10:
	seizure_type = 2
else:
	seizure_type = 1

#frequencies at which the dataset is recorded. 
fs = 500

second = fs
minutes = second*60
#dimension of the LBP = 6 (Section 4.1). T = dimLBP+1
T = 7	#1 + dimension l of binary pattern
totalNumberBP = 2**(T-1)
D = 10000   # dimension of hypervector

# k-fold交叉验证，每次用一个seizure作为测试集，其余为训练集
for k in range(0,Patient_Seizure[Patient-10]):
	
	# read the trainset and corresponding label
	start_point = 1 # 检测是否是第一个训练集数据
	for i in range(0,Patient_Seizure[Patient-10]):
		if i == k: # k作为测试集
			continue
		else:
			if start_point == 1:
				x_train=np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(i+1)+".npy")
				y_train=np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(i+1)+".npy")
				start_point = 0
			else:
				x_train_temp=np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(i+1)+".npy")
				y_train_temp=np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(i+1)+".npy")
				#print(y_test.shape, y_test_temp.shape)
				x_train = np.concatenate((x_train,x_train_temp),axis=0)
				y_train = np.concatenate((y_train,y_train_temp),axis=0)
	nSignals = x_train.shape[1] # number of channels

	# 记录每个label的数量
	num_noSeizure = 0
	num_Seizure = 0

	x_train = torch.from_numpy(x_train*1e5)

	# train the HDC
	print("Training Start")
	for i in range(0,x_train.shape[0]):
	#for i in range(180,185):
		#if num_Seizure == 2000:
		#	break
		if y_train[i] == seizure_type:
			if num_Seizure == 0:
				temp3=LBP_extractor(x_train[i,:,0:250],totalNumberBP,1,T,slen).t() # 
				queeryVectorS1 = temp3 + 0
				num_Seizure = num_Seizure + 1
				temp3=LBP_extractor(x_train[i,:,250:500],totalNumberBP,1,T,slen).t() # 
				queeryVectorS1 = torch.cat((queeryVectorS1, temp3),0)
				num_Seizure = num_Seizure + 1
			else:
				temp3=LBP_extractor(x_train[i,:,0:250],totalNumberBP,1,T,slen).t()  # 
				queeryVectorS1 = torch.cat((queeryVectorS1, temp3),0)
				num_Seizure = num_Seizure + 1
				temp3=LBP_extractor(x_train[i,:,250:500],totalNumberBP,1,T,slen).t()  # 
				queeryVectorS1 = torch.cat((queeryVectorS1, temp3),0)
				num_Seizure = num_Seizure + 1
			
		if i%10000 == 0:
			print(i)



	i = 0
	while i < x_train.shape[0]:
	#while i < 3:
		if y_train[i] == 0:
			if num_noSeizure == 0:
				temp1=LBP_extractor(x_train[i,:,0:250],totalNumberBP,0,T,slen).t() # 
				queeryVectorS0 = temp1 + 0
				num_noSeizure = num_noSeizure + 1
				temp1=LBP_extractor(x_train[i,:,250:500],totalNumberBP,0,T,slen).t() # 
				queeryVectorS0 = torch.cat((queeryVectorS0, temp1),0)
				num_noSeizure = num_noSeizure + 1
				
			else:
				temp1=LBP_extractor(x_train[i,:,0:250],totalNumberBP,0,T,slen).t() # 
				queeryVectorS0 = torch.cat((queeryVectorS0, temp1),0)
				num_noSeizure = num_noSeizure + 1
				temp1=LBP_extractor(x_train[i,:,250:500],totalNumberBP,0,T,slen).t() # 
				queeryVectorS0 = torch.cat((queeryVectorS0, temp1),0)
				num_noSeizure = num_noSeizure + 1

		if i%10000 == 0:
			print(i)

		i = i+1

	print(num_Seizure,num_noSeizure) 

	print(queeryVectorS1.size(), queeryVectorS0.size())

	# train the MLP

	Matrix_train = np.array(torch.cat((queeryVectorS0, queeryVectorS1),0).cpu().numpy())


	#the column 0 contains the labels, the other columns contain the futures
	y = Matrix_train[:,0]
	X = Matrix_train[:,1:totalNumberBP*nSignals+1]


	y_train_2 = np_utils.to_categorical(y)

	X = X.reshape((X.shape[0],1,X.shape[1]))

	X_train, X_validate, Y_train, Y_validate = train_test_split(X, y_train_2, test_size=0.1, random_state=1)

	model = LBPLSTM(nb_classes = 2, Chans = 1, Samples = totalNumberBP*nSignals, dropoutType = 'Dropout')
	#model.summary()
	#model.summary()
	# compile the model and set the optimizers
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
	
	# set a valid path for your system to record model checkpoints
	checkpointer = ModelCheckpoint(filepath='./result/best_model_'+"Patient"+str(Patient)+'_k'+str(k)+'.h5', verbose=1,save_best_only=True)

	fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 100, verbose = 2, validation_data=(X_validate, Y_validate), callbacks=[checkpointer])
	
	
	# test the HDC
	print("Testing Start")
	x_test = []
	y_test = []
	# read the testset and corresponding label

	x_test = np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy") # num*channel*sample
	y_test = np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy")

	nSignals = x_test.shape[1] # number of channels

	num_test = int(x_test.shape[0]*2) # 1s的信号段分为了两个0.5s进行测试


	x_test = torch.from_numpy(x_test*1e5)

	X_test = []

	for i in range(0,num_test):
		if i%2 == 0:
			temp = LBP_extractor(x_test[i//2,:,0:250],totalNumberBP,2,T,slen) # 前0.5second
			X_test_temp = np.array((temp[1:totalNumberBP*nSignals+1].t()).cpu().numpy())
			X_test.append(X_test_temp)
		else:
			temp = LBP_extractor(x_test[i//2,:,250:500],totalNumberBP,2,T,slen) # 后0.5second
			X_test_temp = np.array((temp[1:totalNumberBP*nSignals+1].t()).cpu().numpy())
			X_test.append(X_test_temp)
		if i%10000 == 0:
			print(i)



	X_test = np.array(X_test)
	#print(X_test.shape) # 这时shape已经是(1250, 1, 1216)
	#X_test = X_test.reshape((X_test.shape[0],X_test.shape[2],1))

	# load optimal weights
	model.load_weights('./result/best_model_'+"Patient"+str(Patient)+'_k'+str(k)+'.h5')

	probs       = model.predict(X_test)
	preds       = probs.argmax(axis = -1) 

	# calculate the ACC
	err0 = 0
	err1 = 0
	count = 0
	print(preds.shape)
	for i in range(0,preds.shape[0]):
		if y_test[i//2] == 0:
			if preds[i] != 0:
				err0 = err0 + 1
			count = count + 1
		elif y_test[i//2] == seizure_type:
			if preds[i] == 0:
				err1 = err1 + 1
			count = count + 1
		else:
			continue

	print(err0,err1,count)
	print(1-float(err0+err1)/count)

	Acc_sum[k] = 1-float(err0+err1)/count

	# save distance and perdiction result
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k), preds)

	print('*'*20)
	#print(distanceVectorsS1[0,:])
	#print(distanceVectorsS0[0,:])

print(Acc_sum)
