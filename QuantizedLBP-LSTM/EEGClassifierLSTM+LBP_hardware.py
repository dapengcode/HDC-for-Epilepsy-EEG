#!/usr/bin/env python

import sys
import numpy as np
import os
import scipy.io as scp
from iEEGFunctions import LBP_extractor   #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt

import tensorflow as tf


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

	# load the model
	model = LBPLSTM(nb_classes = 2, Chans = 1, Samples = totalNumberBP*nSignals, dropoutType = 'Dropout')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

	# load optimal weights
	model.load_weights('./result/best_model_'+"Patient"+str(Patient)+'_k'+str(k)+'.h5')


	# 将权重量化为8bit并预测结果

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]

	tflite_model_quant = converter.convert()

	# test the quant model
	interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]


	preds = np.zeros((X_test.shape[0],), dtype=int)

	for i in range(0,X_test.shape[0]):
		test_data = np.expand_dims(X_test[i], axis=0).astype(input_details["dtype"])
		interpreter.set_tensor(input_details["index"], test_data)
		interpreter.invoke()
		output = interpreter.get_tensor(output_details["index"])[0]
		preds[i]       = output.argmax(axis = -1) 


	#probs       = model.predict(X_test)
	#preds       = probs.argmax(axis = -1) 

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
	np.save("./result_hardware/"+"Patient"+str(Patient)+'_k'+str(k), preds)

	print('*'*20)
	#print(distanceVectorsS1[0,:])
	#print(distanceVectorsS0[0,:])

print(Acc_sum)
