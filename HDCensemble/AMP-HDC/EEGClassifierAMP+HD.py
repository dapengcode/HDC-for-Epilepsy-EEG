#!/usr/bin/env python

import sys
import numpy as np
import os
import scipy.io as scp
from EEGFunctions import HD_classifier   #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt

#Number of the GPU used. The script runs by exploiting cuda resources. This macro set the index of the GPU in your machine that will run the code.
device = 3
torch.cuda.set_device(device)

#directories with the dataset. You could change it based on the directory of your pc.
working_dir = '/home/dyp/iEEGSeizure2/dataset/'

slen = 1

Patient = 10
# 每个patient的seizure 数量，注意这里由于Sub13有两种seizure，这里先只尝试serzure_type=3
# 由于subj11的seizure7太长，所以舍弃
Patient_Seizure = [2,6,6,4,8,5] 

Acc_sum = np.zeros((int(Patient_Seizure[Patient-10])))
Distance_classHD = np.zeros((int(Patient_Seizure[Patient-10])))

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
	torch.manual_seed(1);
	#creating the model with the 2 itemMemory, both random and binarized. Creation of C_1...C_64 e E_1...E_n descrived in section 4.2
	model = HD_classifier(totalNumberBP,D,nSignals,T,device, 'random')

	# 记录每个label的数量
	num_noSeizure = 0
	num_Seizure = 0

	x_train = torch.from_numpy((x_train+1)*11).cuda() # 幅值由【-1 1】变为【0 22】

	# train the HDC
	print("Training Start")
	for i in range(0,x_train.shape[0]):
		#if num_Seizure == 2000:
		#	break
		if y_train[i] == seizure_type:
			if num_Seizure == 0:
				temp3=model.learn_HD_proj_rawAmplitude(x_train[i,:,0:250],slen) # 
				queeryVectorS1 = temp3 + 0
				num_Seizure = num_Seizure + 1
				temp3=model.learn_HD_proj_rawAmplitude(x_train[i,:,250:500],slen) # 
				queeryVectorS1 = torch.add(queeryVectorS1,temp3)
				num_Seizure = num_Seizure + 1
			else:
				temp3=model.learn_HD_proj_rawAmplitude(x_train[i,:,0:250],slen) # 
				queeryVectorS1 = torch.add(queeryVectorS1,temp3)
				num_Seizure = num_Seizure + 1
				temp3=model.learn_HD_proj_rawAmplitude(x_train[i,:,250:500],slen) # 
				queeryVectorS1 = torch.add(queeryVectorS1,temp3)
				num_Seizure = num_Seizure + 1
			
		if i%10000 == 0:
			print(i)

	i = 0
	while i < x_train.shape[0]:
		if y_train[i] == 0:
			if num_noSeizure == 0:
				temp1=model.learn_HD_proj_rawAmplitude(x_train[i,:,0:250],slen) # 
				queeryVectorS0 = temp1 + 0
				num_noSeizure = num_noSeizure + 1
				temp1=model.learn_HD_proj_rawAmplitude(x_train[i,:,250:500],slen) # 
				queeryVectorS0 = torch.add(queeryVectorS0,temp1)
				num_noSeizure = num_noSeizure + 1
				
			else:
				temp1=model.learn_HD_proj_rawAmplitude(x_train[i,:,0:250],slen) # 
				queeryVectorS0 = torch.add(queeryVectorS0,temp1)
				num_noSeizure = num_noSeizure + 1
				temp1=model.learn_HD_proj_rawAmplitude(x_train[i,:,250:500],slen) # 
				queeryVectorS0 = torch.add(queeryVectorS0,temp1)
				num_noSeizure = num_noSeizure + 1

		if i%10000 == 0:
			print(i)

		i = i+1


	# I sum and threshold all the intermediate prototype to create the final one stored in the associative memory (green+blu part of the Fig.2 --> Associative Memory (AM), Ictal)
	queeryVectorS0 = (queeryVectorS0>num_noSeizure/2).short()
	queeryVectorS1 = (queeryVectorS1>num_Seizure/2).short()


	# save the class HDV
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k)+'queeryVectorS0', np.array(queeryVectorS0.cpu().numpy()))
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k)+'queeryVectorS1', np.array(queeryVectorS1.cpu().numpy()))

	#print(queeryVectorS0)
	#print(queeryVectorS1)
	Distance_classHD[k] = np.array((model.ham_dist(queeryVectorS0,queeryVectorS1,D)).cpu().numpy())
	print(Distance_classHD[k])



	print(num_Seizure,num_noSeizure) 

	# test the HDC
	print("Testing Start")
	x_test = []
	y_test = []
	# read the testset and corresponding label

	x_test = np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy") # num*channel*sample
	y_test = np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy")

	num_test = int(x_test.shape[0]*2) # 1s的信号段分为了两个0.5s进行测试

	distanceVectorsS0 = torch.zeros(1,num_test).cuda() # no seizure
	distanceVectorsS1 = torch.zeros(1,num_test).cuda() # seizure

	#creation of the vector that goes out from the central box  of Fig.2 (HD computing: Encoding and Associative Memory)
	prediction0 = torch.zeros(1,num_test).cuda()

	x_test = torch.from_numpy((x_test+1)*11).cuda() # 幅值由【-1 1】变为【0 22】

	for i in range(0,num_test):
		if i%2 == 0:
			temp = model.learn_HD_proj_rawAmplitude(x_test[i//2,:,0:250],slen) # 前0.5second
			[distanceVectorsS1[0,i],distanceVectorsS0[0,i],prediction0[0,i]] = model.predict(temp, queeryVectorS1, queeryVectorS0, D)
		else:
			temp = model.learn_HD_proj_rawAmplitude(x_test[i//2,:,250:500],slen) # 后0.5second
			[distanceVectorsS1[0,i],distanceVectorsS0[0,i],prediction0[0,i]] = model.predict(temp, queeryVectorS1, queeryVectorS0, D)
		if i%10000 == 0:
			print(i)

	# calculate the ACC
	err0 = 0
	err1 = 0
	count = 0
	for i in range(0,num_test):
		if y_test[i//2] == 0:
			if prediction0[0,i] != 0:
				err0 = err0 + 1
			count = count + 1
		elif y_test[i//2] == seizure_type:
			if prediction0[0,i] == 0:
				err1 = err1 + 1
			count = count + 1
		else:
			continue

	print(err0,err1,count)
	print(1-float(err0+err1)/count)

	Acc_sum[k] = 1-float(err0+err1)/count


	# save distance and perdiction result
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k), np.array(prediction0.cpu().numpy()))
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k)+'_distanceS1', np.array(distanceVectorsS1.cpu().numpy()))
	np.save("./result/"+"Patient"+str(Patient)+'_k'+str(k)+'_distanceS0', np.array(distanceVectorsS0.cpu().numpy()))

	print('*'*20)
	#print(distanceVectorsS1[0,:])
	#print(distanceVectorsS0[0,:])

print(Acc_sum)
print('*'*20)
print(Distance_classHD)
np.save("./result/"+"Patient"+str(Patient)+'_slen'+str(slen)+'_T'+str(T)+'_Distance_classHD', np.array(Distance_classHD))
