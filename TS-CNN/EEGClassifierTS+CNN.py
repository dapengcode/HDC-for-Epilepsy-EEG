#!/usr/bin/env python

import numpy as np


from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from inception import Classifier_INCEPTION

K.set_image_data_format('channels_last')


#directories with the dataset. You could change it based on the directory of your pc.
working_dir = '/home/dyp/iEEGSeizure2/dataset/'
#working_dir = 'E:/OneDrive - The University of Hong Kong/dataset/Epilepsy EEG Dataset/'

#slen = 1 # 这里没有用到

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

	# convert labels to one-hot encodings.
	y_train_2 = np.zeros((y_train.shape[0]*2))
	for i in range(0, y_train.shape[0]*2):
		if y_train[i//2] != 0:
			y_train_2[i] = 1
	y_train_2 = np_utils.to_categorical(y_train_2)

	# train the model
	x_train_2 = np.zeros((x_train.shape[0]*2, x_train.shape[2]//2, x_train.shape[1])) # 对于LSTM模型来说，channel和sample需要反过来
	for i in range(0,x_train.shape[0]*2):
		if i%2 == 0:
			x_train_2[i,:,:] = np.reshape(x_train[i//2,:,0:250],(250,nSignals))
		else:
			x_train_2[i,:,:] = np.reshape(x_train[i//2,:,250:500],(250,nSignals))

	X_train, X_validate, Y_train, Y_validate = train_test_split(x_train_2, y_train_2, test_size=0.1, random_state=1)
	
	ienet = Classifier_INCEPTION(output_directory=working_dir+'result/', input_shape=(250, nSignals), nb_classes=2,build=False)
	model = ienet.build_model(input_shape=(250, nSignals), nb_classes=2)
	model.summary()
	
	# compile the model and set the optimizers
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
	
	# set a valid path for your system to record model checkpoints
	checkpointer = ModelCheckpoint(filepath='./result/best_model_'+"Patient"+str(Patient)+'_k'+str(k)+'.h5', verbose=1,save_best_only=True)

	fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 100, verbose = 2, validation_data=(X_validate, Y_validate), callbacks=[checkpointer])
	
	
	# test the HDC
	print("Testing Start")

	x_test = np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy") # num*channel*sample
	y_test = np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy")

	x_test_2 = np.zeros((x_test.shape[0]*2,x_test.shape[2]//2,x_test.shape[1])) # 
	for i in range(0,x_test.shape[0]*2):
		if i%2 == 0:
			x_test_2[i,:,:] = np.reshape(x_test[i//2,:,0:250],(250,nSignals))
		else:
			x_test_2[i,:,:] = np.reshape(x_test[i//2,:,250:500],(250,nSignals))


	# load optimal weights
	model.load_weights('./result/best_model_'+"Patient"+str(Patient)+'_k'+str(k)+'.h5')

	probs       = model.predict(x_test_2)
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

print(Acc_sum)
