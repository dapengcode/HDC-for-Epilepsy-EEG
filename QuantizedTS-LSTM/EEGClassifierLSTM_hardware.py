#!/usr/bin/env python

import numpy as np


from EEGModels import EEGLSTM
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
import tensorflow as tf
K.set_image_data_format('channels_last')


#directories with the dataset. You could change it based on the directory of your pc.
working_dir = '/home/dyp/iEEGSeizure2/dataset/'
#working_dir = 'E:/OneDrive - The University of Hong Kong/dataset/Epilepsy EEG Dataset/'

#slen = 1 # 这里没有用到

Patient = 10
# 每个patient的seizure 数量，注意这里由于Sub13有两种seizure，这里先只尝试serzure_type=3
# 由于subj11的seizure7太长，所以舍弃
Patient_Seizure = [2,6,6,4,2,3] 

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

	x_test = np.load(working_dir+str(Patient)+"/x_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy") # num*channel*sample
	y_test = np.load(working_dir+str(Patient)+"/y_Patient"+str(Patient)+'_Seiz'+str(k+1)+".npy")

	nSignals = x_test.shape[1] # number of channels
	
	model = EEGLSTM(nb_classes = 2, Chans = nSignals, Samples = 250, dropoutType = 'Dropout')
	#model.summary()
	
	# compile the model and set the optimizers
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
	
	
	
	# test the HDC
	print("Testing Start")

	

	x_test_2 = np.zeros((x_test.shape[0]*2,x_test.shape[2]//2,x_test.shape[1])) # 
	for i in range(0,x_test.shape[0]*2):
		if i%2 == 0:
			x_test_2[i,:,:] = np.reshape(x_test[i//2,:,0:250],(250,nSignals))
		else:
			x_test_2[i,:,:] = np.reshape(x_test[i//2,:,250:500],(250,nSignals))


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

	preds = np.zeros((x_test_2.shape[0],), dtype=int)

	for i in range(0,x_test_2.shape[0]):
		test_data = np.expand_dims(x_test_2[i], axis=0).astype(input_details["dtype"])
		interpreter.set_tensor(input_details["index"], test_data)
		interpreter.invoke()
		output = interpreter.get_tensor(output_details["index"])[0]
		preds[i]       = output.argmax(axis = -1) 

	#probs       = model.predict(x_test_2)
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

print(Acc_sum)
