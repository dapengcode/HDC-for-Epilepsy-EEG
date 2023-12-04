import numpy as np
import scipy.io
from sklearn.metrics import roc_auc_score

# Patient = 10
# 每个patient的seizure 数量，注意这里由于Sub13有两种seizure，这里先只尝试serzure_type=3
Patient_Seizure = [2,6,6,4,2,3] 

# 这里的每个seizure都是前3min的间期，也就是3*60*2=360个0样本，后面都是1样本

# 确定判决使用的信号长度，采取多数判决法
predlen = [1,3,5,7,9,11,13]
#if Patient != 14:
#	predlen = [1,3,5,7,9,11,13,15,17,19,21,23,25]
#else:
#	predlen = [1,3,5,7,9,11,13,15,17,19]


for Patient in range(10,16): # subj no.10-15
	# 计算latency
	latency = np.zeros((Patient_Seizure[Patient-10]))
	Acc = np.zeros((len(predlen),Patient_Seizure[Patient-10],4)) # 保存不同判决长度下，每一折的specifity和sencetivity,acc,以及auc
	for j in range(0,len(predlen)):
		for k in range(0,Patient_Seizure[Patient-10]):
			# 记录0/1样本数量
			count0 = 0
			count1 = 0
			# 记录0/1错误数量
			err0 = 0
			err1 = 0
			# read the result
			# 这里对test的结果进行联合
			prediction0 = np.load('./AMP-HDC/result/'+"Patient"+str(Patient)+'_k'+str(k)+'.npy')
			prediction1 = np.load('./LBP-HDC/result/'+"Patient"+str(Patient)+'_k'+str(k)+'.npy')
			#prediction2 = np.load('./test3/result/'+"Patient"+str(Patient)+'_k'+str(k)+'.npy')

			#prediction = (prediction0 + prediction1 + prediction2)/3.0
			prediction = (prediction0 + prediction1)/2.0

			#print(prediction.shape) #(1,num)
			# 先将0样本的预测值取出来计算specifity
			pred0 = prediction[0,0:360] 
			#print(pred0.shape)

			# 为了计算AUC，这里保存两个矩阵，一个是model prediction，一个是真实的label
			pred_auc = []
			pred_auc = np.array(pred_auc)
			label_auc = []
			label_auc = np.array(label_auc)
			point_auc = 0 # 当前数组指针

			for i in range(0,pred0.shape[0]//predlen[j]):
				pred0_temp = np.sum(pred0[i*predlen[j]:(i+1)*predlen[j]])
				label_auc = np.append(label_auc,0) # 此为无癫痫阶段
				if pred0_temp >= predlen[j]/2.0:
					err0 = err0 + 1
					pred_auc = np.append(pred_auc,1) # 当前模型预测发生了错误，原本为0判决为了1
				else:
					pred_auc = np.append(pred_auc,0) # 模型预测正确
				count0 = count0 + 1
				point_auc = point_auc + 1


			lat_point = 1 # 计算latency使用的变量

			# 1样本的预测值取出来计算sencetivity
			pred1 = prediction[0,360:prediction.shape[1]]
			#print(pred1.shape)
			for i in range(0,pred1.shape[0]//predlen[j]):
				pred1_temp = np.sum(pred1[i*predlen[j]:(i+1)*predlen[j]])
				label_auc = np.append(label_auc,1) # 此为癫痫发作阶段
				if pred1_temp < predlen[j]/2.0:
					err1 = err1 + 1
					pred_auc = np.append(pred_auc,0) # 模型预测错误
					if lat_point == 1:
						latency[k] = latency[k] + 5.5 #一次延时为5.5s
				else:
					lat_point = 0
					pred_auc = np.append(pred_auc,1) # 模型预测正确
				count1 = count1 + 1
				point_auc = point_auc + 1
			#print(err0,count0,1-float(err0)/count0)
			#print(err1,count1,1-float(err1)/count1)
			Acc[j,k,0] = 1-float(err0)/count0
			Acc[j,k,1] = 1-float(err1)/count1
			Acc[j,k,2] = (Acc[j,k,0] + Acc[j,k,1])/2
			print(Acc[j,k,0],Acc[j,k,1])
			print('*'*20)

			# 计算AUC
			Acc[j,k,3] = roc_auc_score(label_auc,pred_auc)

		print('#'*40)
	# 另存为mat矩阵后续分析
	scipy.io.savemat('result_lbpamphdc_'+str(Patient)+'_t5.mat', mdict={'spe': Acc[:,:,0], 'sen':  Acc[:,:,1], 'acc':  Acc[:,:,2]})

#print('spe',Acc[5,:,0], 'sen',  Acc[5,:,1], 'acc',  Acc[5,:,2])



'''
# 计算将每一折平均后的acc
for j in range(len(predlen)):
	acc = np.sum(Acc[j,:,2])
	print(acc/Patient_Seizure[Patient-10])


# 计算将每一折平均后的spe
for j in range(len(predlen)):
	spe = np.sum(Acc[j,:,0])
	print(spe/Patient_Seizure[Patient-10])


# 计算将每一折平均后的sen
for j in range(len(predlen)):
	sen = np.sum(Acc[j,:,1])
	print(sen/Patient_Seizure[Patient-10])
'''






