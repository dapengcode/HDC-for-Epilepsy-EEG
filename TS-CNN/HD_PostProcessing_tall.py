import numpy as np
import scipy.io



Patient_Seizure = [2,6,6,4,2,3] 

# 这里的每个seizure都是前3min的间期，也就是3*60*2=360个0样本，后面都是1样本

# 确定判决使用的信号长度，采取多数判决法
predlen = [1,3,5,7,9,11,13]
#if Patient != 14:
#	predlen = [1,3,5,7,9,11,13,15,17,19,21,23,25]
#else:
#	predlen = [1,3,5,7,9,11,13,15,17,19]



for Patient in range(10,16): # subj no.10-15
	Acc = np.zeros((len(predlen),Patient_Seizure[Patient-10],3)) # 保存不同判决长度下，每一折的specifity和sencetivity
	for j in range(0,len(predlen)):
		for k in range(0,Patient_Seizure[Patient-10]):
			# 记录0/1样本数量
			count0 = 0
			count1 = 0
			# 记录0/1错误数量
			err0 = 0
			err1 = 0
			# read the result
			prediction = np.load('./result/'+"Patient"+str(Patient)+'_k'+str(k)+'.npy')
			#print(prediction.shape) #(1,num)
			# 先将0样本的预测值取出来计算specifity
			pred0 = prediction[0:360] 
			#print(pred0.shape)

			for i in range(0,pred0.shape[0]//predlen[j]):
				pred0_temp = np.sum(pred0[i*predlen[j]:(i+1)*predlen[j]])
				if pred0_temp >= predlen[j]/2.0:
					err0 = err0 + 1
				count0 = count0 + 1

			# 1样本的预测值取出来计算sencetivity
			pred1 = prediction[360:prediction.shape[0]]
			#print(pred1.shape)
			for i in range(0,pred1.shape[0]//predlen[j]):
				pred1_temp = np.sum(pred1[i*predlen[j]:(i+1)*predlen[j]])
				if pred1_temp < predlen[j]/2.0:
					err1 = err1 + 1
				count1 = count1 + 1
			#print(err0,count0,1-float(err0)/count0)
			#print(err1,count1,1-float(err1)/count1)
			Acc[j,k,0] = 1-float(err0)/count0
			Acc[j,k,1] = 1-float(err1)/count1
			Acc[j,k,2] = (Acc[j,k,0] + Acc[j,k,1])/2
			print(Acc[j,k,0],Acc[j,k,1])
			print('*'*20)
		print('#'*40)

	# 另存为mat矩阵后续分析
	scipy.io.savemat('result_tscnn_'+str(Patient)+'_t5.mat', mdict={'spe': Acc[:,:,0], 'sen':  Acc[:,:,1], 'acc':  Acc[:,:,2]})

'''
# 计算将每一折平均后的acc
for j in range(len(predlen)):
	acc = np.sum(Acc[j,:,2])
	print(acc/Patient_Seizure[Patient-10])
'''




