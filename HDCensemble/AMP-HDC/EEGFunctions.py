#!/usr/bin/env python


import time, sys


import numpy as np
import scipy.special
import math
import torch
import matplotlib.pyplot as plt


class HD_classifier:

	def __init__(self, N_seats1,HD_dim, N_seats2, T, device, string, cuda = True):
		''' Initialize an HD classifier using the torch library
		Parameters
		----------
		N_seats1: number of elements inside the LBP item memory
		HD_dim: dimension of the HD vectprs
		N_seats2: number of channels of the channels item memory
		T: LBP length + 1
		device: gpu to be used to create the itemMemory
		string: type of item Memory for the channel item Memory: random or sandwich
		cuda: this paramether is fixed to true by now. The code MUST be ran on GPU.
		'''
		self.training_done = False
		self.N_seats1 = N_seats1
		self.N_seats2 = N_seats2
		self.HD_dim = HD_dim
		self.T = T
		self.device = device
		
		
		if string == 'random':
			if cuda:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2).cuda(device = device)
			else:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2)
			self.proj_mat_channels[self.proj_mat_channels >=0] = 1
			self.proj_mat_channels[self.proj_mat_channels < 0] = 0

		# Constructin of level HDV
		self.MAXL = 22; # 一共22个level HDV
		initHV = torch.randint(low=0, high=2, size=(self.HD_dim,)).cuda(device = device)
		currentHV = initHV
		randomIndex = torch.randperm(self.HD_dim).cuda(device = device)
		self.proj_mat_level = torch.zeros(self.HD_dim,self.MAXL).cuda(device = device)
		for i in range(self.MAXL):
			self.proj_mat_level[:,i] = currentHV
			SP = self.HD_dim // (2 * self.MAXL)
			startInx = i * SP
			endInx = (i+1) * SP
			currentHV[randomIndex[startInx:endInx]] = (currentHV[randomIndex[startInx:endInx]] + 1)%2


	def predict(self,testVector,Ictalprot, Interictalprot,D):
		''' Prediction function of HD: it gives in addition to the class prediction also the
		distance from the 2 class prototypes.
		Parameters
		----------
		testVector: HV of the unlabled segment.
		Ictalprot: prototype (HD vector) for the ictal state.
		Interictalprot: prototype (HD vector) for the interictal state.
		Return
		------
		distanceVectorsS: distance from ictal Prototype
		distanceVectornS: distance from interictal prototype
		prediction: 1 for seizure, 0 for interictal.
		'''
		distanceVectorsS = self.ham_dist(testVector,Ictalprot,D)
		distanceVectorsnS = self.ham_dist(testVector,Interictalprot,D)
		if distanceVectorsS > distanceVectorsnS:
			prediction = 0
		else:
			prediction = 1
		return distanceVectorsS,distanceVectorsnS,prediction

	def xor(self,vec_a,vec_b):
		''' xor between vec_a and vec_b
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return
		------
		vec_c: vec_a xor vec_b
		'''
		vec_c = (torch.add(vec_a,vec_b) == 1).short()  # xor

		return vec_c

	def ham_dist(self,vec_a,vec_b,D):
		''' calculate relative hamming distance
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return
		------
		rel_dist: relative hamming distance
		'''
		vec_c = self.xor(vec_a,vec_b)

		rel_dist = torch.sum(vec_c) / float(D)

		return rel_dist

	# Test function for proj_mat_level
	#def test(self):
	#	for i in range(self.MAXL-1):
	#		print(self.ham_dist(self.proj_mat_level[:,i],self.proj_mat_level[:,i+1],self.HD_dim).cpu().numpy())


	def learn_HD_proj_rawAmplitude(self,EEG,slen):
		# 通过level HDV对信号幅度进行编码
		''' starting from EEG data, it creates the histogram of LBP for the target segment
		projecting it in the HD space.
		Parameters
		----------
		EEG: samples of EEG on multiple channels.
		Return
		------
		quueryVector: HV in which all the histograms among all the channels are encoded.
		'''
		queeryVector = torch.cuda.ShortTensor(1,1).zero_()
		N_channels,learningEnd = EEG.size()
		
		## 首先将这一段EEG信号的幅值进行归一化，幅值控制在【0，22】
		min_vals, _ = torch.min(EEG, dim=1, keepdim=True)
		max_vals, _ = torch.max(EEG, dim=1, keepdim=True)
		EEG = (EEG - min_vals)*(self.MAXL) / (max_vals - min_vals) 
		#print(EEG.cpu().numpy())

		## 构建这一段EEG信号的HDV
		for iStep in range(learningEnd):
			# 提取当前时刻所有通道的EEG序列
			x = EEG[:,iStep] 
			# 根据当前时刻每个channel的幅值，将其映射到level IM上
			value = x.floor() # 向下取整
			# 由于原信号幅值为【0，22】闭区间，所以可能取到22，这里检测如果有值为22，就令其为21
			if torch.sum(value>21) > 0:
				for i in range(N_channels):
					if value[i] == 22:
						value[i] = 21
			#binding operation represented in the first part of central block of Fig.2. First we use the LBP to index the IM (C1...C64) and then we combine with the HV linked to the electrode number
			bindingVector=self.xor(self.proj_mat_channels,self.proj_mat_level[:,value.long()])
			# symbol of sum in fig.2
			output_vector=torch.sum(bindingVector,dim=1)
			#here we broke ties summing an additional HV in case of an even number
			if N_channels%2==0:
				output_vector = torch.add(self.xor(self.proj_mat_LBP[:,1],self.proj_mat_LBP[:,2]),1,output_vector)
			output_vector=(output_vector>int(math.floor(self.N_seats2/2))).short()
			# accumulation over time represented by the 0.5 seconds rectangle in Fig.2
			queeryVector=torch.add(queeryVector,1,output_vector)
		#final threshold block before H creation
		queeryVector = (queeryVector> int(math.floor(learningEnd/2))).short()
		return queeryVector
