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
		# creation of a random itemMemory of 1 and 0 of dimension N_seats1xHD_dim
		if cuda:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1).cuda(device = device)
		else:
			self.proj_mat_LBP= torch.randn(self.HD_dim, self.N_seats1)
		self.proj_mat_LBP[self.proj_mat_LBP >0] = 1
		self.proj_mat_LBP[self.proj_mat_LBP <= 0] = 0
		
		if string == 'random':
			if cuda:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2).cuda(device = device)
			else:
				self.proj_mat_channels= torch.randn(self.HD_dim, self.N_seats2)
			self.proj_mat_channels[self.proj_mat_channels >=0] = 1
			self.proj_mat_channels[self.proj_mat_channels < 0] = 0


	def learn_HD_proj(self,EEG):
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
		## vector used to correctly weight the LBPs.
		LBP_weight = []
		for i in range(0,self.T-1):
			LBP_weight.append(2**i)
		LBP_weights = torch.cuda.ShortTensor(LBP_weight)
		#LBP_weights = torch.cuda.ShortTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
		## loop of creation of LBP histogram
		for iStep in range(learningEnd-self.T+1):
			# extraction of 7 consecutive samples of EEG to create the LBPs.
			x = EEG[:,iStep:(iStep+self.T)].short()
			# temporal difference between successive points: the result is a 6xn.Channels matrix
			bp = (torch.add(-x[:,0:self.T-1], 1,x[:,1:self.T])>0).short()
			# LBP creation: the result is a 1xn.Channel matrix with the integer value of the LBP for each channel.
			value = torch.sum(torch.mul(LBP_weights,bp), dim=1)
			#binding operation represented in the first part of central block of Fig.2. First we use the LBP to index the IM (C1...C64) and then we combine with the HV linked to the electrode number
			bindingVector=self.xor(self.proj_mat_channels,self.proj_mat_LBP[:,value.long()])
			# symbol of sum in fig.2
			output_vector=torch.sum(bindingVector,dim=1)
			#here we broke ties summing an additional HV in case of an even number
			if N_channels%2==0:
				output_vector = torch.add(self.xor(self.proj_mat_LBP[:,1],self.proj_mat_LBP[:,2]),1,output_vector)
			output_vector=(output_vector>int(math.floor(self.N_seats2/2))).short()
			# accumulation over time represented by the 0.5 seconds rectangle in Fig.2
			queeryVector=torch.add(queeryVector,1,output_vector)
		#final threshold block before H creation
		queeryVector = (queeryVector> (learningEnd-self.T+1)/2).short()
		return queeryVector

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



