#!/usr/bin/env python


import time, sys


import numpy as np
import scipy.special
import math
import torch
import matplotlib.pyplot as plt


def LBP_extractor(EEG,totalNumberBP,label,T):
	''' starting from EEG data, it creates the histogram of LBP for the target segment.
	Parameters
	----------
	EEG: samples of EEG on multiple channels.
	T: LBP length + 1
	label: label of the segment--> during test we can pass a random value, it will not be used
	totalNumberBP: dimensionality of the histogram
	Return
	------
	Features_vector: in position 0 we have the label (for the training). In the other position
	the full histogram encoded, i.e. the features.
	'''
	N_channels,learningEnd = EEG.size()
	Features_vector = torch.FloatTensor(totalNumberBP*N_channels+1,1).zero_()
	LBP_weights = torch.FloatTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])
	for iStep in range(learningEnd-6):
		x = EEG[:,iStep:(iStep+T)].float()
		bp = (torch.add(-x[:,0:T-1], 1,x[:,1:T])>0).float()
		value = torch.sum(torch.mul(LBP_weights,bp), dim=1)+1
		index= torch.add(torch.mul(torch.FloatTensor(np.array(range(N_channels))),totalNumberBP),value)
		Features_vector[index.long()] = Features_vector[index.long()]+1
	Features_vector[0] = label
	return Features_vector
