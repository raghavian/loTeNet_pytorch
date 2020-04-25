import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb

class LIDC(Dataset):
	def __init__(self, rater=4, split='Train', data_dir = './', transform=None):
		super().__init__()

		self.data_dir = data_dir
		self.rater = rater
		self.transform = transform
		self.data, self.targets = torch.load(data_dir+split+'.pt')
		self.targets = self.targets.type(torch.FloatTensor)		   
	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):

		image, label = self.data[index], self.targets[index]
		if self.rater == 4:
			label = (label.sum() > 2).type_as(self.targets)
		else:
			label = label[self.rater]
		image = image.type(torch.FloatTensor)/255.0
		if self.transform is not None:
			image = self.transform(image)
		return image, label




