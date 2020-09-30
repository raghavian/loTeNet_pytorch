import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mps import MPS
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6
class loTeNet(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=2, virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim
		
		### Squeezing of spatial dimension in first step	
		self.kScale = 4
		nCh =  self.kScale**2 * nCh
		self.input_dim = self.input_dim // self.kScale

		self.nCh = nCh
		self.ker = kernel		 
		iDim = (self.input_dim // (self.ker))

		feature_dim = 2*nCh 
		
		### First level MPS blocks
		self.module1 = nn.ModuleList([ MPS(input_dim=(self.ker)**2,
			output_dim=self.virtual_dim, 
			nCh=nCh, bond_dim=bond_dim, 
			feature_dim=feature_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN1 = nn.BatchNorm1d(self.virtual_dim,affine=True)

		
		iDim = iDim // self.ker
		feature_dim = 2*self.virtual_dim
		
		### Second level MPS blocks
		self.module2 = nn.ModuleList([ MPS(input_dim=self.ker**2, 
			output_dim=self.virtual_dim, 
			nCh=self.virtual_dim, bond_dim=bond_dim,
			feature_dim=feature_dim,  parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
			for i in range(torch.prod(iDim))])

		self.BN2 = nn.BatchNorm1d(self.virtual_dim,affine=True)

		iDim = iDim // self.ker

		### Third level MPS blocks
		self.module3 = nn.ModuleList([ MPS(input_dim=self.ker**2,
			output_dim=self.virtual_dim, 
			nCh=self.virtual_dim, bond_dim=bond_dim,  
			feature_dim=feature_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN3 = nn.BatchNorm1d(self.virtual_dim,affine=True)

		### Final MPS block
		self.mpsFinal = MPS(input_dim=len(self.module3), 
				output_dim=output_dim, nCh=1,
				bond_dim=bond_dim, feature_dim=feature_dim, 
				adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, 
				parallel_eval=parallel_eval)
		
	def forward(self,x):
		b = x.shape[0] #Batch size

		# Increase input feature channel
		iDim = self.input_dim
		if self.kScale > 1:
			x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1])
			x = x.reshape(b,iDim[0],iDim[1],-1)

		# Level 1 contraction		 
		iDim = self.input_dim//(self.ker)
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]).reshape(b,
					self.nCh,(self.ker)**2,-1)
		y = [ self.module1[i](x[:,:,:,i]) for i in range(len(self.module1))]
		y = torch.stack(y,dim=2)
		y = self.BN1(y)

		# Level 2 contraction

		y = y.view(b,self.virtual_dim,iDim[0],iDim[1])
		iDim = (iDim//self.ker)
		y = y.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,self.virtual_dim,self.ker**2,-1)
		x = [ self.module2[i](y[:,:,:,i]) for i in range(len(self.module2))]
		x = torch.stack(x,dim=2)
		x = self.BN2(x)


		# Level 3 contraction
		x = x.view(b,self.virtual_dim,iDim[0],iDim[1])
		iDim = (iDim//self.ker)
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,self.virtual_dim,self.ker**2,-1)
		y = [ self.module3[i](x[:,:,:,i]) for i in range(len(self.module3))]

		y = torch.stack(y,dim=2)
		y = self.BN3(y)

		if y.shape[1] > 1:
		# Final layer
			y = self.mpsFinal(y)

		return y.view(b)


