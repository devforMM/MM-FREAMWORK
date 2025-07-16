import torch
from torch.nn.functional import unfold
from CNN.Vectorised_Cnn_operations.Vectorised_Cnn_operations import *


class vec_Conv2D_layer:
   def __init__(self,C_in,C_out,kernel_size,padding,stride,initializer=None):
      self.C_in=C_in
      self.kernel_size=kernel_size
      self.p=padding
      self.stride=stride
      self.C_out=C_out
      if initializer !=None:
            if initializer=="Xaviernormal":
                init= Vec_Conv_XavierNormal()
            elif initializer=="HeNormal":
                init=Vec_Conv_HeNormal()
            elif initializer=="XavierUniform":
                init=Vec_Conv_XavierUniform()
            elif initializer=="HeUniform":
                init=Vec_Conv_HeUniform()
            self.kernels=init.initialize(C_out,C_in,kernel_size)
      else:
          self.kernels=torch.randn(C_out,C_in,self.kernel_size[0],self.kernel_size[1],requires_grad=True)
      self.w=self.kernels


    
      
   def forward (self,x):
        patches=unfold(x,self.kernel_size,padding=self.p,stride=self.stride)
        kernels=self.kernels.view(self.C_out,-1)
        res=kernels@patches
        return res.reshape(res.shape[0],res.shape[1],int(res.shape[2]**0.5),int(res.shape[2]**0.5))
    

# ✅ Correction made by ChatGPT (not by the student)
# This version fixes the issue of channel mixing when using unfold,
# by extracting patches per channel and applying max pooling correctly
# without losing the spatial or channel dimensions.

class Vec_Max_pool_layer:
    def __init__(self, size, stride):
        self.size = size  # Patch size (e.g., 2 or 3)
        self.stride = stride  # Stride between patches

    def forward(self, x):
        batch_size, C_in, H, W = x.shape
        k = self.size[0]
        s = self.stride

        # Extract patches per channel
        # Output shape: [batch_size, C_in, H_out, W_out, k, k]
        patches = x.unfold(2, k, s).unfold(3, k, s)

        # Flatten each patch to shape [batch_size, C_in, H_out, W_out, k*k]
        # Then apply max pooling over each patch
        res = patches.contiguous().view(batch_size, C_in, patches.shape[2], patches.shape[3], -1).max(dim=-1)[0]

        return res

    


