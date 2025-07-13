from CNN.Loop_based_cnn.Cnn_initializers import *
from CNN.Loop_based_cnn.Cnn_operations import *
    
class Conv_layer():
    def  __init__(self,c_in,c_out,kernel_size,padding,stride,initializer=None):

        self.C_in=c_in
        self.C_out=c_out
        self.padding=padding
        self.stride=stride
        self.kernels=[]
        if initializer !=None:
            if initializer=="Xaviernormal":
                init= Conv_XavierNormal()
            elif initializer=="HeNormal":
                init=Conv_HeNormal()
            elif initializer=="XavierUniform":
                init=Conv_XavierUniform()
            elif initializer=="HeUniform":
                init=Conv_HeUniform()
            for _ in range(c_out):
             self.kernels.append(
                 init.initialize(c_in,c_in,c_out,kernel_size)
            )
        else:
            for _ in range(c_out):
             self.kernels.append(
                 torch.randn(c_in,kernel_size[0],kernel_size[1],requires_grad=True)
            )

    def forward(self,batch):
        batch_results=[]
        for i in range(batch.shape[0]):
            res=[]
            for j in range(self.C_out):
                res.append(
                    Conv2D(batch[i],self.kernels[j],self.padding,self.stride)
                )
            res= torch.stack(res)
            batch_results.append(res)
        return torch.stack(batch_results)
    

class max_pool_layer():
    def __init__(self,pooling_size,stride):
        self.pooling_size=pooling_size
        self.stride=stride
    def forward(self,batch):
        results=[]
        for i in range(batch.shape[0]):
            results.append(
                max_pooling(batch[i],self.pooling_size,self.stride)
            )
        
        return torch.stack(results)


class Batch_norm_layer:
    def __init__(self,c_in):
        self.gamma=torch.ones(c_in,1,1,requires_grad=True)
        self.beta=torch.zeros(c_in,1,1,requires_grad=True)
        
    def forward(self,x):
        mean=torch.mean(x,dim=(0,2,3),keepdim=True)
        var=torch.var(x,dim=(0,2,3),keepdim=True)
        x_hat=(x-mean)/torch.sqrt(var+1e-5)

        return self.gamma*x_hat+self.beta