import torch
def padding(p,image):
    out=torch.zeros((image.shape[0],image.shape[1]+2*p,image.shape[2]+2*p))
    out[:,p:out.shape[1]-p,p:out.shape[2]-p]=image
    return out

def Conv2D(image,kernel,p,stride):
    image=padding(p,image)
    kH, kW = kernel.shape[1],kernel.shape[2]
    H_out = int((image.shape[1] - kH) / stride) + 1
    W_out = int((image.shape[2] - kW) / stride) + 1
    output_image = torch.zeros(H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            start_i=stride*i
            end_i=start_i+kH
            start_j=stride*j
            end_j=start_j+kW
            reg=image[:,start_i:end_i,start_j:end_j]
            if reg.shape==kernel.shape:
             output_image[i,j]=torch.sum(reg*kernel)
    return output_image


def max_pooling(image,pooling_size,stride):
    h_out=int((image.shape[1]-pooling_size[0])/stride+1)
    w_out=int((image.shape[2]-pooling_size[0])/stride+1)
    output_img=torch.zeros(image.shape[0],h_out,w_out)
    for i in range(h_out):
        for j in range(w_out):
            start_i=stride*i
            end_i=start_i+pooling_size[0]
            start_j=stride*j
            end_j=start_j+pooling_size[0]
            for c in range(image.shape[0]):
             max_reg=torch.max(image[c,start_i:end_i,start_j:end_j] )
             output_img[c,i,j]=max_reg
    return output_img

               
class Flatten_layer:
    def __init__(self):
        pass

    def forward(self, x):
        shapes = 1
        for i in range(1, len(x.shape)):
            shapes *= x.shape[i]
        return torch.reshape(x, (x.shape[0], shapes))  




   




