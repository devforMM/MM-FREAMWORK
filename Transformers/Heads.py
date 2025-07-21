import torch

class Masked_HeadAttention:
    def __init__(self,dmodel,dim_head):
        self.wq = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wk = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wv = torch.randn(dmodel, dim_head, requires_grad=True)
        self.keys=None
        self.values=None
        self.queries=None
    def QKV_values(self,embedings):
        self.keys=embedings@self.wk
        self.queries=embedings@self.wq
        self.values=embedings@self.wv
    def get_score(self):
       seq_len=self.queries.shape[1]
       raw_scores=(self.queries@self.keys.transpose(-2,-1)/(self.queries.shape[1]**0.5))
       mask = torch.tril(torch.ones(seq_len, seq_len))
       raw_scores = raw_scores.masked_fill(mask == 0, float('-inf'))
       softmaxcore = torch.softmax(raw_scores, dim=-1)
       return softmaxcore@self.values

class masked_Multi_head_attention_layer:
    def __init__(self,dmodel,num_heads):
        self.wo=torch.randn(dmodel,dmodel,requires_grad=True)
        self.weights=[self.wo]
        self.heads=[Masked_HeadAttention(dmodel,dmodel//num_heads) for _ in range(num_heads)]
        for h in self.heads:
            self.weights.extend(
                [h.wk,h.wq,h.wv]
            )
    def forward(self,embedings):
        scores=[]
        for h in self.heads:
            h.QKV_values(embedings)
            scores.append(h.get_score())
        return torch.matmul(torch.cat(scores, dim=-1), self.wo)


class HeadAttention:
    def __init__(self,dmodel,dim_head):
        self.wq = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wk = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wv = torch.randn(dmodel, dim_head, requires_grad=True)
        self.keys=None
        self.values=None
        self.queries=None
    def QKV_values(self,embedings):
        self.keys=embedings@self.wk
        self.queries=embedings@self.wq
        self.values=embedings@self.wv
    def get_score(self):
       raw_scores=(self.queries@self.keys.transpose(-2,-1)/(self.queries.shape[1]**0.5))
       softmaxcore=torch.softmax(raw_scores,dim=-1)
       return softmaxcore@self.values


class Multi_head_attention_layer:
    def __init__(self,dmodel,num_heads):
        self.wo=torch.randn(dmodel,dmodel,requires_grad=True)
        self.weights=[self.wo]
        self.heads=[HeadAttention(dmodel,dmodel//num_heads) for _ in range(num_heads)]
        for h in self.heads:
            self.weights.extend(
                [h.wk,h.wq,h.wv]
            )
    def forward(self,embedings):
        scores=[]
        for h in self.heads:
            h.QKV_values(embedings)
            scores.append(h.get_score())
        return torch.matmul(torch.cat(scores, dim=-1), self.wo)

class cross_HeadAttention:
    def __init__(self,dmodel,dim_head):
        self.wo=torch.randn(dmodel,dmodel,requires_grad=True)
        self.wq = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wk = torch.randn(dmodel, dim_head, requires_grad=True)
        self.wv = torch.randn(dmodel, dim_head, requires_grad=True)
        self.keys=None
        self.queries=None
        self.values=None
    def QKV_values(self,encoder_embedings,decoder_embedings):
        self.queries = decoder_embedings @ self.wq
        self.keys    = encoder_embedings @ self.wk
        self.values  = encoder_embedings @ self.wv
    def get_score(self):
       raw_scores=(self.queries@self.keys.transpose(-2,-1)/(self.queries.shape[1]**0.5))
       softmaxcore=torch.softmax(raw_scores,dim=-1)
       return softmaxcore@self.values
    
    
class cross_Multi_head_attention_layer:
    def __init__(self,dmodel,num_heads):
        self.wo=torch.randn(dmodel,dmodel,requires_grad=True)
        self.weights=[self.wo]
        self.heads=[cross_HeadAttention(dmodel,dmodel//num_heads) for _ in range(num_heads)]
        for h in self.heads:
            self.weights.extend(
                [h.wk,h.wq,h.wv]
            )
    def forward(self,encoder_embedings,decoder_embedings):
        scores=[]
        for h in self.heads:
            h.QKV_values(encoder_embedings,decoder_embedings)
            scores.append(h.get_score())
        return torch.matmul(torch.cat(scores, dim=-1), self.wo)


