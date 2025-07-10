import torch

def accuracy(ytrue,ypred):
    return (ytrue.argmax(dim=1)==ypred.argmax(dim=1)).sum()/len(ytrue)*100
def recall(ytrue,ypred):
    tp=((ypred==1) & (ytrue==1)).sum().item()
    fn=((ypred==0) & (ytrue==1)).sum().item()
    return tp/(tp+fn)
def f1_score(ytrue,ypred):
    return 2*(precision(ytrue,ypred)*recall(ytrue,ypred))/(precision(ytrue,ypred)+recall(ytrue,ypred))
def precision(ytrue,ypred):
    tp=((ypred==1) & (ytrue==1)).sum().item()
    fp=((ypred==1) & (ytrue==0)).sum().item()
    return tp/(tp+fp)

