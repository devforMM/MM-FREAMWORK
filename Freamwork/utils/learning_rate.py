import math

def learning_rate_scheduler(type, lr, epoch, total_epochs, final_lr=0.0001, warmup_steps=5):
    if type == "factor":
        return lr * 0.9

    if type == "multistep_scheduler":
        return lr / 2  

    if type == "cosine":
        return final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

    if type == "warmup":
        if epoch < warmup_steps:
            return lr * (epoch / warmup_steps)
        else:
            return lr  

    return lr  
