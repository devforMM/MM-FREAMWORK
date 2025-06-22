import math

def learning_rate_scheduler(type, lr, epoch, total_epochs=100, final_lr=0.0001, warmup_steps=5):
    if type == "factor":
        return lr * 0.9

    if type == "multistep_scheduler":
        return lr / 2  # ici div est 2 en général ou paramétrable

    if type == "cosine":
        return final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

    if type == "warmup":
        if epoch < warmup_steps:
            return lr * (epoch / warmup_steps)
        else:
            return lr  # ou tu peux enchaîner avec un autre scheduler ici si tu veux

    return lr  # par défaut on garde le même learning rate
