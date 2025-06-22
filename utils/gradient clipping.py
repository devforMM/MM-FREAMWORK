def gradient_clipping(grad,borne):
    if grad.norm()>borne:
        grad*=borne/grad.norm()
    return grad