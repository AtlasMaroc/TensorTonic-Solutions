import math

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """

    list_lr = []

    
    for i in range(warmup_steps):
        lr = base_lr * (i+1)/warmup_steps
        list_lr.append(lr)
    for i in range(total_steps - warmup_steps):
        progress = i / (total_steps - warmup_steps)   # 0 at start of decay, approaches 1 at the end
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        list_lr.append(lr)
        
    return list_lr