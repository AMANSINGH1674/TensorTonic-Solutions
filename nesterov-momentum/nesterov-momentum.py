import numpy as np
import math

def cosine_embedding_loss(x1, x2, label, margin):
    c = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return float(1 - c) if label == 1 else float(max(0, c - margin))

def warmup_decay_schedule(base_lr, warmup_steps, total_steps, step):
    if warmup_steps > 0 and step < warmup_steps: 
        return float(base_lr * step / warmup_steps)
    return float(base_lr * (total_steps - step) / (total_steps - warmup_steps))

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w, g, G = map(np.asarray, (w, g, G))
    G_new = G + g**2
    return w - (lr / np.sqrt(G_new + eps)) * g, G_new

def cosine_annealing_schedule(base_lr, min_lr, total_steps, step):
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps)))

def r2_score(y_true, y_pred) -> float:
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if np.all(t == t[0]): 
        return 1.0 if np.all(t == p) else 0.0
    return float(1 - np.sum((t - p)**2) / np.sum((t - np.mean(t))**2))

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w, v, grad = map(np.asarray, (w, v, grad))
    v_new = momentum * v + lr * grad
    return w - v_new, v_new