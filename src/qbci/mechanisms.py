"""Post-lock diagnostic functions. Evaluation labels never select models."""
import numpy as np
from .kernels import rbf


def class_separation(features, labels):
    means, within = [], 0.0
    for label in (0,1):
        group = features[np.asarray(labels)==label]
        if len(group)<2: return float("nan")
        means.append(group.mean(axis=0))
        within += np.mean(np.sum((group-means[-1])**2,axis=1))
    return float(np.sum((means[1]-means[0])**2)/max(within,1e-15))


def mmd_unbiased(source, target, gamma):
    if min(len(source),len(target))<2: return float("nan")
    a,b,c=rbf(source,gamma=gamma),rbf(target,gamma=gamma),rbf(source,target,gamma)
    return float((a.sum()-np.trace(a))/(len(a)*(len(a)-1)) +
                 (b.sum()-np.trace(b))/(len(b)*(len(b)-1))-2*c.mean())


def kernel_sensitivity(kernel_function, x, parameters, step=1e-5):
    """Finite-difference Jacobian norms diagnose ineffective parameter groups."""
    result=[]
    for i in range(len(parameters)):
        plus,minus=parameters.copy(),parameters.copy()
        plus[i]+=step; minus[i]-=step
        derivative=(kernel_function(x,plus)-kernel_function(x,minus))/(2*step)
        result.append(float(np.linalg.norm(derivative)))
    return result
