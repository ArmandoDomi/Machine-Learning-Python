import numpy as np
from scipy.stats import norm

def nbtrain( x , t ):
    
    x0 = x[np.where(t==0)]
    x1 = x[np.where(t==1)]
    
    t0 = len(x0)
    t1 = len(x1)
    
    prior_prob=np.zeros([2,1])
    prior_prob[0,0]=t0/float(len(t))
    prior_prob[1,0]=t1/float(len(t))
    
    mean_value = np.zeros([2,4])
    variance = np.zeros([2,4])

    for p in range(0,len(x[0,:])):
        
        #for the vectors that belong to class 0
        mean_value[0,p] = np.mean(x0[:,p])
        variance[0,p] = np.std(x0[:,p])
        #for the vectors that belong to class 1
        mean_value[1,p] = np.mean(x1[:,p])
        variance[1,p] = np.std(x1[:,p])
    
    model = {'prior':prior_prob,'mu':mean_value,'sigma':variance}
    return model

def nbpredict( x , model ):
    
    predict=np.zeros(len(x))
    
    for p in range(0,len(x)):
        
        L=model['prior'][1]/float(model['prior'][0])
        
        for i in range(0,len(x[0,:])):
            L=L*((norm.pdf(x[p,i],model['mu'][1,i],model['sigma'][1,i])/
                  float(norm.pdf(x[p,i],model['mu'][0,i],model['sigma'][0,i]))))
        if L > 1:
            predict[p]=1
    
    return predict