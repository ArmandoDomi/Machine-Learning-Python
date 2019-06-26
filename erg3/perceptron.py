import numpy as np
def perceptron (x , t,MAXEPOCHS, beta):
    w=np.random.rand(1,len(x))
    check=1
    epochs=1
    while( check==1 and epochs <= MAXEPOCHS):
        check=0
        for p in range(1,len(x+1)):
            
            u=np.dot(w,x[:,p])
            if(u > 0):
                y=1
            else:
                y=-1
            if(t[p]!= y):
                w=w+beta*(t[p]-y)*np.transpose(x[:,p])
                check=1
        epochs=epochs+1
    print("The number of epochs is "+str(epochs))
    return w    

