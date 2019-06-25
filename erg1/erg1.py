import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



data=pd.read_csv("iris.data",delimiter=',',header=None).values;
#print(data[:,0:5]);
NumberOfAttributes=len(data[0,:]);
print(NumberOfAttributes);
NumberOfPatterns=len(data);
#print(NumberOfPatterns);
map_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":0};
#print(map_dict);
x=data[:,0:NumberOfAttributes-1];
#print(x);
t=np.zeros(shape=(NumberOfPatterns));
#print(t);

i=0;

while i <=NumberOfPatterns-1 :
    if data[i,4] =="Iris-versicolor":
        t[i]=1;
    else:
        t[i]=0;
    i+=1;
#print(t);

#show
plt.figure(1);
plt.plot(data[0:49,0],data[0:49,2],'go');
plt.plot(data[50:99,0],data[50:99,2],'y+');
plt.plot(data[100:149,0],data[100:149,2],'m.');

n_folds=9;

plt.figure(2);
for folds in range(1,n_folds+1):
    
    xtrain,xtest,ttrain,ttest=train_test_split(data,t,test_size=0.1);
    plt.subplot(3,3,folds);
    plt.plot(xtrain[:,0],xtrain[:,2],'b.');
    plt.plot(xtest[:,0],xtest[:,2],'r.');











