import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def evaluate(t,predict,criterion):
    tn=fn=tp=fp=float(0)
    
    for i in range(0,len(t)):
        if(predict[i]== False and t[i]==0):
            tn=tn+1
        if(predict[i]== True and t[i] == 1):
            tp=tp+1
        if(predict[i]== True and t[i]==0):
            fn=fn+1
        if(predict[i]==False and t[i]==1):
            fp=fp+1
    # Xrhsh try-catch. Se periptwsh pou ginei diairesh me to 0 tote apla epistrefw thn timh 0            
    try:
        mydict = {'accuracy':(tp+tn)/(tp+tn+fp+fn), 'precision':(tp)/(tp+fp), 'recall':(tp)/(tp+fn), 'fmeasure':((tp)/(tp+fp))/((tp)/(tp+fn)), 'sensitivity':(tp)/(tp+fn), 'specificity':(tn)/(tn+fp)}
    except ZeroDivisionError:
        return 0
    
    return mydict[criterion]

#read from the file
data=pd.read_csv("iris.data",delimiter=',',header=None).values;

NumberOfAttributes=len(data[0,:])
NumberOfPatterns=len(data)
accuracy = precision = recall = fmeasure = sensitivity = specificity = float(0)
map_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":0}


x=data[:,0:NumberOfAttributes-1]
t=np.zeros(shape=(NumberOfPatterns),dtype=float)
myClass=np.zeros(shape=(NumberOfPatterns),dtype=float)
option=0

gamma=input("Give the number of Gamma: \n")
C=input("Give the number of C \n")    
    
while option !=4:
    print("1.Separate Iris-setosa from Iris-virginica - Iris-versicolor\n");
    print("2.Separate Iris-versicolor from Iris-setosa - Iris-virginica\n");
    print("3.Separate Iris-virginica from Iris-setosa - Iris-versicolor\n");
    print('4.Exit\n');
    option=input("Choose option : \n");
    
    if option == 1:    
        t[0:49]=1
    elif option == 2:
        t[50:99]=1
    elif option==3:
        t[100:]=1
    elif option ==4:
        print "bye"
        break
    else:
        print "** Wrong option ** "
        break
    
    #option ofr the activation function
    flag=True
    while flag :
        print("1. Linear \n")
        print("2. Poly \n")
        print("3. RBF \n")
        print("4. sigmoid \n")
        kernelFun=input("Choose the kernel function: ")
        map_Kernel={1:"linear",2:"poly",3:"rbf",4:"sigmoid"}
        if(kernelFun == 1 or kernelFun == 2 or kernelFun == 3 or kernelFun == 4):
            flag=False
            kernel=map_Kernel[kernelFun]
        else:
            print("Please select on of the option above by pressing 1 or 2 or 3.")
    
    #start_Of_folds
    fig,subplt=plt.subplots(3,3);
    
    n_folds=9;
    for folds in range(0,n_folds):
    
        xtrain,xtest,ttrain,ttest=train_test_split(x,t,test_size=0.25)
    
        numberOfTrain=len(xtrain)
        numberOfTest=len(xtest)
        
        xtrain = np.array(xtrain, dtype=float)
        xtest = np.array(xtest, dtype=float)
        #initializing the netwotk and training it.
        network=SVC(C=C,kernel=kernel,gamma=gamma)
        network.fit(xtrain,ttrain)
        # get the prediction of the network.
        predict=network.predict(xtest)
        
        
        accuracy+=evaluate(ttest,predict,'accuracy')
        precision+=evaluate(ttest,predict,'precision')
        recall+=evaluate(ttest,predict,'recall')
        fmeasure+=evaluate(ttest,predict,'fmeasure')
        sensitivity+=evaluate(ttest,predict,'sensitivity')
        specificity+=evaluate(ttest,predict,'specificity')
    
    #plots
    
        subplt[(folds)/3, (folds)%3].plot(ttest, "ro")
        subplt[(folds)/3, (folds)%3].plot(predict, "b.")

        print('Mean accuracy for all folds is : %f\n',np.mean(accuracy))
        print('Mean precision for all folds is : %f\n',np.mean(precision))
        print('Mean recall for all folds is : %f\n',np.mean(recall))
        print('Mean f-measure for all folds is : %f\n',np.mean(fmeasure))
        print('Mean sensitivity for all folds is : %f\n',np.mean(sensitivity))
        print('Mean specificity for all folds is : %f\n',np.mean(specificity))
        print('\n');
        accuracy = precision = recall = fmeasure = sensitivity = specificity = float(0)