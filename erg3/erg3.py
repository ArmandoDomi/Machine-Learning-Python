import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import perceptron as pr

def evaluate(t,predict,criterion):
    tn=fn=tp=fp=float(0)
    '''
    tn = sum(predict[predict== False] & (t[t==0]))
    tp = sum(predict[predict==True] & (t[t==1]))
    fn = sum(predict[predict==True] & (t[t==0]))
    fp = sum(predict[predict==False] & (t[t==1]))
    '''
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


#diavasma apo to arxeio
data=pd.read_csv("iris.data",delimiter=',',header=None).values;
#plithis protipwn kai dianismaton 
NumberOfAttributes=len(data[0,:])
NumberOfPatterns=len(data)
accuracy = precision = recall = fmeasure = sensitivity = specificity = float(0)
map_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":0}

#arxikopoisi kai toy dianismatos x kai t
x=data[:,0:NumberOfAttributes-1]
t=np.zeros(shape=(NumberOfPatterns),dtype=float)
myClass=np.zeros(shape=(NumberOfPatterns),dtype=float)
option=0

while option !=4:
    print("1.Diaxwrismos Iris-setosa apo Iris-virginica - Iris-versicolor\n");
    print("2.Diaxwrismos Iris-versicolor apo Iris-setosa - Iris-virginica\n");
    print("3.Diaxwrismos Iris-virginica apo Iris-setosa - Iris-versicolor\n");
    print('4.exodos\n');
    option=input("Dwse epilogi : \n");
    
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
        print "** Lathos epilogi ** "
        break
    
    epochs=input("Give the number of max epochs: ")
    
    b=input("Give the number of learning rate: ")
    
    #epauxismos toy pinaka protipwn x
    X=np.hstack((x,np.ones((len(data),1))))

    #start_Of_folds
    fig,subplt=plt.subplots(3,3);
    
    n_folds=9;
    for folds in range(0,n_folds):
    
        xtrain,xtest,ttrain,ttest=train_test_split(X,t,test_size=0.25)
    
        numberOfTrain=len(xtrain)
        numberOfTest=len(xtest)
    
        ttrain1 = 2*ttrain - 1; # metatropi twn 0 se -1 kai to 1 paramenei 1
        ttest1 = 2*ttest - 1;   # metatropi twn 0 se -1 kai to 1 paramenei 1
    

        xtrain = np.array(xtrain, dtype=float)
        xtest = np.array(xtest, dtype=float)

        w=pr.perceptron(np.transpose(xtrain),ttrain1,epochs,b)
        y=np.dot(xtest,np.transpose(w))
        predict=(y>0)
        
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
