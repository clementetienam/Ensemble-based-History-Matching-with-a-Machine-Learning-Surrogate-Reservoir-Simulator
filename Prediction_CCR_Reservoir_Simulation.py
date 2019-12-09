# -*- coding: utf-8 -*-
def Reservoir_Prediction(ii,training_master,inputdataa):
    import numpy as np
    from keras.models import load_model
    import datetime
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    import os
    import multiprocessing
    from scipy.stats import rankdata, norm

    ##------------------------------------------------------------------------------------
    
    ## Start of Programme
    
    print( 'Reservoir simulation output prediction ')
    oldfolder = os.getcwd()
    cores = multiprocessing.cpu_count()
    print(' ')
    print(' This computer has %d cores, which will all be utilised in parallel '%cores)
    #print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
    print(' ')
    from scipy import interpolate
    start = datetime.datetime.now()
    print(str(start))
    def interpolatebetween(xtrain,cdftrain,xnew):
        numrows1=len(xnew)
        numcols = len(xnew[0])
        norm_cdftest2=np.zeros((numrows1,numcols))
    
        for i in range(numcols):
            a=xtrain[:,i]
            b=cdftrain[:,i]
            f = interpolate.interp1d(a, cdftrain[:,i],kind='linear',fill_value=(b.min(), b.max()),bounds_error=False)
            cdftest = f(xnew[:,i])
            norm_cdftest2[:,i]=np.ravel(cdftest)
        return norm_cdftest2
    
    def gaussianizeit(input1):
        numrows1=len(input1)
        numcols = len(input1[0])
        newbig=np.zeros((numrows1,numcols))
        for i in range(numcols):
            input11=input1[:,i]
            newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
            newbig[:,i]=newX.T
        return newbig
    

    	
    #------------------Begin Code-----------------------------------------------------------------#
    print('Load the input data you want to predict from')
    
    print('-------------------LOAD INPUT DATA---------------------------------')
    print('  Loading the training data wit X Y Z Cordinates ')
    oldfolder = os.getcwd()


    current_CCR_path =  oldfolder
    os.chdir(current_CCR_path)
    folder_trueccr = 'CCR_MACHINES'
    CCR_path =  os.path.join(oldfolder,folder_trueccr)
    os.chdir(CCR_path)
    filename1='regressor_%d.h5'%(ii)
    loaded_model = load_model(filename1)
#    loaded_model = pickle.load(open(filename1, 'rb'))
    os.chdir(oldfolder)
#    fillee='MASTER%d.out'%(ii)
    data=training_master
#    data=np.genfromtxt(fillee, dtype='float')
    input1=data[:,0:7]
    inputini= input1   
    outputbig=data[:,7:9]
    print('-------------------Predict for pressure-output 1-----------------')
    
    output=np.reshape(outputbig,(-1,2),'F')
    
    scaler2 = MinMaxScaler()
    (scaler2.fit(output))


    
    print('')
    scaler = MinMaxScaler()
    input2=gaussianizeit(input1)
    input2= scaler.fit(input2).transform(input2)
    
    print('  Loading the test data  ')
    datatest=inputdataa
#    datatest=np.genfromtxt(filletest, dtype='float')
    inputtest=datatest
 
    clement=interpolatebetween(input1,input2,inputtest)
    #inputtest=scaler.transform(inputtest)
    
    inputtest=clement
    outputfirst=output
    output2=outputfirst
    scaler2 = MinMaxScaler()
    output2=gaussianizeit(output2)
    output2= scaler2.fit(output2).transform(output2)
    print('')
    print('Standardize and normalize (make gaussian) the test data')
    

    
    
    numrows=len(inputtest)    # rows of input
    numrowstest=numrows
    numcols = len(inputtest[0])

    
    #-------------------Regression prediction---------------------------------------------------#

#    filename1='regressor_%d.asv'%(ii)
    clementanswer1=np.zeros((numrowstest,2))
    print('')    
    print('predict in series')

##----------------------##------------------------##
    a00=inputtest
    a00=np.reshape(a00,(-1,numcols),'F')
    if a00.shape[0]!=0:
        clementanswer1=loaded_model .predict(a00)
    os.chdir(current_CCR_path)
    
    clementanswer1=interpolatebetween(output2,outputfirst,clementanswer1)
    print('')
        
    matrixyes=np.concatenate((inputini,clementanswer1), axis=1)

    masterp='MASTER%d_prediciton.out'%(ii)
    os.chdir(os.path.join(oldfolder,'True_model'))
#    np.savetxt(masterp,matrixyes, fmt = '%4.7f',delimiter='\t', newline = '\n')
    os.chdir(oldfolder)
    matrixyes2=np.concatenate((inputini[:,0:5],clementanswer1), axis=1)

    os.chdir(oldfolder)

#    np.savetxt('MASTER_updated.out',matrixyes2, fmt = '%4.7f',delimiter='\t', newline = '\n')
    return clementanswer1,matrixyes2

    
    print('-------------------END PREDICTION PROGRAM----------------------------')    
    
    
    
    
    
    
    
    
    
