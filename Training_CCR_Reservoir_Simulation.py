
def Reservoir_Learning(ii,training_master):
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    import os
    import multiprocessing
    from scipy import interpolate
    import shutil
    from scipy.stats import rankdata, norm

    
    oldfolder = os.getcwd()
    cores = multiprocessing.cpu_count()
    print(' ')
    print(' This computer has %d cores, which will all be utilised in parallel '%cores)
    #print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
    print(' ')
    
    start = datetime.datetime.now()
    print(str(start))
    
    
    print('-------------------LOAD FUNCTIONS---------------------------------')
    def interpolatebetween(xtrain,cdftrain,xnew):
        numrows1=len(xnew)
        numcols = len(xnew[0])
        norm_cdftest2=np.zeros((numrows1,numcols))
        for i in range(numcols):
            f = interpolate.interp1d((xtrain[:,i]), cdftrain[:,i],kind='linear')
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
    

    
    
    def plot_history(history):

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(loss) + 1)
    
        plt.figure(figsize=(7, 7))

        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
    from keras.callbacks import EarlyStopping

    

    	
    print('-------------------BEGIN PROGRAM-----------------------------------')    
    ##---------------------Begin Program-------------------------------##
    oldfolder = os.getcwd()
    folder_trueccr = 'CCR_MACHINES'
    if ii==1:
        if os.path.isdir(folder_trueccr): # value of os.path.isdir(directory) = True
            shutil.rmtree(folder_trueccr)      
        os.mkdir(folder_trueccr)
    
    
    #------------------Begin Code-------------------------------------------------------------------#
    print('')
    print('-------------------LOAD INPUT DATA-----------------------------------')
    print('  Loading the ascii data ')
    import scipy.io
    os.chdir(training_master)
    mat = scipy.io.loadmat('training_set.mat')
    train_set=mat['tempbig']
    os.chdir(oldfolder)
    
#    fillee='MASTER%d.out'%(ii)
    data=train_set[:,:,ii-1]

    input1=data[:,0:7]

    output=data[:,7:9]

    
    print('')
    print('Standardize and normalize the input data')

    input1=gaussianizeit(input1) 
    scaler = MinMaxScaler()
    (scaler.fit(input1))
    input1=(scaler.transform(input1))
    

    inputtrain=(input1)
    numclement = len(input1[0])
    print('-------------------BEGIN MACHINE LEARNING----------------------------')


    outputtrain=output
    outputtrain=np.reshape(outputtrain,(-1,2),'F')
    outputtrain=gaussianizeit(outputtrain) 
    ydamir=outputtrain
    scaler1 = MinMaxScaler()
    (scaler1.fit(ydamir))
    ydamir=(scaler1.transform(ydamir))
    print('')
    #-------------------#---------------------------------#

    
    
    #-------------------Regression---------------------------------------------------#
    print('Learn regression of the clusters with different labels from k-means ' )
    print('')
    print('Start the regression')    
    print('')    
    print('')

    ##
    import multiprocessing
    import numpy as np
    #from sklearn.neural_network import MLPRegressor
    
    import os

    
    filename1='regressor_%d.h5'%(ii)

    def parad(filename1):
        from keras.layers import Dense
        from keras.models import Sequential
        np.random.seed(7)
        modelDNN = Sequential()
        modelDNN.add(Dense(200, activation = 'relu', input_dim = numclement))
        modelDNN.add(Dense(units = 420, activation = 'relu'))
        modelDNN.add(Dense(units = 21, activation = 'relu'))
        modelDNN.add(Dense(units = 2))
        modelDNN.compile(loss= 'mean_squared_error', optimizer='Adam', metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)         
        a0=inputtrain

    
        b0=ydamir

        if a0.shape[0]!=0 and b0.shape[0]!=0:
            history=modelDNN.fit(a0, b0,validation_split=0.01, batch_size = 50, epochs = 300,callbacks=[es])
        plot_history(history)
        os.chdir(os.path.join(oldfolder,'CCR_MACHINES'))
        modelDNN.save(filename1)
#        pickle.dump(modelDNN, open(filename1, 'wb'))
        os.chdir(oldfolder)
        
    parad(filename1)       
        
    print('')

        
    #Parallel(n_jobs=nclusters, verbose=50)(delayed(
    #    parad)(j)for j in number_of_realisations)
    
    
    os.chdir(oldfolder)
    print('-------------------END TRAINING PROGRAM------------------------------')    
    
    
