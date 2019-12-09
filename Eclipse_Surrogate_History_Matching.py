# -*- coding: utf-8 -*-
"""
Created on Tuesday November 05 12:05:47 2019
@author: Dr Clement Etienam
@External Collaborator: Dr Rossmary Villegas
@External Colaborator: Dr Oliver dorn
This code is for Ensemble Surrogate reservoir history matching
We also check the fidelity of our surrogate and use the surrogate in a gradeint
history matching approach
Size of reservoir data set is 84*27*4

"""
from __future__ import print_function
print(__doc__)

print('A NOVEL ENSEMBLE BASED HISTORY MATCHING WITH A DEEP LEARNING SURROGATE')
print('.........................IMPORT SOME LIBRARIES.....................')
from numpy import *
import numpy as np
import shutil
import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import interpolate
from scipy.stats import rankdata, norm
from sklearn.preprocessing import MinMaxScaler
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
import scipy.io 
from joblib import Parallel, delayed
import scipy.ndimage.morphology as spndmo
import datetime 
from collections import OrderedDict 
import multiprocessing
from multiprocessing import Pool
from keras.models import load_model
import os




print('')
print('-----------DEFINE SOME FUNCTIONS------------------------------------------') 
 ## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
##------------------------------------------------------------------------------------

def interpolatebetween(xtrain,cdftrain,xnew):
        numrows1=len(xnew)
        numcols = len(xnew[0])
        norm_cdftest2=np.zeros((numrows1,numcols))
        for i in range(numcols):
            a=xtrain[:,i]
            b=cdftrain[:,i]
            f = interpolate.interp1d(a, cdftrain[:,i],kind='linear',fill_value=(b.min(), b.max()),bounds_error=False)
            #f = interpolate.interp1d((xtrain[:,i]), cdftrain[:,i],kind='linear')
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
"""
def read_ecl(filename=None):

    out = struct
    if not exist(mstring('filename'), mstring('var')):
        error(mcat([mstring('\''), filename, mstring('\' does not exist')]))
        end
        # Open file
        fclose(mstring('all'))
        fid = fopen(filename)
        if fid < 3:
            error
            mstring('Error while opening file')
            end

            # Skip header
            fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))

            # Read one property at the time
            i = 0
            while not feof(fid):
                i = i + 1

                # Read field name (keyword) and array size
                keyword = deblank(fread(fid, 8, mstring('uint8=>char')).cT)
                keyword = strrep(keyword, mstring('+'), mstring('_'))
                num = fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))

                # Read and interpret data type
                dtype = fread(fid, 4, mstring('uint8=>char')).cT
                __switch_0__ = dtype
                if 0:
                    pass
                elif __switch_0__ == mstring('INTE'):
                    conv = mstring('int32=>double')
                    wsize = 4
                elif __switch_0__ == mstring('REAL'):
                    conv = mstring('single=>double')
                    wsize = 4
                elif __switch_0__ == mstring('DOUB'):
                    conv = mstring('double')
                    wsize = 8
                elif __switch_0__ == mstring('LOGI'):
                    conv = mstring('int32')
                    wsize = 4
                elif __switch_0__ == mstring('CHAR'):
                    conv = mstring('uint8=>char')
                    num = num * 8
                    wsize = 1
                    end

                    # Skip next word
                    fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))

                    # Read data array, which may be split into several consecutive
                    # arrays
                    data = mcat([])
                    remnum = num
                    while remnum > 0:
                        # Read array size
                        buflen = fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))
                        bufnum = buflen / wsize

                        # Read data and append to array
                        data = mcat([data, OMPCSEMI, fread(fid, bufnum, conv, 0, mstring('b'))])                    ##ok<AGROW>

                        # Skip next word and reduce counter
                        fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))
                        remnum = remnum - bufnum
                        end

                        # Special post-processing of the LOGI and CHAR datatypes
                        __switch_1__ = dtype
                        if 0:
                            pass
                        elif __switch_1__ == mstring('LOGI'):
                            data = logical(data)
                        elif __switch_1__ == mstring('CHAR'):
                            data = reshape(data, 8, mcat([])).cT
                            end
                            # Add array to struct. If keyword already exists, append data.
                            if isfield(out, keyword):

                            else:
                                data
                                end
                                # Skip next word
                                fread(fid, 1, mstring('int32=>double'), 0, mstring('b'))
                                end

                                fclose(fid)
                                end
"""                                
def pinvmat(A,tol = 0):
    V,S1,U = np.linalg.svd(A,full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = np.amax((A.size)*np.spacing(np.linalg.norm(S1,np.inf)))  
    
    r1 = sum(S1 > tol)+1
    v = V[:,:r1-1]
    U1 = U.T
    u = U1[:,:r1-1]
    S11 = S1[:r1-1]
    s = S11[:]
    S = 1/s[:]
    X = (u*S.T).dot(v.T)

    return (V,X,U)
def plot_history(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(7, 7))

    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def prepare_grid(nx,ny,nz,perm,poro,pressure,saturation,pre2,sat2):
    xc = np.linspace(1., nx, nx)
    yc = np.linspace(1., ny, ny)
    zc = np.linspace(1., nz, nz)
    
    xq, yq, zq = np.meshgrid(xc, yc, zc, indexing='ij')
    
    assert np.all(xq[:,0,0] == xc)
    assert np.all(yq[0,:,0] == yc)
    assert np.all(zq[0,0,:] == zc)
    xq=np.reshape(xq,(-1,1),'F')
    yq=np.reshape(yq,(-1,1),'F')
    zqall=np.zeros((nx*ny,4))
    for i in range(nz):
        zqall[:,i]=np.ravel(np.reshape(zq[:,:,i],(-1,1),'F'))
        
        
    zyes = np.array([])
    for i in range(nz):
        zyes = np.append(zyes, zqall[:,i],axis=0)
    zyes=np.reshape(zyes,(-1,1),'F')
#    temp=np.concatenate(xq,yq,zyes,perm,poro,pressure,saturation,pre2,sat2,axis=1)
    temp=np.stack([xq,yq,zyes,perm,poro,pressure,saturation,pre2,sat2], axis=1)
    return temp

def Plot_Production(x,true,machine,true2,machine2):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, machine, 'b', label='CCR')
    plt.plot(x, true, 'r', label='True model')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_o(Sm^{3}/day)')
    plt.ylim((0,25000))
    plt.title('Oil production')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    plt.plot(x, machine2, 'b', label='CCR')
    plt.plot(x, true2, 'r', label='True model')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_w(Sm^{3}/day)')
    plt.ylim((1,40))
    plt.title('Water production')
    plt.legend()
    os.chdir(Resultsf)
    plt.savefig("Surrogate_learn.pdf")
    os.chdir(oldfolder)
    plt.show()
    
def Peaceman_well(data,perm,poro,nx,ny,nz,A,Ausew,jigg):
    viscosity_water=0.5
    viscosity_oil=1.18
    oil_density=52
    water_density=62.4
    BHP=140
    water_injection_rate=15000
    Injector_location=np.array([9, 9])
    producer_location=np.array([69, 9])
    h1=100
    h2=100
    h3=50
    perm=np.reshape(perm,(nx,ny,nz),'F')
    poro=np.reshape(poro,(nx,ny,nz),'F')
    perm_producer=perm[69,9,3]
    poro_producer=poro[69,9,3]
    perm_injector=perm[9,9,3]
    poro_injector=poro[9,9,3]
#    % Rd=(sqrt((100^2+100^2))/2)*exp(-0.5*(3-atan(1)-atan(1)))
    Rd=0.3468*100
    Re=(0.14*(((1**0.5)*(100**2))+((1**0.5)*(100**2)))**0.5)/(0.5*(1**0.25)+(1**0.25));
    Qoil=np.zeros((15,4))
    Qwater=np.zeros((15,4))
    pi=3.142

    for i in range (15):
        Ause=A[i,:]
        jigguse=jigg[i,:]
        pressure=np.reshape(data[:,0,i],(84,27,4),'F')
        water_saturation=np.reshape(data[:,1,i],(84,27,4),'F')
        yess=0
        yessw=0

        for j in range(4):
            perm_producer=perm[69,9,j]    
            pressureg=pressure[69,9,j]
            
            water_saturationg=water_saturation[69,9,j]
            oil_saturationg=1-water_saturationg
            mixture_density=(oil_density*oil_saturationg)+ (water_density*water_saturationg)
            up=2*pi*mixture_density*h3*sqrt(perm_producer*perm_producer)
            mix_viscosity=(viscosity_oil+viscosity_water)/2
            down=mix_viscosity*(np.log(Re/Rd))
            deltap=((mixture_density*32.2*yess)/2)
            deltapw=((mixture_density*32.2*yessw))
            side=(BHP-deltap-pressureg)
            sidew=(BHP-deltapw-pressureg)
            oil_ratio=(oil_density*oil_saturationg)/mixture_density
            water_ratio=(water_density*water_saturationg)/mixture_density
            Qoil[i,j]=(((((up/down)*side)*oil_ratio)/oil_density)/1.6)*0.00001
            Qwater[i,j]=(((((up/down)*sidew)*water_ratio)/water_density)*0.000001)*jigguse
            yess=yess+Ause
            yessw=yessw+Ausew

    Qoil_true=np.mean(Qoil,axis=1)
    Qwater=np.mean(Qwater,axis=1)
    return Qoil_true,Qwater

def Create_input_stage_1(nx,ny,nz,perm2,poro2):
    xc = np.linspace(1., nx, nx)
    yc = np.linspace(1., ny, ny)
    zc = np.linspace(1., nz, nz)
    
    xq, yq, zq = np.meshgrid(xc, yc, zc, indexing='ij')
    
    assert np.all(xq[:,0,0] == xc)
    assert np.all(yq[0,:,0] == yc)
    assert np.all(zq[0,0,:] == zc)
    xq=np.reshape(xq,(-1,1),'F')
    yq=np.reshape(yq,(-1,1),'F')
    zqall=np.zeros((nx*ny,nz))
    for i in range(nz):
        zqall[:,i]=np.ravel(np.reshape(zq[:,:,i],(-1,1),'F'))
        
        
    zyes = np.array([])
    for i in range(nz):
        zyes = np.append(zyes, zqall[:,i])
    zyes=np.reshape(zyes,(-1,1),'F')
   
    
    #temp=np.concatenate((xq,yq,zyes,np.reshape(perm2,(-1,1),'F'),np.reshape(poro2,(-1,1),'F')), axis=1)
    temp=np.stack([xq,yq,zyes,np.reshape(perm2,(-1,1),'F'),np.reshape(poro2,(-1,1),'F')], axis=1)


    os.chdir(oldfolder)
    masterp='MASTERin0.out'
    np.savetxt(masterp,temp[:,:,0], fmt = '%4.7f',delimiter='\t', newline = '\n')
    return temp[:,:,0]

def Reservoir_Learning(ii,training_master):

    folder_trueccr = 'CCR_MACHINES'
    if ii==1:
        if os.path.isdir(folder_trueccr): # value of os.path.isdir(directory) = True
            shutil.rmtree(folder_trueccr)      
        os.mkdir(folder_trueccr)

    
    os.chdir(training_master)
    mat = scipy.io.loadmat('training_set.mat')
    train_set=mat['tempbig']
    os.chdir(oldfolder)
    
#    fillee='MASTER%d.out'%(ii)
    data=train_set[:,:,ii-1]

    input1=data[:,0:7]

    output=data[:,7:9]

    input1=gaussianizeit(input1) 
    scaler = MinMaxScaler()
    (scaler.fit(input1))
    input1=(scaler.transform(input1))
    

    inputtrain=(input1)
    numclement = len(input1[0])

    outputtrain=output
    outputtrain=np.reshape(outputtrain,(-1,2),'F')
    outputtrain=gaussianizeit(outputtrain) 
    ydamir=outputtrain
    scaler1 = MinMaxScaler()
    (scaler1.fit(ydamir))
    ydamir=(scaler1.transform(ydamir))
    print('')
    #-------------------#---------------------------------#
    
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
    
def Train_True_Model_CCR(nx,ny,nz,folder_true,oldfolder,true_master,training_master):
    
    if os.path.isdir(folder_true): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder_true)      
    os.mkdir(folder_true)
    shutil.copy2('POROVANCOUVER.DAT',folder_true)
    shutil.copy2('KVANCOUVER.DAT',folder_true)
    shutil.copy2('FAULT.DAT',folder_true)
    shutil.copy2('MASTER0.DATA',folder_true)
    shutil.copy2('Read_off.m',folder_true)
    shutil.copy2('read_ecl.m',folder_true)
    
    os.chdir(oldfolder)

    
    os.chdir(true_master)
    os.system("@eclrun eclipse MASTER0.DATA")
    ROLAND ="matlab -r Read_off"
    os.system(ROLAND)
    
    time.sleep(30)    # pause 5.5 seconds
    response= str(input('press Y if Matlab executed else wait '))
    Pressure=np.genfromtxt("Pressure_ensemble.out", dtype='float')
    saturation=np.genfromtxt("Saturation_ensemble.out", dtype='float')
    
    os.chdir(oldfolder)

    
    
    perm=np.genfromtxt("KVANCOUVER.DAT",skip_header = 1,skip_footer = 1, dtype='float')
    poro=np.genfromtxt("POROVANCOUVER.DAT",skip_header = 1,skip_footer = 1, dtype='float')
    perm=np.reshape(perm,(-1,1),'F')
    poro=np.reshape(poro,(-1,1),'F')
    tempbig=np.zeros((9072,9,15))
    for i in range (15):
#        i=0
        j=i+1

        filename = 'MASTER%d.out'%(j)
        temp=prepare_grid(nx,ny,nz,perm,poro,np.reshape(Pressure[:,i],(-1,1),'F'),np.reshape(saturation[:,i],(-1,1),'F'),np.reshape(Pressure[:,i+1],(-1,1),'F'),np.reshape(saturation[:,i+1],(-1,1),'F'))
        os.chdir(true_master)
        np.savetxt(filename,temp[:,:,0], fmt = '%4.4f',delimiter='\t', newline = '\n')
        os.chdir(oldfolder)

        tempbig[:,:,i]=temp[:,:,0]
    os.chdir(training_master)
    sio.savemat('training_set.mat', {'tempbig':tempbig})
    os.chdir(oldfolder)
        
    for i in range(15):
        jj=i+1
        print('Begin for time',jj )
        
        Reservoir_Learning(jj,training_master)
        print('')
        print('End for time',jj )
    
    
    os.chdir(training_master)
    mat = scipy.io.loadmat('training_set.mat')
    train_set=mat['tempbig']
    os.chdir(oldfolder)

    data=train_set[:,:,0]

    input1=data[:,[0,1,2,3,4]]

    output=data[:,[5,6]]


    input1=gaussianizeit(input1) 
    scaler = MinMaxScaler()
    (scaler.fit(input1))
    input1=(scaler.transform(input1))
    

    inputtrain=(input1)
    numclement = len(input1[0])

    outputtrain=output
    outputtrain=np.reshape(outputtrain,(-1,2),'F')
    outputtrain=gaussianizeit(outputtrain) 
    ydamir=outputtrain
    scaler1 = MinMaxScaler()
    (scaler1.fit(ydamir))
    ydamir=(scaler1.transform(ydamir))

    
    filename1='regressor_0.h5'

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
#    pickle.dump(modelDNN, open(filename1, 'wb'))
    os.chdir(oldfolder)
    return perm,poro




def prediction_stage_1(fillee,masterp):
    
    
    data=fillee
    
    input1=data[:,[0,1,2,3,4]]
    inputini= input1   
    outputbig=data[:,[5,6]]
    
    output=np.reshape(outputbig,(-1,2),'F')
    
    scaler2 = MinMaxScaler()
    (scaler2.fit(output))
    print('')
    scaler = MinMaxScaler()
    input2=gaussianizeit(input1)
    input2= scaler.fit(input2).transform(input2)

    datatest=masterp
    inputtest=datatest
     
    clement=interpolatebetween(input1,input2,inputtest)

    
    inputtest=clement
    outputfirst=output
    output2=outputfirst
    scaler2 = MinMaxScaler()
    output2=gaussianizeit(output2)
    output2= scaler2.fit(output2).transform(output2)

    numrows=len(inputtest)    # rows of input
    numrowstest=numrows
    numcols = len(inputtest[0])
    
    
    #-------------------Regression prediction---------------------------------------------------#
    
    filename1='regressor_0.h5'

    clementanswer1=np.zeros((numrowstest,2))
    print('')    
    print('predict in series')
    
    os.chdir(os.path.join(oldfolder,'CCR_MACHINES'))
    loaded_model = load_model(filename1)
#    loaded_model = pickle.load(open(filename1, 'rb'))

    os.chdir(oldfolder)
    
    ##----------------------##------------------------##
    a00=inputtest
    a00=np.reshape(a00,(-1,numcols),'F')
    if a00.shape[0]!=0:
        clementanswer1=loaded_model .predict(a00)
    clementanswer1=interpolatebetween(output2,outputfirst,clementanswer1)
    print('')
        
    matrixyes1=np.concatenate((inputini,clementanswer1), axis=1)

    os.chdir(oldfolder)
    return matrixyes1
# -*- coding: utf-8 -*-
def Reservoir_Prediction(ii,training_master,inputdataa):


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

    
    output=np.reshape(outputbig,(-1,2),'F')
    
    scaler2 = MinMaxScaler()
    (scaler2.fit(output))


    
    print('')
    scaler = MinMaxScaler()
    input2=gaussianizeit(input1)
    input2= scaler.fit(input2).transform(input2)

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
    
    numrows=len(inputtest)    # rows of input
    numrowstest=numrows
    numcols = len(inputtest[0])

    
    #-------------------Regression prediction---------------------------------------------------#

#    filename1='regressor_%d.asv'%(ii)
    clementanswer1=np.zeros((numrowstest,2))


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
    
def honour2(rossmary, rossmaryporo,sgsim2,DupdateK,nx,ny,nz,N):


    print('  Reading true permeability field ')
    uniehonour = np.reshape(rossmary,(nx,ny,nz), 'F')
    unieporohonour = np.reshape(rossmaryporo,(nx,ny,nz), 'F')

    # Read true porosity well values

    aa = np.zeros((4))
    bb = np.zeros((4))


    aa1 = np.zeros((4))
    bb1 = np.zeros((4))

    
    # Read true porosity well values
    for j in range(4):
        aa[j] = uniehonour[9,9,j]
        bb[j] = uniehonour[69,9,j]


        aa1[j] = unieporohonour[9,9,j]
        bb1[j] = unieporohonour[69,9,j]


    # Read permeability ensemble after EnKF update
    A = np.reshape(DupdateK,(nx*ny*nz,N),'F')          # thses 2 are basically doing the same thing
    C = np.reshape(sgsim2,(nx*ny*nz,N),'F')

    # Start the conditioning for permeability
    print('   Starting the conditioning  ')

    output = np.zeros((nx*ny*nz,N))
    outputporo = np.zeros((nx*ny*nz,N))

    for j in range(N):
        B = np.reshape(A[:,j],(nx,ny,nz),'F')
        D = np.reshape(C[:,j],(nx,ny,nz),'F')
    
        for jj in range(4):
            B[9,9,jj] = aa[jj]
            B[69,9,jj] = bb[jj]
  
            D[9,9,jj] = aa1[jj]
            D[69,9,jj] = bb1[jj]
   
        
        output[:,j:j+1] = np.reshape(B,(nx*ny*nz,1), 'F')
        outputporo[:,j:j+1] = np.reshape(D,(nx*ny*nz,1), 'F')
    
    output[output >= 1500] = 1500         # highest value in true permeability
    output[output <= 0.05] = 0.05

    outputporo[outputporo >= 0.4] = 0.4
    outputporo[outputporo <= 0.05] = 0.05

    return (output,outputporo)

def ESMDA(sg,sgporo,f,Sim1,alpha,c,nx,ny,nz,No,N):



    sgsim11 = np.reshape(np.log(sg),(nx*ny*nz,N),'F')
    sgsim11poro = np.reshape(sgporo,(nx*ny*nz,N),'F')
    

    stddWOPR1 = 0.1*f[0]


    stddWWCT1 = 0.1*f[1]


    print('        Generating Gaussian noise ')
    Error1 = np.ones((No,N))                  
    Error1[0,:] = np.random.normal(0,stddWOPR1,(N))
    Error1[1,:] = np.random.normal(0,stddWWCT1,(N))


    Cd2 = (Error1.dot(Error1.T))/(N - 1)

    Dj = np.zeros((No, N))
    for j in range(N):
        Dj[:,j] = f + Error1[:,j]


    overall = np.zeros((2*nx*ny*nz + No,N))


    overall[0:nx*ny*nz,0:N] = sgsim11
    overall[nx*ny*nz:2*nx*ny*nz,0:N] = sgsim11poro
    overall[2*nx*ny*nz:2*nx*ny*nz + No,0:N] = Sim1


    Y = overall

    M = np.mean(Sim1, axis = 1)
    M2 = np.mean(overall, axis = 1)

    S = np.zeros((Sim1.shape[0],N))
    yprime = np.zeros(((2)*nx*ny*nz + No,N))
           
    for j in range(N):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2

    print ('    Updating the new ensemble')
    Cyd = (yprime.dot(S.T))/(N - 1)
    Cdd = (S.dot(S.T))/(N - 1)

    Usig,Sig,Vsig = np.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    print('  Update the new ensemble  ')
    #(V,X,U) = np.linalg.pinv((Cdd + (alpha*Cd2)))
    
    Ynew = Y + ((Cyd.dot(np.linalg.pinv((Cdd + (alpha*Cd2))))).dot(Dj - Sim1))

    print('   Extracting the active permeability fields ')
    value1 = Ynew[0:nx*ny*nz,0:N]

    DupdateK = np.exp(value1)
    DupdateK[DupdateK >= 1500]=1500
    DupdateK[DupdateK <= 100]=100
	
	
	
    sgsim2 = Ynew[nx*ny*nz:nx*ny*nz*2,0:N]
    sgsim2[sgsim2 >=0.45]=0.45
    sgsim2[sgsim2 <= 0.1]=0.1

    return sgsim2,DupdateK

def ESMDALocalisation2(sg,sgporo,f,Sim1,alpha,c,nx,ny,nz,No,N):

    print('      Loading the files ')
    ## Get the localization for all the wells

    A = np.zeros((84,27,4))
    for jj in range(5):
        A[9,9,jj] = 1
        A[69,9,jj] = 1


    print( '      Calculate the Euclidean distance function to the 6 producer wells')
    lf = np.reshape(A,(nx,ny,nz),'F')
    young = np.zeros((int(nx*ny*nz/nz),4))
    for j in range(4):
        sdf = lf[:,:,j]
        (usdf,IDX) = spndmo.distance_transform_edt(np.logical_not(sdf), return_indices = True)
        usdf = np.reshape(usdf,(int(nx*ny*nz/nz)),'F')
        young[:,j] = usdf

    sdfbig = np.reshape(young,(nx*ny*nz,1),'F')
    sdfbig1 = abs(sdfbig)
    z = sdfbig1
    ## the value of the range should be computed accurately.
      
    c0OIL1 = np.zeros((nx*ny*nz,1))
    
    print( '      Computing the Gaspari-Cohn coefficent')
    for i in range(nx*ny*nz):
        if ( 0 <= z[i,:] or z[i,:] <= c ):
            c0OIL1[i,:] = -0.25*(z[i,:]/c)**5 + 0.5*(z[i,:]/c)**4 + 0.625*(z[i,:]/c)**3 - (5.0/3.0)*(z[i,:]/c)**2 + 1

        elif ( z < 2*c ):
            c0OIL1[i,:] = (1.0/12.0)*(z[i,:]/c)**5 - 0.5*(z[i,:]/c)**4 + 0.625*(z[i,:]/c)**3 + (5.0/3.0)*(z[i,:]/c)**2 - 5*(z[i,:]/c) + 4 - (2.0/3.0)*(c/z[i,:])

        elif ( c <= z[i,:] or z[i,:] <= 2*c ):
            c0OIL1[i,:] = -5*(z[i,:]/c) + 4 -0.667*(c/z[i,:])

        else:
            c0OIL1[i,:] = 0
      
    c0OIL1[c0OIL1 < 0 ] = 0
      
    print('      Getting the Gaspari Cohn for Cyd') 
     
    schur = c0OIL1
    Bsch = np.tile(schur,(1,N))
      
    yoboschur = np.ones((2*nx*ny*nz + No,N))
     
    yoboschur[0:nx*ny*nz,0:N] = Bsch
    yoboschur[nx*ny*nz:2*nx*ny*nz,0:N] = Bsch

    sgsim11 = np.reshape(np.log(sg),(nx*ny*nz,N),'F')
    sgsim11poro = np.reshape(sgporo,(nx*ny*nz,N),'F')
    
    print('        Determining standard deviation of the data ')
    stddWOPR1 = 0.15*f[0]


    stddWWCT1 = 0.2*f[1]


    print('        Generating Gaussian noise ')
    Error1 = np.ones((No,N))                  
    Error1[0,:] = np.random.normal(0,stddWOPR1,(N))
    Error1[1,:] = np.random.normal(0,stddWWCT1,(N))


    Cd2 = (Error1.dot(Error1.T))/(N - 1)

    Dj = np.zeros((No, N))
    for j in range(N):
        Dj[:,j] = f + Error1[:,j]

    print('      Generating the ensemble state matrix with parameters and states ')
    overall = np.zeros((2*nx*ny*nz + No,N))


    overall[0:nx*ny*nz,0:N] = sgsim11
    overall[nx*ny*nz:2*nx*ny*nz,0:N] = sgsim11poro
    overall[2*nx*ny*nz:2*nx*ny*nz + No,0:N] = Sim1

    Y = overall

    M = np.mean(Sim1, axis = 1)
    M2 = np.mean(overall, axis = 1)

    S = np.zeros((Sim1.shape[0],N))
    yprime = np.zeros(((2)*nx*ny*nz + No,N))
           
    for j in range(N):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2

    print ('    Updating the new ensemble')
    Cyd = (yprime.dot(S.T))/(N - 1)
    Cdd = (S.dot(S.T))/(N - 1)

    Usig,Sig,Vsig = np.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    print('  Update the new ensemble  ')
    (V,X,U) = pinvmat((Cdd + (alpha*Cd2)),tol)
    
    Ynew = Y + yoboschur*((Cyd.dot(X)).dot(Dj - Sim1))

    print('   Extracting the active permeability fields ')
    value1 = Ynew[0:nx*ny*nz,0:N]

    DupdateK = np.exp(value1)

    sgsim2 = Ynew[nx*ny*nz:nx*ny*nz*2,0:N]

    return sgsim2,DupdateK

def main_ESMDA_covariance(observation,overallsim,rossmary,rossmaryporo,perm,poro,alpha,c,nx,ny,nz,N,No,Nt):
    
    sgsim = np.reshape(perm,(nx*ny*nz,N), 'F')
    sgsimporo = np.reshape(poro,(nx*ny*nz,N),'F')

    sg = sgsim
    sgporo = sgsimporo
    

    Sim11 = np.reshape(overallsim,(2,15,N), 'F')

    # History matching using ESMDA
    for i in range(Nt):
        print(' Now assimilating timestep %d '%(i+1))

        Sim1 = Sim11[:,i,:]
        Sim1 = np.reshape(Sim1,(No,N))

        f = observation[:,i]

#        (sgsim2,DupdateK) = ESMDALocalisation2 (sg,sgporo, f,Sim1,alpha,c)
        sgsim2,DupdateK = ESMDA (sg,sgporo, f,Sim1,alpha,c,nx,ny,nz,No,N)

        #(output,outputporo) = honour2(rossmary, rossmaryporo,sgsim2,DupdateK,nx,ny,nz,N)

        sg = np.reshape(DupdateK,(nx*ny*nz,N),'F')
        sgporo = np.reshape(sgsim2,(nx*ny*nz,N),'F')

        print('Finished assimilating timestep %d'%(i+1))

    sgassimi = DupdateK
    sgporoassimi = sgsim2

    mumyperm = np.reshape(sgassimi,(nx*ny*nz,N),'F')
    mumyporo = np.reshape(sgporoassimi,(nx*ny*nz,N),'F')


    return( mumyperm,mumyporo )
    
def Plot_ensmeble_mean(permmean,poromean,trueperm,trueporo,nx,ny,nz):
    permmean=(np.reshape(permmean,(nx,ny,nz),'F'))
    poromean=np.reshape(poromean,(nx,ny,nz),'F')
    trueperm=(np.reshape(trueperm,(nx,ny,nz),'F'))
    trueporo=np.reshape(trueporo,(nx,ny,nz),'F')
    XX, YY = np.meshgrid(np.arange(nx),np.arange(ny))  
    
    plt.figure(figsize=(8, 8))
    plt.subplot(4, 2, 1)
    plt.pcolormesh(XX.T,YY.T,trueperm[:,:,0],cmap = 'jet')
    plt.title('True Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(' K (mD)',fontsize = 13)
    plt.clim(100,1500)

    plt.subplot(4, 2, 2)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.title('mean Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)

    plt.subplot(4, 2, 3)
    plt.pcolormesh(XX.T,YY.T,trueperm[:,:,1],cmap = 'jet')
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(100,1500)
    
    plt.subplot(4, 2, 4)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,1],cmap = 'jet')
    plt.title('mean Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)

    plt.subplot(4, 2, 5)
    plt.pcolormesh(XX.T,YY.T,trueperm[:,:,2],cmap = 'jet')
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)
    
    plt.subplot(4, 2, 6)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,2],cmap = 'jet')
    plt.title('mean Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)

    plt.subplot(4, 2, 7)
    plt.pcolormesh(XX.T,YY.T,trueperm[:,:,3],cmap = 'jet')
    plt.title('True Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)
    
    plt.subplot(4, 2, 8)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,3],cmap = 'jet')
    plt.title('mean Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('K (mD)',fontsize = 13)
    plt.clim(100,1500)
    
    plt.tight_layout(rect = [0,0,1,0.95])
    plt.suptitle('Permeability comparison between mean and true model', fontsize = 25)
    os.chdir(Resultsf)
    plt.savefig("Comparison_Mean_true_perm.png")
    os.chdir(oldfolder)
    plt.show()
    
    
    plt.figure(figsize=(8, 8))
    plt.subplot(4, 2, 1)
    plt.pcolormesh(XX.T,YY.T,trueporo[:,:,0],cmap = 'jet')
    plt.title('True Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)

    plt.subplot(4, 2, 2)
    plt.pcolormesh(XX.T,YY.T,poromean[:,:,0],cmap = 'jet')
    plt.title('mean Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)

    plt.subplot(4, 2, 3)
    plt.pcolormesh(XX.T,YY.T,trueporo[:,:,1],cmap = 'jet')
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)
    
    plt.subplot(4, 2, 4)
    plt.pcolormesh(XX.T,YY.T,poromean[:,:,1],cmap = 'jet')
    plt.title('mean Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)

    plt.subplot(4, 2, 5)
    plt.pcolormesh(XX.T,YY.T,trueporo[:,:,2],cmap = 'jet')
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)
    
    plt.subplot(4, 2, 6)
    plt.pcolormesh(XX.T,YY.T,poromean[:,:,2],cmap = 'jet')
    plt.title('mean Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)

    plt.subplot(4, 2, 7)
    plt.pcolormesh(XX.T,YY.T,trueporo[:,:,3],cmap = 'jet')
    plt.title('True Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)
    
    plt.subplot(4, 2, 8)
    plt.pcolormesh(XX.T,YY.T,poromean[:,:,3],cmap = 'jet')
    plt.title('mean Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('unit',fontsize = 13)
    plt.clim(0.1,0.5)
    
    plt.tight_layout(rect = [0,0,1,0.95])
    plt.suptitle('Porosity comparison between mean and true model', fontsize = 25)
    os.chdir(Resultsf)
    plt.savefig("Comparison_Mean_true_poro.png")
    os.chdir(oldfolder)
    plt.show()
    
def Surrogate_Forward_problem(j):
    nx=84
    ny=27
    nz=4

    folder='Realization%d'%(j)
    fileperm='perm%d.out'%(j)
    fileporo='poro%d.out'%(j)
    filec='RSM.mat'
    os.chdir(os.path.join(oldfolder,folder))
    perm=np.reshape(np.genfromtxt(fileperm,dtype='float'),(nx,ny,nz),'F')
    poro=np.reshape(np.genfromtxt(fileporo, dtype='float'),(nx,ny,nz),'F')

    robust=np.zeros((nx*ny*nz,2,15))    
    masterp=Create_input_stage_1(nx,ny,nz,perm,poro)
    os.chdir(training_master)
    mat = scipy.io.loadmat('training_set.mat')
    train_set=mat['tempbig']
    os.chdir(oldfolder)
    data=train_set[:,:,0]
    inputdataa=prediction_stage_1(data,masterp)  
    for i in range(15):
        jj=i+1
        print('Predict for time ',jj )
        clementanswer,matrixyesout=Reservoir_Prediction(jj,train_set[:,:,i],inputdataa)
        robust[:,:,i]=clementanswer
        inputdataa=matrixyesout
        print('')
        print('End for time',jj )

    #Qoilccr_h,Qwatercc_h= Peaceman_well(rossmaryout,x[:,0],x[:,1],84,27,4,A,0.007583,jigg)
    A = np.array([11.41,9.50,9.376,9.29,9.256,9.23,9.24,9.29,9.34,9.37,8.42,7.88,7.06,6.67,6.49])
    A=np.reshape(A,(-1,1),'F')
    jigg = np.array([1.843,1.208,1.177,1.345,1.525,1.669,1.77,1.85,1.92,1.96,1.87,1.78,1.66,1.61,1.59])
    jigg=np.reshape(jigg,(-1,1),'F')


    Qoilccr,Qwaterccr= Peaceman_well(robust,perm,poro,nx,ny,nz,A,0.007583,jigg)
    tempbig=np.concatenate((np.reshape(Qoilccr,(-1,1),'F'),np.reshape(Qwaterccr,(-1,1),'F')), axis=1)
    os.chdir(os.path.join(oldfolder,folder))
    sio.savemat(filec, {'tempbig':tempbig})
    print(" Ensemble member %d has been processed"%(j))
    os.chdir(oldfolder)


print('......................BEGIN THE MAIN CODE...........................')
start = datetime.datetime.now()
print(str(start))
oldfolder = os.getcwd()
os.chdir(oldfolder)
nx=np.int(input(' Enter the size of the reservoir in x direction (84): '))
ny=np.int(input(' Enter the size of the reservoir in y direction(27): '))
nz=np.int(input(' Enter the size of the reservoir in z direction(4): '))
N = int(input(' Number of realisations(100) : '))
No = int(input(' Number of observations which are Qoil and Qwater(2) : '))
Nt = int(input(' Number of timesteps for history period(15) : '))
alpha = int(input(' Inflation paramter for ESMDA 4-8) : '))

folder_true = str(input('Enter the folder name you want to learn CCR from using Eclipse-use-True_model: '))

true_master =  os.path.join(oldfolder,folder_true)

training_true = str(input('Enter the folder name you want to save the training set: '))
if os.path.isdir(training_true): 
    shutil.rmtree(training_true)      
os.mkdir(training_true)
training_master =  os.path.join(oldfolder,training_true)
perm,poro=Train_True_Model_CCR(nx,ny,nz,folder_true,oldfolder,true_master,training_master)
perm_true=perm
poro_true=poro
print('-----------END OF TRAINING STEP---------------------------------------') 
print('')
print('-----------BEGIN PREDICTON STEP---------------------------------------')

robust=np.zeros((nx*ny*nz,No,Nt)) 
masterp=Create_input_stage_1(nx,ny,nz,perm,poro)
os.chdir(training_master)
mat = scipy.io.loadmat('training_set.mat')
train_set=mat['tempbig']
os.chdir(oldfolder)
data=train_set[:,:,0]
inputdataa=prediction_stage_1(data,masterp)  
for i in range(15):
    jj=i+1
    print('Predict for time ',jj )
    clementanswer,matrixyesout=Reservoir_Prediction(jj,train_set[:,:,i],inputdataa)
    robust[:,:,i]=clementanswer
    inputdataa=matrixyesout
    print('')
    print('End for time',jj )
os.chdir(true_master)
True_RSM=np.genfromtxt("MASTER0.RSM",skip_header = 8, dtype='float')
os.chdir(oldfolder)
timme=True_RSM[:,0]
True_oil=np.reshape(True_RSM[:,2],(-1,1),'F')
True_water=np.reshape(True_RSM[:,4],(-1,1),'F')

observat=np.concatenate((True_oil,True_water), axis=1)
observation=np.zeros((No,Nt))

observation[0,:]=np.ravel(True_oil.T)
observation[1,:]=np.ravel(True_water.T)


A = np.array([11.41,9.50,9.376,9.29,9.256,9.23,9.24,9.29,9.34,9.37,8.42,7.88,7.06,6.67,6.49])
A=np.reshape(A,(-1,1),'F')
jigg = np.array([1.843,1.208,1.177,1.345,1.525,1.669,1.77,1.85,1.92,1.96,1.87,1.78,1.66,1.61,1.59])
jigg=np.reshape(jigg,(-1,1),'F')
Qoilccr,Qwaterccr= Peaceman_well(robust,perm,poro,84,27,4,A,0.007583,jigg)

folder_resultss = 'Results'
if os.path.isdir(folder_resultss): # value of os.path.isdir(directory) = True
    shutil.rmtree(folder_resultss)      
os.mkdir(folder_resultss)
Resultsf =  os.path.join(oldfolder,folder_resultss)

os.chdir(Resultsf)
sio.savemat('Surrogate_predictions_dynamic.mat', {'robust':robust})
os.chdir(oldfolder)
Plot_Production(timme,True_oil,Qoilccr,True_water,Qwaterccr)

print('.................ENSEMBLE HISTORY MATCHING ROUTINE..........................')
print('')
print('ENSEMBLE BASED HISTORY MATCHING WITH ES-MDA')

          
for j in range(N):
    folder = 'Realization%d'%(j)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)            
            
perm_realii=np.reshape(np.genfromtxt("sgsim.out",dtype='float'),(nx*ny*nz,100),'F')
poro_realii=np.reshape(np.genfromtxt("sgsimporo.out", dtype='float'),(nx*ny*nz,100),'F')

perm_reali=perm_realii[:,0:N]
poro_reali=poro_realii[:,0:N]

perm = np.reshape(perm_reali,(nx*ny*nz,N),'F')
poro = np.reshape(poro_reali,(nx*ny*nz,N),'F')

decreasingnorm = np.zeros((N,alpha+1))

permbigall=np.zeros((nx*ny*nz,N,alpha))
porobigall=np.zeros((nx*ny*nz,N,alpha))               

for iyobo in range(alpha):
    print('Now running the code for assimilating Iteration %d '%(iyobo+1))   
#    # Loading Porosity and Permeability ensemble files
#    print(' Permeability and porosity fields')
#
#    if iyobo == 0:
#        print(' Loading the permeability and porosity fields')
##        perm_reali=np.reshape(np.genfromtxt("sgsim.out",dtype='float'),(nx*ny*nz,N),'F')
##        poro_reali=np.reshape(np.genfromtxt("sgsimporo.out", dtype='float'),(nx*ny*nz,N),'F')
#
#
#        perm = np.reshape(perm_reali,(nx*ny*nz,N),'F')
#        poro = np.reshape(poro_reali,(nx*ny*nz,N),'F')
#
#    else:
#        perm = np.reshape(mumyperm,(nx*ny*nz,N),'F')
#        poro = np.reshape(mumyporo,(nx*ny*nz,N),'F')

    os.chdir(oldfolder) # setting original directory

    for i in range(N):
        folder = 'Realization%d'%(i)
        filename1= 'perm%d.out'%(i)
        filename2= 'poro%d.out'%(i)
        os.chdir(os.path.join(oldfolder,folder))
        np.savetxt(filename2,poro[:,i], fmt = '%4.4f',delimiter='\t', newline = '\n')      
        np.savetxt(filename1,perm[:,i], fmt = '%4.4f',delimiter='\t', newline = '\n')

    
        os.chdir(oldfolder) # returning to original cd

    # Running Simulations in Parallel
    print( ' Use the surrogate Forward Problem' )
    os.chdir(oldfolder)
#    for i in range(N):
#        Surrogate_Forward_problem(i,nx,ny,nz)
    number_of_realisations = range(N)
##
#    p = Pool(multiprocessing.cpu_count())
#    p.map(Surrogate_Forward_problem,number_of_realisations)
#    p.close()
#    p.join()
    Parallel(n_jobs=4, verbose=50)(delayed(
        Surrogate_Forward_problem)(i)for i in number_of_realisations)

    print('  Plotting production profile')

    oldfolder = os.getcwd()
    os.chdir(oldfolder)


    WOPRA = np.zeros((Nt,N))    
    WWPRB = np.zeros((Nt,N))


    for i in range(N):
        folder = 'Realization%d'%(i)
        os.chdir(folder)
        mat = scipy.io.loadmat('RSM.mat')
        train_set=mat['tempbig']
    
        WOPR1 = train_set[:,0]

        Time = timme
     
        WWPR1 = train_set[:,1]
        
    
        WOPRA[:,i] = WOPR1
        WWPRB[:,i] = WWPR1
        	
        os.chdir(oldfolder)

    os.chdir(oldfolder)
    
    TO1 = True_oil

    TW1 = True_water
    
      # To group all the plots with realisations together for legend display purposes
    #subplot(2,2,1)
    plt.plot(Time,WOPRA[:,0:N],color = 'c', lw = '2', label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('$Q_o$($Sm^{3}$/day)')
    plt.ylim((5000,25000))
    plt.title('Producer')

    plt.plot(Time,TO1, color = 'red', lw = '2', label ='True model' )
    plt.axvline(x = 1500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    os.chdir(Resultsf)
    plt.savefig('PRO_OIL_assimi%d'%(iyobo+1))
    os.chdir(oldfolder)          # save as png
    #plt.savefig('PRO1_OIL_assimi%d.eps'%(iyobo+1))     # This is for matplotlib 2.1.2
    #plt.show()                                     # preventing the figures from showing
    plt.clf()                                       # clears the figure
                      
    #subplot(2,2,2)
    plt.plot(Time, WWPRB[:,0:N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('$Q_w$($Sm^{3}$/day)')
    plt.ylim((1,40))
    plt.title('Producer')

    plt.plot(Time,TW1, color = 'red', lw = '2', label ='True model' )
    plt.axvline(x = 1500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    os.chdir(Resultsf)
    plt.savefig('PRO_WATER_assimi%d'%(iyobo+1))
    os.chdir(oldfolder)
    #plt.savefig('PRO2_OIL_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()
  
 
                 
    EWOP1 = np.zeros((N,1))
    EWWP1 = np.zeros((N,1))
    
    for i in range(N):
        EWOP1[i:i+1,:] = np.mean((WOPRA[:,i:i+1] - np.reshape(TO1,(Nt,1)))**2)
        EWWP1[i:i+1,:] = np.mean((WWPRB[:,i:i+1] - np.reshape(TW1,(Nt,1)))**2)
       
    TOTALERROR = np.ones((N,1))

    TOTALERROR = (EWOP1/np.std(TO1,ddof = 1))+(EWWP1/np.std(TW1,ddof = 1))

    TOTALERROR = TOTALERROR/Nt
    jj = np.amin(TOTALERROR)        #minimum of flattened array
    bestnorm = np.argmin(TOTALERROR)        #Index of minimum value in array
    # In MATLAB realisation 1 stored in column index 1, in Python stored in column 0
    print('The best norm is number %i ' %(bestnorm + 1) +'with value %4.4f'%jj)

    reali = np.arange(1,N+1)
    plttotalerror = np.reshape(TOTALERROR,(N))

    plt.bar(reali,plttotalerror, color = 'c')
    #plt.xticks(reali)
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.ylim(ymin = 0)
    plt.title('Cost function for Realizations')

    plt.scatter(reali,plttotalerror, color ='k')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.xlim([1,(N - 1)])
    os.chdir(Resultsf)
    plt.savefig('RMS %d iteration'%iyobo)
    os.chdir(oldfolder)
    #plt.savefig('RMS%d iteration'%iyobo,format = 'eps')
    #plt.show()
    plt.clf()

    decreasingnorm[:,iyobo] =plttotalerror
 
 
    print( 'Get the simulated files for all the time step from surrogate')

    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    overallsim = np.zeros((No,Nt,N))
    for ii in range(N):
        folder = 'Realization%d'%(ii)
        os.chdir(folder)
        mat = scipy.io.loadmat('RSM.mat')
        train_set=mat['tempbig']
    
        TO1 = train_set[:,0]
        TW1 = train_set[:,1]
        

        observationsim = np.zeros((No,Nt))

        for i in range(Nt):
            obs = np.zeros((No,1))
            obs[0,:] = TO1[i]        
            obs[1,:] = TW1[i]
            
                
            observationsim[:,i:i+1] = obs

        overallsim[:,:,ii] = observationsim 
        os.chdir(oldfolder)

    os.chdir(oldfolder)

    mumyperm,mumyporo = main_ESMDA_covariance(observation,overallsim,perm_true,poro_true,perm,poro,alpha,10,nx,ny,nz,N,No,Nt)
    

    perm = np.reshape(mumyperm,(nx*ny*nz,N),'F')
    poro = np.reshape(mumyporo,(nx*ny*nz,N),'F')
    
    permbigall[:,:,iyobo]=perm
    porobigall[:,:,iyobo]=poro
    print('Finished Iteration %d'%iyobo)


end = datetime.datetime.now()
timetaken = end - start
print(' Time taken : '+ str(timetaken))
print( '  Creating the output of permeability and porosity history matched model for the next run')
os.chdir(Resultsf)
np.savetxt('sgsimfinal.out', perm, fmt = '%4.4f', newline = '\n')
np.savetxt('sgsimporofinal.out', poro, fmt = '%4.4f', newline = '\n')     
np.savetxt('genesisNorm.out', decreasingnorm, fmt = '%4.4f', newline = '\n')
sio.savemat('permeability_evoltion_realizations', {'permbigall':permbigall})
sio.savemat('porosity_evoltion_realizations', {'porobigall':porobigall})
os.chdir(oldfolder)

zz=decreasingnorm[:,iyobo-1]
labelDA0=np.asscalar(np.ravel((np.asarray(np.where(zz == np.min(zz))))))
print('Now plot the permeability reconstruction')

Plot_ensmeble_mean(perm[:,labelDA0],poro[:,labelDA0],perm_true,poro_true,nx,ny,nz)

#
#
#response = input('Do you want to plot the permeability map ( Y/N ) ? ');
#
#if response == 'Y':
#    import plot3D
#else:
#    print('Pixel map not needed')
#
#print('  The overall programme has been executed  ')

##----------------------------------------------------------------------------------
## End part of preventing Windows from sleeping
if osSleep:
    osSleep.uninhibit()

os.chdir(oldfolder)

print('..............PROGRAM EXECUTED SUCCSFULY.............................')