"""
Author: Oren Kraus (https://github.com/okraus, September 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, July 2023)

"""

import numpy as np
import scipy
from copy import deepcopy
from scipy import ndimage
from sympy import nsolve, symbols, functions
from sympy.functions.special.gamma_functions import digamma
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
import mahotas
DEBUG = False

def I_MM_BEN(arrayFarRed,arrayRed):
    #I_MM_1 for cytoplasm based on far red (diff back from cyt)
    K_1=2
    Nh_1=7
    Kj_1=[1,1]
    Ui_1=[[0],[22500]]    # Initial MEANS
    Si_1=[[10000.0**2],[10000.0**2]]    # Initial VARIANCE
    Vi_1=[[1],[100]]     # Initial DEGREE OF FREEDOM for t dist
    Wi_1=[.7,.3]    
    
    I_MM_1,Ptotal=Run_MM(arrayFarRed,K_1,Nh_1,Kj_1,Ui_1,Si_1,Vi_1,Wi_1)
    #I_MM_2 for sept and Nuc

    # changed for thanasis
    K_2=4
    Nh_2=3
    Kj_2=[1,1,1,1]
    Ui_2=[[1000],[2000],[4000],[8000]]    # Initial MEANS
    Si_2=[[10000.0**2],[10000.0**2],[5000.0**2],[10000.0**2]]   # Initial VARIANCE
    Vi_2=[[100],[100],[1],[100]]      # Initial DEGREE OF FREEDOM for t dist
    Wi_2=[.6,.3,.02,.08]

    I_MM_2,Ptotal=Run_MM(arrayRed,K_2,Nh_2,Kj_2,Ui_2,Si_2,Vi_2,Wi_2)
    #I_MM = I_MM_1 + Nuc from I_MM_2
    
    
    K_3=3
    Nh_3=7
    Kj_3=[1,2,1]
    Ui_3=[[1000],[10000,20000],[50000]]    # Initial MEANS
    Si_3=[[10000.0**2],[10000.0**2,10000.0**2],[10000.0**2]]   # Initial VARIANCE
    Vi_3=[[100],[100,100],[100]]      # Initial DEGREE OF FREEDOM for t dist
    Wi_3=[.6,.3,.1]     

    I_MM_3,Ptotal=Run_MM(arrayRed,K_3,Nh_3,Kj_3,Ui_3,Si_3,Vi_3,Wi_3)

        
    #se2=nd.generate_binary_structure(2,15).astype(np.int)
    #pdb.set_trace()
    
    NucLab=mahotas.label(I_MM_3==2)[0]
    NucSizes=mahotas.labeled.labeled_size(NucLab)
    too_small_Nuc = np.where(NucSizes < 30)
    NucLab = mahotas.labeled.remove_regions(NucLab, too_small_Nuc)
    Nuc=NucLab>0    
    

    se5=np.ones((5,5)).astype(np.int)
    dilatedNuc=nd.binary_dilation(np.uint8(Nuc),se5)     
    #pdb.set_trace()
    
    I_MM_Sep=np.uint8((np.int8(I_MM_2==2)-np.int8(dilatedNuc))>0)
    
    #se3=np.ones((3,3)).astype(np.int) #changed for thanasis
    #I_MM_Sep=nd.binary_erosion(np.uint8(I_MM_Sep),se3) #changed for thanasis
    
    SepLab=mahotas.label(I_MM_Sep==1)[0]
    SepSizes=mahotas.labeled.labeled_size(SepLab)
    #too_small_Sep = np.where(SepSizes < 30)
    too_small_Sep = np.where(SepSizes < 15) #changed for thanasis
    SepLab = mahotas.labeled.remove_regions(SepLab, too_small_Sep)
    I_MM_Sep=SepLab>0        
    
    se7=np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]],dtype=int)
    I_MM_Sep=nd.binary_dilation(np.uint8(I_MM_Sep),se7) 
    
    
    Fgm=np.uint8(I_MM_1==1)
    se2=np.ones((2,2)).astype(np.int)
    Fgm=nd.binary_erosion(np.uint8(Fgm),se2)         
    
    Fgm_Lab,Fgm_num=nd.measurements.label(Fgm)
    Nuc_Loc_1d=np.where(np.ravel(Nuc==1))[0]
    Sep_Loc_1d=np.where(np.ravel(I_MM_Sep==1))[0]

    I_MM=Fgm+np.uint8(Nuc)
    
    # label number of cyt regions
    # count number of nucs and septins in each cyt region
    # replace nucs with septins as seeds
    '''
    Fgm=np.uint8(I_MM_1==1)
    Fgm_Lab,Fgm_num=nd.measurements.label(Fgm)
    Nuc_Loc_1d=np.where(np.ravel(Nuc==1))[0]
    for Lab in range(Fgm_num):
        Fgm_Loc_1d=np.where(np.ravel(Fgm_Lab==Lab))[0]
    
        overlap_nucs=np.zeros(np.shape(I_MM_1))
        overlap_nucs=np.ravel(overlap_nucs)
        overlap_nucs[(np.intersect1d(Fgm_Loc_1d,Nuc_Loc_1d))]=1
        overlap_nucs=np.reshape(overlap_nucs,np.shape(I_MM_1))
        overlap_nuc_lab,overlap_nuc_num = nd.measurements.label(overlap_nucs)
        
        if overlap_nuc_num==0:
            Fgm[Fgm_Lab==Lab]=0
        elsif overlap_nuc_num>2:
            Nuc[overlap_nucs==1]=0
            Fgm[Fgm_Lab==Lab]=0
            
    '''
    
    return I_MM, np.uint8(I_MM_Sep)    


def Run_MM(Iorig, K, Nh, Kj, Ui, Si, V_i, Wi):

    #####       Initial Parameters         ############
    
    Im_ad=(np.double(Iorig)*2**16/Iorig.max()).round()
    #K=2    # number of components
    #Kj=[1,1]   # t-distributions per component
    #Nh=7    # neighborhood for averaging
    max_iterations=100#100    # max_iterations to run
    M,N=Iorig.shape    # shape of image
    Eta_i=[]
    for k in range(K):
        Eta_i.append((1.0/Kj[k])*np.ones(Kj[k])); # mixing proportions for t distributions
  
    #Ui=[[0],[22500]]    # Initial MEANS
    #Si=[[10000.0**2],[10000.0**2]]    # Initial VARIANCE
    #V_i=[[1],[100]]     # Initial DEGREE OF FREEDOM for t dist
    
    B=12
    Bold=0
    Vsym=symbols('Vsym')

    learnRate=10**-6
    
    ##################################################
    
    ##### estimate initial mixing proportions ########
    
    #I_ad1=Iorig>100

    #nucNum=ndimage.measurements.label(I_ad1)[1] #estimate of nuclei  #####
    
    #meanNucSize=700;  #estimate of expected nucleus size  #####
    #meanCellSize=2000;  #estimate of expected cell size   #####
    
    #NucArea=nucNum*meanNucSize/(M*N)
    #CytArea=nucNum*meanCellSize/(M*N)
    '''    
    CytArea=I_ad1.sum()    
    BackArea=1-CytArea

    if BackArea<0.1:
        BackArea=0.1
        CytArea=1-BackArea
    if CytArea<0.1:
        CytArea=0.2
        BackArea=1-CytArea
        
    Wi=[ BackArea, CytArea]
    
    print Wi
    '''
    ##################################################   

    #####       Initialize Parameters         #########
    MP=np.ones((M,N,K))     # mixing proportions per pixel
    Z=np.zeros((M,N,K))     # posterior probability
    AveLocZ=np.zeros((M,N,K))    # posterior probability
    Temp0=np.zeros((M,N,K))
    Temp2=np.zeros((M,N,K))
    LogLike=np.zeros((max_iterations))
    Y=[]
    u=[]
    Temp1=[]
    StudPDFVal=[]
    for k in range(K):
        MP[:,:,k]=Wi[k]
        Y.append(np.zeros((M,N,Kj[k])))
        u.append(np.zeros((M,N,Kj[k])))
        Temp1.append(np.zeros((M,N,Kj[k])))
        StudPDFVal.append(np.zeros((M,N,Kj[k])))
    #print MP
    
    U=Ui
    S=Si
    Eta=Eta_i
    V=V_i
    
    ###################################################

    for iters in range(max_iterations):
    
    #########       RUN ITERATIONS         ############
    
        #################   E-STEP    #####################
        for k in range(K):
            temp=np.zeros((M,N))
            for m in range(Kj[k]):
                StudPDFVal[k][:,:,m]=StudPDF(Im_ad,U[k][m],S[k][m],V[k][m])
                temp=temp+Eta[k][m]*StudPDFVal[k][:,:,m]
                Y[k][:,:,m]=Eta[k][m]*StudPDFVal[k][:,:,m]
                u[k][:,:,m]=(V[k][m]+1)/(V[k][m]+(Im_ad-U[k][m])**2/S[k][m])
            Z[:,:,k]=MP[:,:,k]*temp
            ### more efficient maybe?
            
            sumYk=Y[k].sum(axis=2)
            for m2 in range(Kj[k]):
                Y[k][:,:,m2]=Y[k][:,:,m2]/sumYk
        sumZ=Z.sum(axis=2)
        for k in range(K):
            Z[:,:,k]=Z[:,:,k]/sumZ
        
        ###################################################
        
        #################   E-STEP    #####################
        for k in range(K):
            for m in range(Kj[k]):
                U[k][m]=(Z[:,:,k]*Y[k][:,:,m]*u[k][:,:,m]*Im_ad).sum()/(Z[:,:,k]*Y[k][:,:,m]*u[k][:,:,m]).sum();
                try:
                    V[k][m]=np.fabs(float(nsolve(-digamma(Vsym/2)+functions.log(Vsym/2)+1+\
                            ((Z[:,:,k]*Y[k][:,:,m]*(np.log(u[k][:,:,m])-u[k][:,:,m])).sum()/(Z[:,:,k]*Y[k][:,:,m]).sum())+\
                            digamma((V[k][m]+1)/2)-np.log((V[k][m]+1)/2),Vsym,1,verify=False)))
                except TypeError:
                    V[k][m]=np.fabs(float(nsolve(-digamma(Vsym/2)+functions.log(Vsym/2)+1+\
                            ((Z[:,:,k]*Y[k][:,:,m]*(np.log(u[k][:,:,m])-u[k][:,:,m])).sum()/(Z[:,:,k]*Y[k][:,:,m]).sum())+\
                            digamma((V[k][m]+1)/2)-np.log((V[k][m]+1)/2),Vsym,1,verify=False).as_real_imag()[0]))
                    V[k][m] = 100
                Eta[k][m]=(Z[:,:,k]*Y[k][:,:,m]).sum()/(Z[:,:,k]*Y[k].sum(axis=2)).sum()
            scipy.ndimage.uniform_filter(Z[:,:,k],size=Nh,output=MP[:,:,k],mode='constant')
            MP[:,:,k]=np.exp(B*MP[:,:,k])
            scipy.ndimage.uniform_filter(Z[:,:,k],size=Nh,output=AveLocZ[:,:,k],mode='constant')
        
        sumMP=MP.sum(axis=2)
        for k in range(K):
            MP[:,:,k]= MP[:,:,k]/sumMP
            for m in range(Kj[k]):
                if S[k][m]>500:
                    S[k][m]=(Z[:,:,k]*Y[k][:,:,m]*u[k][:,:,m]*(Im_ad-U[k][m])**2).sum()/(Z[:,:,k]*Y[k][:,:,m]).sum()
                else:
                    S[k][m]=500
        
        #### CORRECT COMPONENTS
        Utemp=deepcopy(U)
        Stemp=deepcopy(S)
        Vtemp=deepcopy(V)
        
        if K==2:
            if (max(U[0])>min(U[1])):
                indSwitch1=np.argmin(U[1])
                indSwitch2=np.argmax(U[0])
                U[0][indSwitch2]=Utemp[1][indSwitch1]
                S[0][indSwitch2]=Stemp[1][indSwitch1]
                V[0][indSwitch2]=Vtemp[1][indSwitch1]
                
                U[1][indSwitch1]=Utemp[0][indSwitch2]
                S[1][indSwitch1]=Stemp[0][indSwitch2]
                V[1][indSwitch1]=Vtemp[0][indSwitch2]
            
                Utemp=deepcopy(U)
                Stemp=deepcopy(S)
                Vtemp=deepcopy(V)
                
        elif K==3:
            if (max(U[0])>min(U[1])):
                indSwitch1=np.argmin(U[1])
                indSwitch2=np.argmax(U[0])
                U[0][indSwitch2]=Utemp[1][indSwitch1]
                S[0][indSwitch2]=Stemp[1][indSwitch1]
                V[0][indSwitch2]=Vtemp[1][indSwitch1]
                
                U[1][indSwitch1]=Utemp[0][indSwitch2]
                S[1][indSwitch1]=Stemp[0][indSwitch2]
                V[1][indSwitch1]=Vtemp[0][indSwitch2]
            
                Utemp=deepcopy(U)
                Stemp=deepcopy(S)
                Vtemp=deepcopy(V)
            
            if (max(U[2])<max(U[1])):
                indSwitch=np.argmax(U[1])
                U[2]=[Utemp[1][indSwitch]]
                S[2]=[Stemp[1][indSwitch]]
                V[2]=[Vtemp[1][indSwitch]]
                
                U[1][indSwitch]=Utemp[2][0]
                S[1][indSwitch]=Stemp[2][0]
                V[1][indSwitch]=Vtemp[2][0]    
            
        elif K==4:
            if (max(U[0])>min(U[1])):
                indSwitch1=np.argmin(U[1])
                indSwitch2=np.argmax(U[0])
                U[0][indSwitch2]=Utemp[1][indSwitch1]
                S[0][indSwitch2]=Stemp[1][indSwitch1]
                V[0][indSwitch2]=Vtemp[1][indSwitch1]
                
                U[1][indSwitch1]=Utemp[0][indSwitch2]
                S[1][indSwitch1]=Stemp[0][indSwitch2]
                V[1][indSwitch1]=Vtemp[0][indSwitch2]
            
                Utemp=deepcopy(U)
                Stemp=deepcopy(S)
                Vtemp=deepcopy(V)
            
            if (max(U[2])<max(U[1])):
                
                indSwitch1=np.argmin(U[2])
                indSwitch2=np.argmax(U[1])
                
                U[1][indSwitch2]=Utemp[2][indSwitch1]
                S[1][indSwitch2]=Stemp[2][indSwitch1]
                V[1][indSwitch2]=Vtemp[2][indSwitch1]
                
                U[2][indSwitch1]=Utemp[1][indSwitch2]
                S[2][indSwitch1]=Stemp[1][indSwitch2]
                V[2][indSwitch1]=Vtemp[1][indSwitch2]
            
                Utemp=deepcopy(U)
                Stemp=deepcopy(S)
                Vtemp=deepcopy(V)            
            
            if (max(U[3])<max(U[2])):
                
                indSwitch1=np.argmin(U[3])
                indSwitch2=np.argmax(U[2])
                
                U[2][indSwitch2]=Utemp[3][indSwitch1]
                S[2][indSwitch2]=Stemp[3][indSwitch1]
                V[2][indSwitch2]=Vtemp[3][indSwitch1]
                
                U[3][indSwitch1]=Utemp[2][indSwitch2]
                S[3][indSwitch1]=Stemp[2][indSwitch2]
                V[3][indSwitch1]=Vtemp[2][indSwitch2]
            
        while np.fabs(B-Bold)>0.05:
            Bold=B
            expAveLocZ=np.exp(B*AveLocZ)
            SumExpLocZ=(AveLocZ*expAveLocZ).sum(axis=2)/expAveLocZ.sum(axis=2)
            for k in range(K):
                Temp0[:,:,k]=AveLocZ[:,:,k]-SumExpLocZ
            deltaH=-((Z*Temp0).sum(axis=2)).sum()
            Bnew=Bold-learnRate*deltaH
            B=Bnew
        
        
        for k in range(K):
            for m in range(Kj[k]):
                Temp1[k][:,:,m]=Eta[k][m]*StudPDFVal[k][:,:,m]
            
            Temp2[:,:,k]=MP[:,:,k]*Temp1[k].sum(axis=2)
        
        LogLike[iters]=np.log(Temp2.sum(axis=2)).sum()
        
        if iters>0:
            if DEBUG:
                print('iterations= ', iters, ' loglike= ', LogLike[iters], ' diffLike= ', np.fabs(LogLike[iters-1]-LogLike[iters]))
                print(U)
                print(S)
                print(V)
             
            Ptot=LogLike[iters]
            if np.fabs(LogLike[iters-1]-LogLike[iters])<3000: ######## THRESHOLD FOR EARLY STOPPING
                break
        
        Imout=np.argmax(Z,axis=2)
        
    return np.uint8(Imout), Ptot


def StudPDF(X,U,Covar,Dof):
    try:
        ProbT=(np.math.gamma(Dof/2.0+1/2.0)*(Covar)**(-1/2.0))/(np.sqrt(Dof*np.pi)*np.math.gamma(Dof/2.0))/(1+(X-U)**2/(Dof*Covar))**((Dof+1)/2.0)
    except OverflowError:
        Dof = float(str(Dof).split('+')[0].replace('e', ''))
        ProbT=(np.math.gamma(Dof/2.0+1/2.0)*(Covar)**(-1/2.0))/(np.sqrt(Dof*np.pi)*np.math.gamma(Dof/2.0))/(1+(X-U)**2/(Dof*Covar))**((Dof+1)/2.0)
    return ProbT
