"""
Author: Oren Kraus (https://github.com/okraus, July 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, July 2023)
"""

import scipy.ndimage as nd
import numpy as np
import mahotas


def Watershed_MRF(Iin,I_MM):
    
    Sds=(I_MM==2)
    Fgm=(I_MM>0)
    
    SdsLab=mahotas.label(I_MM==2)[0]
    SdsSizes=mahotas.labeled.labeled_size(SdsLab)
    too_small_Sds = np.where(SdsSizes < 30)
    SdsLab = mahotas.labeled.remove_regions(SdsLab, too_small_Sds)
    Sds=SdsLab>0

    #Sds=nd.morphology.binary_opening(Sds,structure=np.ones((10,10))).astype(np.int)    
    #Sds=nd.morphology.binary_propagation(Sds,structure=np.ones((10,10))).astype(np.int)
    
    #Sds=mm.areaopen(Sds,30) ##should stay
       
    se2=nd.generate_binary_structure(2,2).astype(np.int)
    dilatedNuc=nd.binary_dilation(Sds,se2) 
    Fgm=(dilatedNuc.astype(np.int)+Fgm.astype(np.int))>0
    
    #Fgm=mm.areaopen(Fgm,30)  ##should stay

    FgmLab=mahotas.label(Fgm)[0]
    FgmSizes=mahotas.labeled.labeled_size(FgmLab)
    too_small_Fgm = np.where(FgmSizes < 30)
    FgmLab = mahotas.labeled.remove_regions(FgmLab, too_small_Fgm)
    Fgm=FgmLab>0
    
    se3=nd.generate_binary_structure(2,1).astype(np.int)
    Fgm=nd.binary_erosion(Fgm,structure=se3)
        
    Fgm_Lab,Fgm_num=nd.measurements.label(Fgm)    
    
    Nuc_Loc_1d=np.where(np.ravel(Sds==1))[0]
    for Lab in range(Fgm_num):
        Fgm_Loc_1d=np.where(np.ravel(Fgm_Lab==Lab))[0]
        if not bool((np.intersect1d(Fgm_Loc_1d,Nuc_Loc_1d)).any()):
            Fgm[Fgm_Lab==Lab]=0
            
    Im_ad=(np.double(Iin)*2**16/Iin.max()).round()
    #pdb.set_trace()
    Im_ad=nd.filters.gaussian_filter(Im_ad,.5,mode='constant')
    
    Im_ad_comp=np.ones(Im_ad.shape)    
    Im_ad_comp=Im_ad_comp*Im_ad.max()
    Im_ad_comp=Im_ad_comp-Im_ad
    mask=((Sds==1).astype(np.int)+(Fgm==0).astype(np.int))
    mask=nd.label(mask)[0]
    #LabWater=pymorph.cwatershed(np.uint16(Im_ad_comp),mask)
    LabWater=mahotas.cwatershed(np.uint16(Im_ad_comp),mask)
    back_loc_1d=np.where(np.ravel(Fgm==0))[0]
    for Lab in range(2,LabWater.max()):
        cell_Loc_1d=np.where(np.ravel(LabWater==Lab))[0]
        if  bool((np.intersect1d(cell_Loc_1d,back_loc_1d)).any()):
            LabWater[LabWater==Lab]=1   
    
    return LabWater
