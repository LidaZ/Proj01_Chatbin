import numpy as np
from scipy import ndimage

debugFlag = 0

def CrossCorrelationNonNormalized (X,Y):
    if (X.shape != Y.shape) :
        print("X and Y must have the same shape.")
        return(0)
    
    # Add zeros at the end of the data-sequence.
    # This is to avoid aliasing of the data during FFT.
    Z = np.zeros(X.shape)
    tLength = Z.shape[0]
    XX = np.concatenate((X,Z),axis=0)
    XX_ft = np.fft.fft(XX,axis=0)

    # Branch process for cross-correlation and auto-correlation.
    # If X == Y (i.e. auto-correlation), the process can be accelarated by
    # skiping one FFT operation.
    if X is Y:
        YY_ft = XX_ft
        if debugFlag >= 1:
            print("Optimized code used.")
    else:
        YY = np.concatenate((Y,Z),axis=0)
        YY_ft = np.fft.fft(YY,axis=0)
        
    
    CC = np.fft.ifft(np.multiply(np.conj(XX_ft),YY_ft), axis=0)
    CC = CC[0:tLength]
    return(CC)  

def CrossCorrelationNormalized (X,Y):
    CC = CrossCorrelationNonNormalized(X,Y)
    DcSig = np.ones((X.shape[0],1,1))
    Denom = CrossCorrelationNonNormalized(DcSig,DcSig)
    CC= np.divide(CC,Denom)
    return(CC)  

def MaskedCorrelation1D(F,G, optLevel=2):    
    Mf = np.ones((F.shape))
    Mg = np.ones((G.shape))
    fmf_gmg = CrossCorrelationNonNormalized(F, G)
    if optLevel >= 2:
        fmf_mg = FastCorrWithMask(F, maskfirst=False)
        mf_gmg = FastCorrWithMask(G, maskfirst=True)
    else:
        fmf_mg  = CrossCorrelationNonNormalized(F, Mg)
        mf_gmg  = CrossCorrelationNonNormalized(Mf,G)
        
    # Compute mf_mg in a optimal way.
    if optLevel >= 1:
        # Fast algorithm
        mf_mg = CrossCorrelationNonNormalized(Mf[:,0,0], Mg[:,0,0])
        mf_mg = mf_mg.reshape(F.shape[0],1,1)
        if debugFlag >= 1:
            print("Optimal for mf_mg")
    else :
        #Slow but established method
        mf_mg   = CrossCorrelationNonNormalized(Mf,Mg)
        if debugFlag >= 1:
            print("Non Optimal for mf_mg")

    if optLevel >= 2:
        fmf2_mg = FastCorrWithMask(np.square(F), maskfirst=False)
        mf_gmg2 = FastCorrWithMask(np.square(G), maskfirst=True)
    else:
        fmf2_mg  = CrossCorrelationNonNormalized(np.square(F),Mg)
        mf_gmg2  = CrossCorrelationNonNormalized(Mf,np.square(G))
    
    cov=np.divide((np.subtract(fmf_gmg,np.divide(np.multiply(fmf_mg,mf_gmg),mf_mg))),mf_mg) #eq30 Checn2017
    KNL=np.ones((1,2,4))
    # COV=ndimage.convolve(np.real(cov),KNL, mode='constant')
    #splitting real and imag and collect them again
    cov_real=ndimage.convolve(np.real(cov),KNL, mode='constant')
    cov_imag=ndimage.convolve(np.imag(cov),KNL, mode='constant')
    COV=cov_real+1j*cov_imag
    
    var1=np.divide((np.subtract(fmf2_mg,(np.divide(np.square(fmf_mg ),mf_mg)))),mf_mg)   #eq31 Checn2017
    # VAR1=ndimage.convolve(np.real(var1),KNL, mode='constant')
    #splitting real and imag and collect them again
    var1_real=ndimage.convolve(np.real(var1),KNL, mode='constant')
    var1_imag=ndimage.convolve(np.imag(var1),KNL, mode='constant')
    VAR1=var1_real+1j*var1_imag
    
    var2=np.divide((np.subtract(mf_gmg2,(np.divide((np.square(mf_gmg)),mf_mg)))),mf_mg) #eq32 Checn2017
    # VAR2=ndimage.convolve(np.real(var2),KNL, mode='constant')
    
    #splitting real and imag and collect them again
    var2_real=ndimage.convolve(np.real(var2),KNL, mode='constant')
    var2_imag=ndimage.convolve(np.imag(var2),KNL, mode='constant')
    VAR2=var2_real+1j*var2_imag
    
    Denominator=np.multiply(np.sqrt(VAR1),np.sqrt(VAR2))
    
    # Numerator = np.subtract (fmf_gmg, np.divide(np.multiply(fmf_mg,mf_gmg),mf_mg))
    # Denominator = np.sqrt(np.multiply(np.subtract(fmf2_mg,np.divide(np.square(fmf_mg),mf_mg)), np.subtract(mf_gmg2,np.divide(np.square(mf_gmg),mf_mg)))) 
    CorrCoef_Fast= np.divide(COV, Denominator)
    return(CorrCoef_Fast)
    
def FastCorrWithMask(F, maskfirst=True):
    """
    This function compute the cross correlation function between
    F (1-3 D array) with 1D rectangle mask.
    maskFirst = ture, for M (*) F where M is the mask, (*) is correlation operation.
        It is (hopefully) equivalent with cogCrossCorrelationNonNormalized(M, F)
    maskFirst = false, for F (*) M where M is the mask, (*) is correlation operation.
        It is (I believe) 
        equivalent with cogCrossCorrelationNonNormalized(F, M)
    """
    Fall = np.sum(F,axis=0)
    if maskfirst is True:
        Fcumsum = np.nancumsum(F,axis=0)
    else:
        Fcumsum = np.nancumsum(F[::-1],axis=0)
    zeroarray = np.zeros(F[0].shape)
    Fcumsum_conc=np.insert(Fcumsum,0,zeroarray,axis=0)[:-1,:,:]
    result = np.subtract( Fall,Fcumsum_conc)
    return(result)

# Validation code
if __name__ == "__main__":
    pass
