import numpy as np
import cv2


def hsv_to_rgb(H, S, V):
        """
        Converts HSL color array to RGB array
    
        H = [0..360]
        S = [0..1]
        V = [0..1]
    
        http://en.wikipedia.org/wiki/HSL_and_HSV#From_HSL
        
        Arguments:
            H: Hue
            S: Saturation
            V: Value (brightness)
    
        Returns:
            R, G, B in [0..255]
        """
        # chroma
        C = V * S
        
        # H' and X (intermediate value for the second largest component)
        Hp = H / 60.0
        X = C * (1 - np.absolute(np.mod(Hp, 2) - 1))
    
        # initilize with zero
        R = np.zeros(H.shape, float)
        G = np.zeros(H.shape, float)
        B = np.zeros(H.shape, float)
    
        # handle each case:
        mask = (Hp >= 0) == ( Hp < 1)
        R[mask] = C[mask]
        G[mask] = X[mask]
    
        mask = (Hp >= 1) == ( Hp < 2)
        R[mask] = X[mask]
        G[mask] = C[mask]
    
        mask = (Hp >= 2) == ( Hp < 3)
        G[mask] = C[mask]
        B[mask] = X[mask]
    
        mask = (Hp >= 3) == ( Hp < 4)
        G[mask] = X[mask]
        B[mask] = C[mask]
    
        mask = (Hp >= 4) == ( Hp < 5)
        R[mask] = X[mask]
        B[mask] = C[mask]
    
        mask = (Hp >= 5) == ( Hp < 6)
        R[mask] = C[mask]
        B[mask] = X[mask]
        
        # adding the same amount to each component, to match value
        m = V - C
        R += m
        G += m
        B += m
        
        # [0..1] to [0..255]
        R *= 255.0
        G *= 255.0
        B *= 255.0
    
        return R.astype(int), G.astype(int), B.astype(int)
    
def hsvToRgbImage(H, S, V):
    R, G, B = hsv_to_rgb(H, S, V)
    temp = np.concatenate([[R], [G], [B]], axis = 0)
    rgbImage = np.rollaxis(temp, 0, len(np.shape(R))+1).astype('uint8')
    return(rgbImage)


def valueRerange(img, inRange, outRange):
    inMin = inRange[0]
    inMax = inRange[1]
    outMin = outRange[0] 
    outMax = outRange[1]

    outImg = np.clip(img, inMin, inMax)
    outImg = ((outImg - inMin) / (inMax - inMin) * (outMax-outMin)) + outMin #120 to make the  maximum color as geren and max SV= 20 and min=0
    return (outImg)


def makePseudoColorImage(H = 0, S = 0, V = 0,
                         inputRanges =  [(43., 70.), (0., 0.), (10., 30.)],
                         outputRanges =[(0., 120.), (0., 0.), (0., 1.)], 
                         blurKernels = [(1,1), (1,1), (1,1)]):

    # Check the H, S, V volumes have the same shape.
    fullChList = [H, S, V]
    volList = []
    i = 0
    npArrayType = type(np.zeros(0))
    for vol in [H, S, V]:
        if type(vol) == npArrayType:
            volList.append(vol)
        i += 1
    
    volShape = volList[0].shape
    for vol in volList:
        if vol.shape != volShape :
            print("makePseudoColorImage: All volumes must have the same shape.")
    
    # Make pseudo color image (volume)
    # Initialize output image matrix (with three channels)
    outImage = np.zeros(volShape + (3,)) # (3ch, x or z size, z or x size)

    thisBscan = np.zeros((3,) + volShape[1:]) # (3ch, x or z size, z or x size)
    for bScanIndex in range(0, volShape[0]):
        chIndex = 0
        for chVol in fullChList:
            if type(chVol) ==  npArrayType:
                # If chVol is a real volume (not a scalar)...
                chBscan = chVol[bScanIndex]
                if blurKernels[chIndex] != (1,1):
                    chBscan = cv2.blur(chBscan, blurKernels[chIndex])
                chBscan = valueRerange(chBscan, inputRanges[chIndex], outputRanges[chIndex])
                thisBscan[chIndex] = chBscan
            else:
                a = valueRerange(chVol, inputRanges[chIndex], outputRanges[chIndex]) # Here chVol is a scalar
                thisBscan[chIndex] = np.ones(volShape[1:]) * a

            chIndex += 1
            
        outImage[bScanIndex] = hsvToRgbImage(thisBscan[0], thisBscan[1], thisBscan[2])
        
    return(outImage)

def makeHVCompiteImage(H, V, 
                       inputRanges = [(10., 30.), (43., 70.)],
                       outputRanges = [(0., 120.), (0., 1.)],
                       blurKernels = [(3,3), (1,1)]):
    outImage = makePseudoColorImage(H = H, S = 1., V = V,
                         inputRanges =  [inputRanges[0],  (0., 1.), inputRanges[1]],
                         outputRanges = [outputRanges[0],  (0., 1.), outputRanges[1]], 
                         blurKernels = [blurKernels[0], (1,1), blurKernels[1]])
    return(outImage)

def makeGrayImage(Vol,
                  inputRange = (43., 70.), outputRange = (0., 1.),
                  blurKernel = (1,1)):
    outImage = makePseudoColorImage(H = 0., S = 0., V = Vol,
                         inputRanges =  [(0., 1.), (0., 1.), inputRange],
                         outputRanges = [(0., 1.), (0., 1.), outputRange], 
                         blurKernels = [(1, 1), (1,1), blurKernel])
    return(outImage)
    
#----------------------------------------
# Test codes 
#----------------------------------------
if __name__ == "__main__":
    A1 = np.zeros((10,10,3))
    A2 = 5.
    A3 = np.zeros((10,10,3))
    A = makePseudoColorImage(H=A1, S=A2, V = A3)
