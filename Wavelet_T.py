import cmath
import numpy as np
from math import log, ceil
import pylab as plt
import cv2

import numpy as np


img = cv2.imread('im_3.jpg')
# show original image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (512, 512))
cv2.imshow('test', img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

print(img)

## Haar's DWT transformation function

def HDWT(img, t, i):
    
    if(t == 0):
    
        return img

    else:
    
        x = np.array(img, dtype=float)
        w = x.shape[1]//(2**i)
        h = x.shape[0]//(2**i)
        print(h, w)
        if(w%2 == 1):
            # x = np.pad(x, [(0,0), (0,1)], mode='constant')
            w = w - 1

        HorizAvg = ( x[0:h,0:w:2] + x[0:h,1:w:2] ) / 2**0.5
        HorizDiff = ( x[0:h,0:w:2] - x[0:h,1:w:2] ) / 2**0.5
        Horiz = np.concatenate((HorizAvg,HorizDiff),axis=1)
        
        x[0:Horiz.shape[0], 0:Horiz.shape[1]] = Horiz
        
        if(h%2 == 1):
            # x = np.pad(x, [(0,1), (0,0)], mode='constant')
            h = h - 1

        VertAvg = ( x[0:h:2, 0:w] + x[1:h:2, 0:w] ) / 2**0.5
        VertDiff = ( x[0:h:2, 0:w] - x[1:h:2, 0:w] ) / 2**0.5
        Vert = np.concatenate((VertAvg,VertDiff),axis=0)
        
        x[0:Vert.shape[0], 0:Vert.shape[1]] = Vert

        return HDWT(x, t-1, i+1)

## Haar's Inverse DWT transformation function

def iHDWT(img, t, i):
    
    if(t == 0):
    
        return img

    else:
    
        x = np.array(img, dtype=float)
        w = x.shape[1]//(2**(t-1))
        h = x.shape[0]//(2**(t-1))
        print(h, w)
        
        if(h%2 == 1):
            # x = np.pad(x, [(0,1), (0,0)], mode='constant')
            h = h - 1
        # print(( x[0:h//2, 0:w] - x[h//2:h, 0:w] ) / 2**0.5, ( x[0:h//2, 0:w] + x[h//2:h, 0:w] ) / 2**0.5, x[1:h:2, 0:w])
        VertDiff = ( x[0:h//2, 0:w] - x[h//2:h, 0:w] ) / 2**0.5
        VertAvg = ( x[0:h//2, 0:w] + x[h//2:h, 0:w] ) / 2**0.5
        x[0:h:2, 0:w] = VertDiff
        x[1:h:2, 0:w] = VertAvg
        # print('--', x)
        # Vert = np.concatenate((VertAvg,VertDiff),axis=0)
        
        # x[0:Vert.shape[0], 0:Vert.shape[1]] = Vert
        
        if(w%2 == 1):
            # x = np.pad(x, [(0,0), (0,1)], mode='constant')
            w = w - 1

        HorizDiff = ( x[0:h,0:w//2] - x[0:h,w//2:w] ) / 2**0.5
        HorizAvg = np.abs( x[0:h,0:w//2] + x[0:h,w//2:w] ) / 2**0.5
        x[0:h, 0:w:2] = HorizDiff
        x[0:h, 1:w:2] = HorizAvg
        # Horiz = np.concatenate((HorizAvg,HorizDiff),axis=1)

        # x[0:Horiz.shape[0], 0:Horiz.shape[1]] = Horiz

        return iHDWT(x, t-1, i+1)

## Enhance DWT image using SVD decomposition

def enhance(img, t):
    
    x = np.array(img, dtype=float)
    w = x.shape[1]//(2**t)
    h = x.shape[0]//(2**t)
    print(h, w)
    
    u, s, vh = np.linalg.svd(img[0:h, 0:w], full_matrices=True)
    print(u.shape, s.shape, vh.shape)
    smat = np.zeros((u.shape[1], vh.shape[0]), dtype=complex)
    print(s)
    ## pump diagonal elements by 1.5 times
    s = 1.5*s
    print(s)
    smat[:s.shape[0], :s.shape[0]] = np.diag(s)
    img[0:h, 0:w] = np.dot(u, np.dot(smat, vh))

    return img

## 1st image

dwt = HDWT(img, 1, 0)
cv2.imshow('test', dwt/255)
cv2.imwrite('dwt_3.jpg', dwt)
cv2.waitKey(0)
cv2.destroyAllWindows()

enh_dwt = enhance(dwt, 1)

idwt = iHDWT(enh_dwt, 1, 0)
cv2.imshow('test', idwt/255)
cv2.imwrite('enhanced_3.jpg', idwt)
print(idwt) 
cv2.waitKey(0)
cv2.destroyAllWindows()


## 2nd image

img = cv2.imread('im_4.jpg')
# show original image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dwt = HDWT(img, 1, 0)
cv2.imshow('test', dwt/255)
cv2.imwrite('dwt_4.jpg', dwt)
cv2.waitKey(0)
cv2.destroyAllWindows()

enh_dwt = enhance(dwt, 1)

idwt = iHDWT(enh_dwt, 1, 0)
cv2.imshow('test', idwt/255)
cv2.imwrite('enhanced_4.jpg', idwt)
print(idwt) 
cv2.waitKey(0)
cv2.destroyAllWindows()