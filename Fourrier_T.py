import cmath
import numpy as np
from math import log, ceil
import pylab as plt
import cv2

import numpy as np

## 1D DFT function

def DFT_1D(fx):
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    fu = fx.copy()

    for i in range(M):
        u = i
        sum = 0
        for j in range(M):
            x = j
            tmp = fx[x]*np.exp(-2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        # print(sum)
        fu[u] = sum
    # print(fu)

    return fu

## 1D inverse DFT function

def inverseDFT_1D(fu):
    fu = np.asarray(fu, dtype=complex)
    M = fu.shape[0]
    fx = np.zeros(M, dtype=complex)

    for i in range(M):
        x = i
        sum = 0
        for j in range(M):
            u = j
            tmp = fu[u]*np.exp(2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        fx[x] = np.divide(sum, M, dtype=complex)

    return fx

## 2D Inverse DFT function

def inverseDFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseDFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseDFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseDFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

## 2D DFT function

def DFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)
    print(fu.shape)
    if len(fu.shape) == 2:
        for i in range(h):
            print(i)
            fx[i, :] = DFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = DFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = DFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

## Euclidean distance function between two points

def distance(point1,point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

## Ideal Low pass filtering function

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

## Ideal High pass filtering function

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base



img = cv2.imread('im_1.jpg')
# show original image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('test', img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

dft = DFT_2D(img)

dft_shift = np.fft.fftshift(dft)

print(dft)

cv2.imshow('dft', np.log(1+np.abs(dft_shift)))
 
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('dft.jpg', np.log(1+np.abs(dft_shift)))

dft_image_filtered = dft_shift * idealFilterHP(50,dft_shift.shape) 

dft_hp_img = np.fft.ifftshift(dft_image_filtered)   

img_hp = inverseDFT_2D(dft_hp_img)

cv2.imshow('img_hp', img_hp)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('highpass.jpg', img_hp)

dft_image_filtered = dft_shift * idealFilterLP(50,dft_shift.shape) 

dft_lp_img = np.fft.ifftshift(dft_image_filtered)   

img_lp = inverseDFT_2D(dft_lp_img)

cv2.imshow('img_lp', img_lp)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('lowpass.jpg', img_lp)




## Bonus question

## 1D FFT function

def FFT_1D(fx):
    
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    minDivideSize = 4

    if M % 2 != 0:
        print("the input size must be 2^n")
        return

    if M <= minDivideSize:
        return DFT_1D(fx)
    else:
        fx_even = FFT_1D(fx[::2])  # compute the even part
        fx_odd = FFT_1D(fx[1::2])  # compute the odd part
        W_ux_2k = np.exp(-2j * np.pi * np.arange(M) / M)

        f_u = fx_even + fx_odd * W_ux_2k[:M//2]

        f_u_plus_k = fx_even + fx_odd * W_ux_2k[M//2:]

        fu = np.concatenate([f_u, f_u_plus_k])

    return fu

## 1D Inverse FFT function

def inverseFFT_1D(fu):
    
    fu = np.asarray(fu, dtype=complex)
    fu_conjugate = np.conjugate(fu)

    fx = FFT_1D(fu_conjugate)

    fx = np.conjugate(fx)
    fx = fx / fu.shape[0]

    return fx

## 2D FFT function

def FFT_2D(fx):
    
    h, w = fx.shape[0], fx.shape[1]

    fu = np.zeros(fx.shape, dtype=complex)

    if len(fx.shape) == 2:
        for i in range(h):
            fu[i, :] = FFT_1D(fx[i, :])

        for i in range(w):
            fu[:, i] = FFT_1D(fu[:, i])
    elif len(fx.shape) == 3:
        for ch in range(3):
            fu[:, :, ch] = FFT_2D(fx[:, :, ch])
    return fu

## 2D Inverse FFT function

def inverseFFT_2D(fu):

    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseFFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseFFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseFFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx