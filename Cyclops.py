import sys
import os
import dlib
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.misc
from skimage.restoration import inpaint
import cv2 as cv
from scipy.spatial import ConvexHull
import scipy.misc
from Poisson import *

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    return XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def getMask(hull, Y, shape):
    J, I = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    J = J.flatten()
    I = I.flatten()
    inside = np.ones(J.size, dtype = np.uint8)
    for a, b, c in hull.equations:
        res = a*J + b*I + c
        inside *= (res < 0)
    return np.reshape(inside, (shape[0], shape[1]))

def getPtsEye(X, eyenum=0):
    eyeidx = np.concatenate((np.arange(17, 22), np.arange(36, 42)))
    if eyenum > 0:
        eyeidx = np.concatenate((np.arange(22, 27), np.arange(42, 48)))
    idxs1 = [37, 38, 40, 41]
    if eyenum == 1:
        idxs1 = [43, 44, 46, 47]
    mu = np.mean(X[idxs1, :], 0)
    dR = X[39, :] - X[36, :]
    if eyenum > 0:
        dR = X[45, :] - X[42, :]
    dR = dR/np.sqrt(np.sum(dR**2))
    Y = X[eyeidx, :]
    idxs2 = range(17, 22)
    if eyenum > 0:
        idxs2 = range(22, 27)
    for idx in idxs2:
        dy = X[idx, :] - mu
        dyproj = dy - dR*np.sum(dy*dR)
        y = X[idx, :] - 2*dyproj
        Y = np.concatenate((Y, y[None, :]), 0)
    return Y

def getEyeMask(X, eye, offx = 0, offy = 0, expandfac = 1.05):
    X2 = np.array(X)
    X2[:, 0] += offx
    X2[:, 1] += offy
    Y = getPtsEye(X2, eye)
    mu = np.mean(Y, 0)
    Y = Y - mu
    Y *= expandfac
    Y = Y + mu
    hull = ConvexHull(Y)
    return getMask(hull, Y, imgbox.shape)

def getBothEyeMask(X, expandfac = 1.05):
    Y = getPtsEye(X, 0)
    Y = np.concatenate((Y, getPtsEye(X, 1)))
    mu = np.mean(Y, 0)
    Y = Y - mu
    Y *= expandfac
    Y = Y + mu
    hull = ConvexHull(Y)
    return getMask(hull, Y, imgbox.shape)

def fillmask(img, mask):
    res = np.zeros_like(img)
    for c in range(3):
        resc = img[:, :, c]
        resc *= (1-mask)
        res[:, :, c] = resc
    return res

keypts = np.concatenate((np.arange(17), np.arange(30, 36), np.arange(48, 68)))
eyebottomidx = 27

predictor_path = "shape_predictor_68_face_landmarks.dat"
filename = "faces/2009_004587.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = dlib.load_rgb_image(filename)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
fac = 4
for k, d in enumerate(dets[0:1]):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    X = shape_to_np(shape)    
    bounds = [min(d.left(), np.min(X[:, 0])), max(d.right(), np.max(X[:, 0])) \
              , min(d.top(), np.min(X[:, 1])), max(d.bottom(), np.max(X[:, 1])) ]
    X[:, 0] -= bounds[0]
    X[:, 1] -= bounds[2]

    imgbox = img[bounds[2]:bounds[3], bounds[0]:bounds[1], :]
    imgbox = scipy.misc.imresize(imgbox, (imgbox.shape[0]*fac, imgbox.shape[1]*fac))
    X = np.array(X*fac, dtype=np.float32)

    plt.subplot(131)
    plt.imshow(imgbox)
    mask = getBothEyeMask(X)
    #mask = getEyeMask(X, 0) + getEyeMask(X, 1)

    plt.subplot(132)
    res = np.array(imgbox)
    res = fillmask(res, mask)
    plt.imshow(res)
    
    plt.subplot(133)
    #res = inpaint.inpaint_biharmonic(imgbox, mask, multichannel=True)
    res = cv.inpaint(imgbox, mask, 3, cv.INPAINT_TELEA)

    # Move left eye to top of nose
    offset = 1.0*(X[27, :] - np.mean(X[[37, 38, 40, 41], :], 0))
    X = np.array(np.round(X), dtype = np.int32)
    offset = np.array(np.round(offset), dtype = np.int32)
    masksource = getEyeMask(X, 0)
    maskdest = getEyeMask(X, 0, offset[0], offset[1])

    solve_poisson(imgbox, res, masksource, maskdest)

    """
    eyebefore = eyebefore.flatten()
    eyeafter = getEyeMask(X, 0, offset[0], offset[1]).flatten()
    for c in range(3):
        resc = res[:, :, c]
        resc = resc.flatten()
        resb = imgbox[:, :, c].flatten()
        resc[eyeafter == 1] = resb[eyebefore == 1]
        res[:, :, c] = np.reshape(resc, (res.shape[0], res.shape[1]))
    #res = fillmask(res, eye0)
    #scipy.misc.imsave("mask.png", eyebefore)

    #res = blend(res, res, mask, eyebefore)
    plt.imshow(mask)
    """
    plt.imshow(res)

    plt.show()