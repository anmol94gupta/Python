#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 00:24:09 2021

@author: anmol
"""

# IMPORTING PACKAGES
import imutils
import cv2

#---------------------
# LOADING IMAGES
# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("1.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution

#---------------------
# PRINTING IMAGES
# v2.imshow("Image", image)
# cv2.waitKey(0)

#---------------------
#ACCESSING RGB VALUES

# access the RGB pixel located at x=50, y=100, keepind in mind that
# OpenCV stores images in BGR order rather than RGB
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

#---------------------
# REGION OF INTERESTS

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
# roi = image[60:160, 320:420]
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)

#---------------------
# RESIZING - ignoring aspect ratio
# resize the image to 200x200px, ignoring aspect ratio
# resized = cv2.resize(image, (100, 200))
# cv2.imshow("Fixed Resizing", resized)
# cv2.waitKey(0)

#---------------------
# RESIZING - with aspect ratio
# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
# r = 300.0 / w
# dim = (300, int(h * r))
# resized = cv2.resize(image, dim)
# cv2.imshow("Aspect Ratio Resize", resized)
# cv2.waitKey(0)
# or can us the following =>
# manually computing the aspect ratio can be a pain so let's use the
# imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

#---------------------
# ROTATION
# rotation can also be easily accomplished via imutils with less code
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)