# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:38:04 2021

@author: Administrator
"""

import numpy as np
from imutils.object_detection import non_max_suppression
from joblib import dump, load
import cv2

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window    
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
t_image = cv2.imread('D:\\disease plant detection\\medicine_zone\\10_11_part2.tif')
orig = t_image.copy()
rects = []
winW = 128
winH = 128
for (x, y, window) in sliding_window(t_image, stepSize=64, windowSize=(winW,winH)):
    print(x,y)
    img = cv2.resize(window,(128,128),interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1,-1)
    clf = load('Random Forest.joblib') 
    pred = clf.predict(img)
    if pred[0] == 0:
        rects.append([x, y, x + winW, y + winH])
        cv2.rectangle(orig, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
        
rects = np.array(rects)
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(t_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
cv2.imwrite('RF Before NMS.jpg',orig)
cv2.imwrite('RF After NMS.jpg',t_image)