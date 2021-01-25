# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:22:39 2021

@author: Administrator
"""

import cv2
import os
import numpy as np
import scipy.io

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
        if img is not None:
            images.append(img)
    images =  np.array(images)
    images = images.reshape(images.shape[0],-1)
    return images

train_AI = load_images_from_folder('./train/AI')
train_PI = load_images_from_folder('./train/PI')

val_AI = load_images_from_folder('./val/AI')
val_PI = load_images_from_folder('./val/PI')

train_AI_X = np.concatenate([train_AI,val_AI])
train_PI_X = np.concatenate([train_PI,val_PI])

train_AI_y = np.ones([len(train_AI_X),1])
train_PI_y = np.zeros([len(train_PI_X),1])

train_AI = np.hstack([train_AI_X,train_AI_y])
train_PI = np.hstack([train_PI_X,train_PI_y])

train_data = np.concatenate([train_AI,train_PI])


test_AI_X = load_images_from_folder('./test/AI')
test_PI_X = load_images_from_folder('./test/PI')

test_AI_y = np.ones([len(test_AI_X),1])
test_PI_y = np.zeros([len(test_PI_X),1])

test_AI = np.hstack([test_AI_X,test_AI_y])
test_PI = np.hstack([test_PI_X,test_PI_y])
test_data = np.concatenate([test_AI,test_PI])

#scipy.io.savemat('train_data.mat',{'train_data':train_data})
#scipy.io.savemat('test_data.mat',{'test_data':test_data})

#test_images = load_images_from_folder('test')
#val_images = load_images_from_folder('val')