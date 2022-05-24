# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:55:36 2021

@author: honyu
"""

import numpy as np
import cv2
import csv
import pandas as pd
import pprint
import pathlib
import os

x=0
l=1
p = open(r'F:\000001.csv', 'w', encoding='utf-8') 
p.close()
 
for num in range(49):
 x=x+1
 l=l+1
 print("flow") 
 print(x)
 cap = cv2.VideoCapture(r"D:\data2\{0:010d}.png".format(x))
 cap2 = cv2.VideoCapture(r"D:\data2\{0:010d}.png".format(l))
# params for ShiTomasi corner detection
 feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
 lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
 color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
 ret, old_frame = cap.read()
 old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
 p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
 mask = np.zeros_like(old_frame)

#while(1):
 ret,frame = cap2.read()
 frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
 p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
 good_new = p1[st==1]
 good_old = p0[st==1]
 countx = (len(err))
 county = 100 - countx
 data = [['null']]
 
 with  open(r'F:\000001.csv','a') as g:
  writer = csv.writer(g)
  
  writer.writerows(err)
  """
  for com in range(county):
       writer.writerows(data)
       """
    # draw the tracks
 for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
 img = cv2.add(frame,mask)

 cv2.imshow('frame',img)
 k = cv2.waitKey(30) & 0xff


    # Now update the previous frame and previous points
 old_gray = frame_gray.copy()
 p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()