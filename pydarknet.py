import cv2
import numpy as np
import sys
import libpydarknet

def load(cfg_file, weight_file):
    libpydarknet.load(cfg_file, weight_file)

def draw_dets(img, dets, threshold = 0.5, text = ''):
    if text != None and text != '':
        cv2.putText(img, text, (10, 30),0,0.8,(0,0,0), 2)

    for (name, thres, left, right, top, bot) in dets:
        if thres < threshold:
            continue
        p1 = (left, top)
        p2 = (right, bot)
        cv2.putText(img, name + ' %.2f' % thres,(left+5, bot+15),0,0.8,(0,0,255), 2)
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
    return img

def detect_image(img, threshold = 0.5):
    (h, w, c) = img.shape
    dets = libpydarknet.detect(img.data, w, h, c, threshold)
    return dets

def detect_file(file, threshold = 0.5):
    img = cv2.imread(file)
    if img == None:
        print('Cannot open image file')
    else:
        detect_image(img, threshold)

