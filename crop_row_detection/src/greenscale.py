import cv2
import numpy as np
from os import path
import os
import time
import math
from matplotlib import pyplot as plt

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value

def skeletonize(image_in):
    '''Inputs and grayscale image and outputs a binary skeleton image'''
    size = np.size(image_in)
    skel = np.zeros(image_in.shape, np.uint8)

    ret, image_edit = cv2.threshold(image_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    # while not done:
    eroded = cv2.erode(image_edit, element, iterations = 2)
        # temp = cv2.dilate(eroded, element)
        # temp = cv2.subtract(image_edit, temp)
        # skel = cv2.bitwise_or(skel, temp)
        # image_edit = eroded.copy()

        # zeros = size - cv2.countNonZero(image_edit)
        # if zeros == size:
        #     done = True
    sobelx = cv2.Sobel(eroded,cv2.CV_8U,1,0,ksize=5)
    return sobelx

def main():
    #read image
    src = cv2.imread('../crop_row_detection/CRBD/0-CRBD-bag/bag (1).jpg',  cv2.IMREAD_UNCHANGED)
    print(src.shape)

    ## convert to hsv
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (36, 255,255))86
    mask = cv2.inRange(hsv, (30, 70, 0), (50, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(src, np.uint8)
    green[imask] = src[imask]
    # print(green.shape)

    # extract green channel -> grayscale
    green_channel = green[:,:,1]
    # print(green_channel.shape)
    height, width  = green_channel.shape
    window_name = 'image'
    # Using cv2.imshow() method
    # Displaying the image
   

    # result = np.zeros(my_image.shape, my_image.dtype)
    for j in range(0, int(height/2)):
        for i in range(0, width):
            sum_value = 0 * green_channel[j, i]
            green_channel[j, i] = saturated(sum_value)
    print(green_channel.shape)
    
    w_list = list(range(0,10))
    w_list.extend(range(width-10,width))
    for j in range(int(height/2), height):
        for i in w_list:
            sum_value = 0 * green_channel[j, i]
            green_channel[j, i] = saturated(sum_value)

    ret, image_edit = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8) #3,6 - 2,3,2,4
    dilation = cv2.dilate(image_edit,kernel,iterations = 2)
    cv2.imshow(window_name, dilation)
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    cv2.imshow('erosion', erosion)
    # dilation2 = cv2.dilate(erosion,kernel,iterations = 1)
    # erosion2 = cv2.erode(erosion,kernel,iterations = 1)
    # opening = cv2.morphologyEx(green_channel, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('open', erosion2)
    
    

    #laplacian = cv2.Laplacian(erosion,cv2.CV_8U)
    image_edit = skeletonize(erosion)
    cv2.imshow('erosion_skel', image_edit)
    # image_edit2 = skeletonize(erosion2)
    # cv2.imshow('open_skel', image_edit2)

    # concatenate image Horizontally
    # Hori = np.concatenate((img1, img2), axis=1)
    
    # concatenate image Vertically
    # Verti = np.concatenate((img1, img2), axis=0)
    
    # cv2.imshow('HORIZONTAL', Hori)
    # cv2.imshow('VERTICAL', Verti)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # im_path = '../crop_row_detection/edits/bag_bsobel.png'
    #save image
    # check_file(im_path)
    # cv2.imwrite(im_path,sobelx) 



""" def check_file(filePath):
    if path.exists(filePath):
        numb = 1
        while True:
            newPath = "{0}_{2}{1}".format(*path.splitext(filePath) + (numb,))
            if path.exists(newPath):
                numb += 1
                cv2.imwrite(newPath,green_channel) 
            else:
                return newPath
    return filePath """

main()