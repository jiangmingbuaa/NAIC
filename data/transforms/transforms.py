# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random

'''
class RandomErasing(object):
    # Cutout(probability = 0.5, size=64, mean=[0.0, 0.0, 0.0]) img size: 384*128
    def __init__(self, probability = 0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = 0.5
        self.mean = (0,0,0)
        self.size = 64
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        h = self.size
        w = self.size
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        return img
'''
import numpy as np
import cv2
from PIL import Image

class HisEqulColor(object):
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, img):
        # print(type(img))
        if random.uniform(0, 1) >= self.probability:
            return img
        
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0],channels[0])
        cv2.merge(channels,ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img

# class HisEqulColor(object):

#     def histeq(self, imarr):
#         hist, bins = np.histogram(imarr, 255)
#         cdf = np.cumsum(hist)
#         cdf = 255 * (cdf/cdf[-1])
#         res = np.interp(imarr.flatten(), bins[:-1], cdf)
#         res = res.reshape(imarr.shape)
#         return res, hist
    
#     def __call__(self, img):
#         imarr = np.array(img)
#         r_arr = imarr[...,0]
#         g_arr = imarr[...,1]
#         b_arr = imarr[...,2]
        
#         r_res, _ = self.histeq(r_arr)
#         g_res, _ = self.histeq(g_arr)
#         b_res, _ = self.histeq(b_arr)
        
#         new_imarr = np.zeros(imarr.shape, dtype='uint8')
#         new_imarr[...,0] = r_res
#         new_imarr[...,1] = g_res
#         new_imarr[...,2] = b_res
        
#         return Image.fromarray(new_imarr, mode='RGB')

# class HisEqulColor(object):

#     def __init__(self, probability=0.5):
#         self.probability = probability

#     def histeq(self, imarr):
#         hist, bins = np.histogram(imarr, 255)
#         cdf = np.cumsum(hist)
#         cdf = 255 * (cdf/cdf[-1])
#         res = np.interp(imarr.flatten(), bins[:-1], cdf)
#         res = res.reshape(imarr.shape)
#         return res, hist
    
#     def __call__(self, img):
#         if random.uniform(0, 1) >= self.probability:
#             return img

#         imarr = np.array(img)
#         r_arr = imarr[...,0]
#         g_arr = imarr[...,1]
#         b_arr = imarr[...,2]
        
#         imarr2 = np.average(imarr, axis=2)
#         hist, bins = np.histogram(imarr2, 255)
#         cdf = np.cumsum(hist)
#         cdf = 255 * (cdf/cdf[-1])
        
#         r_res = np.interp(r_arr, bins[:-1], cdf)
#         g_res = np.interp(g_arr, bins[:-1], cdf)
#         b_res = np.interp(b_arr, bins[:-1], cdf)
        
#         new_imarr = np.zeros(imarr.shape, dtype="uint8")
#         new_imarr[...,0] = r_res
#         new_imarr[...,1] = g_res
#         new_imarr[...,2] = b_res
        
#         return Image.fromarray(new_imarr, mode='RGB')

class RandomShuffle(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        
        index = np.arange(3)
        np.random.shuffle(index)
        img = img[index, :, :]
        return img



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        # print(mean)
        mean=(0,0,0)
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img