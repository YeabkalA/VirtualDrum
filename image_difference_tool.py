from skimage.metrics import structural_similarity
import cv2

from PIL import Image
from PIL import ImageChops

import numpy as np

BASE_IMG_DIR = 'test_imgs/base_img.jpg'
NEW_IMG = 'test_imgs/curr_img.jpg'
DIFF_IMG_DIR = 'test_imgs/image_Chops.jpg'

def isColorWHite(col):
    return col[0] == 255

def get_average_color(img):
    bgr = np.asarray([0,0,0])
    for i in range(len(img)):
        for j in range(len(img[0])):
            col = img[i][j]
            bgr[0] += col[0]
            bgr[1] += col[1]
            bgr[2] += col[2]
    
    return bgr / (len(img) * len(img[0]))
    

class ImageDifferenceTool(object):
    def __init__(self):
        self.logs = []
    
    def StructuralSimilarityIndex(self, img1, img2, cvtToGray = False):
        if cvtToGray:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, diff = structural_similarity(img1, img2, full=True)
        cv2.imwrite('dif.jpg', diff)

        return score
    
    def DiffImageThroughImageChops(self):
        img1 = Image.open(BASE_IMG_DIR)
        img2 = Image.open(NEW_IMG)

        diff = ImageChops.difference(img1, img2)
        if diff.getbbox():
            diff.save(DIFF_IMG_DIR)
    
    # Expects a non stimulated image to be black.
    # Stimulated areas are made white.
    def ColorDiff(self, img):
        white_pixel_count = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if isColorWHite(img[i][j]):
                    white_pixel_count += 1
        
        img_area = len(img) * len(img[0])
        white_pixel_percentage = 100.0 * white_pixel_count/img_area

        return white_pixel_percentage > 3.0
    
    def ColorDiffGrayscale(self,img1, img2):
        img1 = cv2.resize(img1, (32,32))
        img2 = cv2.resize(img2, (32,32))
        
        diff_img = np.asarray(img1) - np.asarray(img2)
        # for i in range(len(diff_img)):
        #     for j in range(len(diff_img[0])):
        #         if diff_img[i][j] < 0:
        #             diff_img[i][j] = -diff_img[i][j]
        #         if diff_img[i][j] < 180:
        #             diff_img[i][j] = 0
        
        diff_img = cv2.medianBlur(diff_img,5)
        
        cv2.imshow('Diff image', diff_img)
        cv2.waitKey(1)

        diff_flatten = diff_img.flatten()
        blacks = diff_flatten[diff_flatten<150]
        perc = len(blacks)/len(diff_flatten)
        
        return perc
    
    def GetWhitePercentage(self,img):
        img = cv2.resize(img, (8,8))
        img = img.flatten()
        whites = img[img==255]
        perc = len(whites)/len(img)
        

        return perc

    
    # img1 and img2 are color images.
    def ColorDiffVector(self, img1, img2):
        
        av_col1, av_col2 = get_average_color(img1), get_average_color(img2)
        
        return np.linalg.norm(av_col1 - av_col2)
    
    def ColorDiffRob(self, base, img):
        base = np.asarray(base)
        img = np.asarray(img)
        diff = img - base
        diff = np.abs(diff)

        # cv2.imshow('abs', diff)
        # cv2.waitKey(0)
        diff[diff<100] = 0
        diff[diff>100] = 255

        img = img & diff
        
        return img, diff

def testColorDiffGrayscale():
    tool = ImageDifferenceTool()
    img1, img2 = [[1,2], [3,-1]], [[0,1], [1,1]]
    diff = tool.ColorDiffGrayscale(img1, img2)
    print(diff)

def testMaskDiff():
    tool = ImageDifferenceTool()
    stick = cv2.imread('mask_images/stick2.jpg', cv2.IMREAD_GRAYSCALE)
    base = cv2.imread('mask_images/bg.jpg', cv2.IMREAD_GRAYSCALE)
    img, mask = tool.ColorDiffRob(base, stick)
    mask = cv2.medianBlur(mask,5)
    cv2.imshow('Diff', img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

if __name__ == '__main__':
    #testColorDiffGrayscale()
    testMaskDiff()