import cv2
import numpy as np
import ml_kit
import math

# Gaussian blur parameters
KERNEL_SIZE = (11, 11)
SIGMA_X = 100.0

# Unsharp weights
ORIG_WEIGHT = 4.0
GAUSSIAN_WEIGHT = -2.5


# Canny thresholds
THRESH_ONE = 100
THRESH_TWO = 100

# HoughLinesP parameters
RHO = 1
THETA = math.pi/180
LINE_THRESH = 50
MIN_LENGTH = 25.0
MAX_GAP = 15.0

class ImageProcessor(object):
    def __init__(self):
        pass

    @staticmethod
    def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])
        
        return img

    def overlay_images(self, image_bottom, image_top, pos):
        return ImageProcessor.overlay_image_alpha(image_bottom,
                        image_top[:, :, 0:3],
                        pos,
                        image_top[:, :, 3] / 255.0)
    
    def preprocess_image(self, img, cvtToGray=False):
        grayImage = img
        if cvtToGray:
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gaussian_blurr = cv2.GaussianBlur(grayImage, ksize = (7, 7), sigmaX = 0)
        morphological_transformation = cv2.morphologyEx(gaussian_blurr, cv2.MORPH_OPEN, kernel = np.ones((3, 3), np.uint8))
        thresholded_image = cv2.adaptiveThreshold(morphological_transformation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 19, C=2)

        '''
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresholded_image,kernel,iterations = 3)
        dilation = cv2.dilate(erosion,kernel,iterations = 2)
        '''

        return thresholded_image
    
    def draw_contour(self, img, contour):
        polygon = cv2.approxPolyDP(contour, epsilon = 0.0001*cv2.arcLength(contour, True), closed = True)
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(polygon)))

        color = (150,0,100)
        thickness = 1
        
        cv2.drawContours(img, [box], 0, color, thickness)

    def draw_all_contours(self, img, preprocess = False):
        if preprocess:
            img = self.preprocess_image(img, True)

        contours_image = np.zeros((len(img),len(img[0]),3), np.uint8)
        
        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours:
            self.draw_contour(contours_image, contour)
        
        return contours_image
    
    def sectional_grids(self, img, draw=False):
        sections = ml_kit.sectional_density(img)
        return sections
    
    def horizontal_flip(self, img):
        return cv2.flip(img, 1)

    def unsharp_mask(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian_img = cv2.GaussianBlur(gray_img, KERNEL_SIZE, SIGMA_X)
        sharp_img = cv2.addWeighted(gray_img, ORIG_WEIGHT, gaussian_img,\
            GAUSSIAN_WEIGHT, 0, gray_img)

        return sharp_img
    
    def Hough_lines(self, img):
        # Perform edge detection
        edges = cv2.Canny(img, THRESH_ONE, THRESH_TWO)
        # Get Hough lines
        lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH,\
            minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP)

        return lines
