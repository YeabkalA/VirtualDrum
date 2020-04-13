from skimage.metrics import structural_similarity
import cv2

from PIL import Image
from PIL import ImageChops

BASE_IMG_DIR = 'test_imgs/base_img.jpg'
NEW_IMG = 'test_imgs/curr_img.jpg'
DIFF_IMG_DIR = 'test_imgs/image_Chops.jpg'

def isColorWHite(col):
    return col[0] == 255

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
    
    def DiffThroughImageChops(self):
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

