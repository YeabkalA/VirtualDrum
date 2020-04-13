from skimage.metrics import structural_similarity
import cv2

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