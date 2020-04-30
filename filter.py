# import the necessary packages
import numpy as np
import cv2
import image_processor

ip = image_processor.ImageProcessor()

def filter_black(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # for i in range(len(img)):
        #         for j in range(len(img[0])):
        #                 if img[i][j] < 50:
        #                         img[i][j] = 0
        #                 else:
        #                         img[i][j] = 255

        # _,thresh = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
        
        # return thresh
        return img

# define the list of boundaries
# Getting ranges = https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
def filter(frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #low_blue = np.array([94, 80, 2])
        # define range of blue color in HSV
        lower_blue = np.array([160,100,100])
        upper_blue = np.array([185,255,255])
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        #blue = ip.draw_all_contours(blue, True)

        #cv2.imshow("blue", blue)

        return blue

        #     boundaries = [
#             #([17, 15, 100], [50, 56, 200]),
#             #([86, 31, 4], [220, 88, 50]),
#             ([25, 146, 190], [62, 174, 250]),
#             ([103, 86, 65], [145, 133, 128])
#     ]
#     count = 0
#     # loop over the boundaries
#     for (lower, upper) in boundaries:
#             # create NumPy arrays from the boundaries
#             lower = np.array(lower, dtype = "uint8")
#             upper = np.array(upper, dtype = "uint8")
#             # find the colors within the specified boundaries and apply
#             # the mask
#             mask = cv2.inRange(image, lower, upper)
#             output = cv2.bitwise_and(image, image, mask = mask)
#             # show the images
#             cv2.imshow("orig", image)
#             name = f"images{count}"
#             names.append(name)
#             cv2.imshow(name, output)
#             count += 1

img_process = image_processor.ImageProcessor()

names = []
def main():
        cap = cv2.VideoCapture(0)

        while(True):
                #compute_time_start = time.time()
                _, frame = cap.read()
                frame = img_process.horizontal_flip(frame)

                filter(frame)


                #print(f'Compute time = {time.time() - compute_time_start}')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
        main()