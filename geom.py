import cv2
import numpy as np

def rotate_clockwise(num_times, point, square_dim):
    if num_times == 0:
        return point
    point = (point[1], square_dim - point[0])
    
    return rotate_clockwise(num_times - 1, point, square_dim)

def rotate(rotation_type, point, square_dim):
    if rotation_type == cv2.ROTATE_90_CLOCKWISE:
        return rotate_clockwise(3, point, square_dim)
    elif rotation_type == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return rotate_clockwise(1, point, square_dim)
    elif rotation_type == cv2.ROTATE_180:
        return rotate_clockwise(2, point, square_dim)
    else:
        return None

# Flip codes:
# 0 - vertically
# > 0 - horizontally
# < 0 - vertically and horizontally
def flip(flip_type, point, square_dim):
    if flip_type == 0:
        return (point[0], square_dim - point[1])
    elif flip_type > 0:
        return (square_dim - point[0], point[1])
    else:
        return (square_dim - point[0], square_dim - point[1])

def draw_circle(img, center):
    img = img.copy()
    return cv2.circle(img, center, 5, (255,0,0), -1)

def test_rotation():
    square_dim = 300
    blank_image = np.zeros((square_dim,square_dim,3), np.uint8)
    point = (50, 200)

    original_image = draw_circle(blank_image.copy(), point)
    rccw_90 = rotate(cv2.ROTATE_90_COUNTERCLOCKWISE, point, square_dim)
    rcw_90 = rotate(cv2.ROTATE_90_CLOCKWISE, point, square_dim)
    r180 = rotate(cv2.ROTATE_180, point, square_dim)

    rccw_90 = draw_circle(blank_image.copy(), rccw_90)
    rcw_90 = draw_circle(blank_image.copy(), rcw_90)
    r180 = draw_circle(blank_image.copy(), r180)
    
    cv2.imshow('CW90', rcw_90)
    cv2.imshow('CCW90', rccw_90)
    cv2.imshow('180', r180)
    cv2.imshow('Orig', original_image)
    cv2.waitKey(0)

if __name__ == '__main__': 
    test_rotation()
    