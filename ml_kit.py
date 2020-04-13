import cv2
import consts
import time

def sectional_density(image, draw=False, sparsity=1, w=20, h=20):
    steps = 0
    #image_size = image.shape[0]
    image_size = 240
    CELL_WIDTH, CELL_HEIGHT = w, h
    pixel_percentages = [0 for i in range((image_size // CELL_WIDTH) * (image_size // CELL_HEIGHT))]
    total_black_pixels, count = 0, 0

    for corner_y in range(0, (image_size - CELL_HEIGHT + 1), CELL_HEIGHT):
        for corner_x in range(0, (image_size - CELL_WIDTH + 1), CELL_WIDTH):
            if draw:
                cv2.rectangle(image, (corner_x, corner_y), (corner_x+CELL_WIDTH, corner_y+CELL_HEIGHT), \
                     consts.RED, consts.AREA_BOUNDARY_THICKNESS) 
            for i in range(0, CELL_HEIGHT, sparsity):
                for j in range(0, CELL_WIDTH, sparsity):
                    steps += 1
                    if image[corner_y + i][corner_x + j] > 0: # Pixel is black.
                        pixel_percentages[count] += 1
                        total_black_pixels += 1
            count += 1
    # Convert to percentages.
    if total_black_pixels == 0: return [1]*49 # No image here, don't crash


    return pixel_percentages, steps

img1 = [[i*j + 2 for j in range(240)]for i in range(240)]
img2 = [[i**j + 20 for j in range(240)]for i in range(240)]

sd, steps = sectional_density(img1, sparsity=5)
print(f'Len = {len(sd)} sum = {sum(sd)}, steps = {steps}')