import cv2
import numpy as np
import util
import csv

image = cv2.imread('preprocessed.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read input file containing bounding boxes coordinates
bounding_boxes_coordinates = list()
with open('text_detection_boxes.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        bounding_boxes_coordinates.append([int(n) for n in row])

for box in bounding_boxes_coordinates:
    image_segment = image[box[1]:box[3]+1, box[0]:box[2]+1]
    util.show_image_plt(image_segment)
