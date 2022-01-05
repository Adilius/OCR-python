import cv2

def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)

def write_image(image_name, image):
    cv2.imwrite(image_name, image)