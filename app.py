import cv2
import numpy as np
from skimage import io


def show_image(image):
    io.imshow(image)
    io.show()

def min_max(img):
    img_min = np.min(img)
    img_max = np.max(img)
    new_img = (img-img_min)/(img_max-img_min)
    return new_img

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_image(img):
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

def noise_remove(img):
    return cv2.medianBlur(img.astype(np.float32), 3)

def deskew_image(img):
    # detect box
    img = cv2.convertScaleAbs(img)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # find angle
    if angle < -45:
	    angle = -(90 + angle)
    else:
        angle = -angle

    # deskew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print(img.shape)
    return img


# Input image
img = cv2.imread("image.png", cv2.IMREAD_COLOR)
show_image(img)

# Turn image to gray
img = grayscale_image(img)
show_image(img)

# Remove noise
img = noise_remove(img)
img = cv2.convertScaleAbs(img)
show_image(img)

# Rotate image
img = deskew_image(img)
show_image(img)

# Normalize pixel intensity value
img = min_max(img)
show_image(img)