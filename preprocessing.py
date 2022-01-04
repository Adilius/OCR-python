import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import util

def min_max(img):
    img_min = np.min(img)
    img_max = np.max(img)
    new_img = (img - img_min) / (img_max - img_min)
    return new_img


def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img):
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)


def noise_remove(img):
    # blur
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=33, sigmaY=33)

    # divide
    divide = cv2.divide(img, blur, scale=255)

    # otsu threshold
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


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
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return img


def preprocess_image(
    image, grayscale: bool, noise_removal: bool, deskew: bool, normalize: bool
):

    # Turn image to gray
    if grayscale:
        image = grayscale_image(image)

    # Remove noise gaussian
    if noise_removal:
        image = noise_remove(image)
        image = cv2.convertScaleAbs(image)

    # Rotate image
    if deskew:
        image = deskew_image(image)

    # Normalize pixel intensity value
    if normalize:
        image = min_max(image)

    return image


if __name__ == "__main__":

    # Input image
    image = cv2.imread("image.png", cv2.IMREAD_COLOR)
    util.show_image(image)

    # Preprocess
    image = preprocess_image(
        image=image, grayscale=True, noise_removal=True, deskew=True, normalize=True
    )
    util.show_image(image)
