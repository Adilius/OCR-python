import preprocessing
import util
import cv2

# Input image
image = cv2.imread("image.png", cv2.IMREAD_COLOR)
util.show_image(image)
# Preprocess
image = preprocessing.preprocess_image(
    image=image, grayscale=True, noise_removal=True, deskew=True, normalize=True
)
util.show_image(image)
cv2.imwrite('preprocessed.png', 255*image)