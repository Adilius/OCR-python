import preprocessing
import text_detection
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

image = cv2.merge([image,image,image])

# Text detection
image, final_boxes = text_detection.detect_text(image, dimensions=True, threshold = 0.1, overlap_threshold = 0.5)

util.show_image(image)