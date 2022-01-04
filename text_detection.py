import cv2
import numpy as np
import util
from imutils.object_detection import non_max_suppression

THRESHOLD = 0.1

def get_new_dimensions(multiple: int = 32):

    # Prepare image shape
    height, width, _ = image.shape
    print("Old dimensions",height, width)
    new_height = (height//multiple)*multiple    # EAST model requires image size multiples of 32
    new_width = (width//multiple)*multiple
    print("New dimensions",new_height, new_width)

    # Get radio change
    h_ratio = height/new_height
    w_ratio = width/new_width
    print("Ratio change", h_ratio, w_ratio)

    return new_height, new_width, h_ratio, w_ratio

def text_prediction(model: str = "east"):
    pass

def text_detection(image):
    # Load image
    #image = cv2.imread('preprocessed.jpg')

    # Load the frozen EAST model
    model = cv2.dnn.readNet('frozen_east_text_detection.pb')

    new_height, new_width, h_ratio, w_ratio = get_new_dimensions(32)

    # Create a 4D input blob
    blob = cv2.dnn.blobFromImage(image, 1, (new_width, new_height),(123.68, 116.78, 103.94), True, False)

    # Pass the blob to network
    model.setInput(blob)

    # Get output layers
    output_layers = model.getUnconnectedOutLayersNames()
    print(output_layers)

    # Text detection prediction
    print("Text detection...")
    (geometry, scores) = model.forward(output_layers)

    # Thresholding
    print("Thresholding...")
    rectangles = list()
    confidence_score = list()
    for i in range(geometry.shape[2]):
        for j in range(geometry.shape[3]):

            # If score for bounding box falls below threshold
            if scores[0][0][i][j] < THRESHOLD:
                continue

            bottom_x = int(j*4 + geometry[0][1][i][j])
            bottom_y = int(i*4 + geometry[0][2][i][j])


            top_x = int(j*4 - geometry[0][3][i][j])
            top_y = int(i*4 - geometry[0][0][i][j])

            rectangles.append((top_x, top_y, bottom_x, bottom_y))
            confidence_score.append(float(scores[0][0][i][j]))

    # use Non-max suppression to get the required rectangles
    fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)

    img_copy = image.copy()
    for (x1, y1, x2, y2) in fin_boxes:

        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)


        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Text Detection", img_copy)
    cv2.waitKey(0)

if __name__ == "__main__":

    # Input preprocessed image
    image = cv2.imread("preprocessed.png", cv2.IMREAD_COLOR)
    util.show_image(image)

    # Text detection
    text_detection(image)