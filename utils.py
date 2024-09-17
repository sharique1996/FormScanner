import cv2
import numpy as np
import pickle
from constants import CANNY_TH1,CANNY_TH2,IMAGE_HEIGHT,IMAGE_WIDTH,FORM_PATH

##########################################################
### RAW IMAGE PROCESSIG RELATED FUNCTIONS
##########################################################

def preprocess_image(image):
    """
    Convert image to grayscale and apply Gaussian blur.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    return blurred_image

def detect_edges(gray_image):
    """
    Perform edge detection using the Canny algorithm.
    """
    return cv2.Canny(gray_image, CANNY_TH1, CANNY_TH2)  # Adjust thresholds based on image characteristics

def find_largest_contour(canny_image):
    """
    Find and return the largest contour in the image.
    """
    contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def approximate_contour(contour):
    """
    Approximate the contour to a polygon with fewer vertices.
    """
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.02 * peri, True)

def reorder_points(points):
    """
    Reorder points to [top-left, top-right, bottom-right, bottom-left].
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    
    new_points[0] = points[np.argmin(add)]  # Top-left
    new_points[2] = points[np.argmax(add)]  # Bottom-right
    new_points[1] = points[np.argmin(diff)] # Top-right
    new_points[3] = points[np.argmax(diff)] # Bottom-left
    
    return new_points

def warp_perspective(image, points):
    """
    Apply a perspective transform to obtain a top-down view of the quadrilateral.
    """
    ordered_points = reorder_points(points)
    width1 = np.linalg.norm(ordered_points[0] - ordered_points[1])
    width2 = np.linalg.norm(ordered_points[2] - ordered_points[3])
    max_width = int(max(width1, width2))
    
    height1 = np.linalg.norm(ordered_points[0] - ordered_points[3])
    height2 = np.linalg.norm(ordered_points[1] - ordered_points[2])
    max_height = int(max(height1, height2))
    
    dst_points = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered_points, dst_points)
    
    return cv2.warpPerspective(image, M, (max_width, max_height))

def get_warped_image(frame):
    try:
        # preprocess the image
        gray_img = preprocess_image(frame)
        
        # Edge detection
        canny_img = detect_edges(gray_img)
        
        # Find the largest contour
        largest_contour = find_largest_contour(canny_img)
        
        # Approximate and check if it's a quadrilateral
        approx = approximate_contour(largest_contour)
        if len(approx) != 4:
            return cv2.imread(FORM_PATH)
        
        # Warp perspective and show the result
        warped_img = warp_perspective(frame, approx)

        # Resize image
        warped_img=cv2.resize(warped_img,(IMAGE_HEIGHT,IMAGE_WIDTH))
        return warped_img
    
    
    except Exception as e:
        print(f"An error occurred: {e}")


###############################################################
## CROPPED IMAGE PROCESSING RELATED FUNCTIONS
###############################################################


def load_pickle(file_path):
    """Load regions from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            regions = pickle.load(f)
        return regions
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file: {file_path}, Error: {e}")

def process_fields(form_image, regions , ocr):
    """Extract regions from an image based on provided coordinates and perform ocr."""
    image=get_warped_image(form_image)
    cropped_images = {}
    for label, coords in regions[:-1]:
        (x1, y1), (x2, y2) = coords
        # Ensure coordinates are within the image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        roi = image[y1:y2, x1:x2]

        # Extract text
        result = ocr.ocr(roi, cls=True)
        text=""

        if result[0]:
            for result in result[0]:
                text+=result[-1][0]+" "

        cropped_images[label]={"roi":roi,
                               "text":text}
    last_label, (box_top_left, box_bottom_right) = regions[-1]

    # Extract coordinates
    x1, y1 = box_top_left
    x2, y2 = box_bottom_right
    box=image[y1:y2, x1:x2]
    return image,cropped_images,box