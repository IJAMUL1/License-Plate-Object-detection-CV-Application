import torch
import cv2
import numpy as np

lower_hue_value = 0
upper_hue_value = 30
lower_saturation_value = 0
upper_saturation_value = 118
lower_value_value = 195
upper_value_value = 255

def draw_license_plate_boxes(frame, results, color_selector):
    expanded_roi = None  # Initialize expanded_roi here
    for _, det in enumerate(results.pred[0]):
        if det[-1].item() == 0:  # Assuming license plates are class 0
            box = det[:4].tolist()
            x1, y1, x2, y2 = map(int, box)
            # Crop the region of interest (ROI) from the original image
            roi = frame[y1:y2, x1:x2]
            new_width = 2 * (x2 - x1)  # Increase the width
            new_height = 2 * (y2 - y1)  # Increase the height
            # Resize the ROI to the desired dimensions
            expanded_roi = cv2.resize(roi, (new_width, new_height))
            hsv = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2HSV)
            # Define the lower and upper HSV ranges
            lower_range = np.array([0, 0, 195])
            upper_range = np.array([31, 118, 255])
            # Create a mask based on the defined range
            mask = cv2.inRange(hsv, lower_range, upper_range)
            # Calculate the percentage of red pixels in the image
            total_pixels = hsv.shape[0] * hsv.shape[1]
            red_pixels = np.count_nonzero(mask)
            red_percentage = (red_pixels / total_pixels) * 100
            # Set a threshold to determine if the image is predominantly red
            threshold = 5  # Adjust as needed
            is_red = red_percentage >= threshold
            if color_selector == "red":
                if is_red:
                    print("The image is predominantly red.")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a green rectangle around the license plate
                else:
                    print("we are in blue")
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a green rectangle around the license plate
            elif color_selector == "blue":
                if is_red:
                    print("The image is predominantly red.")
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a green rectangle around the license plate
                else:
                    print("we are in blue")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a green rectangle around the license plate
                
                # elif is_red and :
                #     print("The image is not predominantly red.")
                
    return frame, expanded_roi


# Path to your YOLOv5 model
model_path = r'C:\Users\ifeda\iio-2004-vision-midterm\best.pt'
   
    # Load your YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
vid_path = r'C:\Users\ifeda\Downloads\vid.mp4'
cap = cv2.VideoCapture(vid_path)
color_selector = input("Enter Red or Blue: ")

color_selector = color_selector.lower()

while True:
    ret, frame = cap.read()
    
    # Run inference on the processed frame using YOLOv5
    results = model(frame)  # Assuming YOLOv5 takes an image as input
    
    
    # Draw bounding boxes around license plates
    result, expanded_roi = draw_license_plate_boxes(frame, results, color_selector)
    cv2.imshow("Webcam1",result)
    
    # Check if expanded_roi is not None and has valid dimensions
    if expanded_roi is not None and expanded_roi.shape[0] > 0 and expanded_roi.shape[1] > 0:
        cv2.imshow("Webcam2", expanded_roi)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


    