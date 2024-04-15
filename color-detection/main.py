import numpy as np
import cv2


def get_limits(color):
    #HSV Values not RGB
    colors = {
        'blue': ([88, 100, 100], [140, 255, 255]),  # Lower and upper limits for blue
        'yellow': ([20, 100, 100], [30, 255, 255]),  # Lower and upper limits for yellow
        'green': ([40, 100, 100], [80, 255, 255]),  # Lower and upper limits for green

    }
    color = color.lower()
    if color in colors:
        lower_limit, upper_limit = colors[color]
        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)
        return lower_limit, upper_limit
    else:
        return None, None


def adjust_brightness(img, factor=1.2):
    # Convert to HSV to adjust brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detect_color(frame, color_list):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = []

    for color_name in color_list:
        lower_limit, upper_limit = get_limits(color_name)
        if lower_limit is not None and upper_limit is not None:
            mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

            # Apply adaptive thresholding
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to reduce noise
            kernel = np.ones((9, 9), np.uint8)  # Larger kernel for noise reduction
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours and filter based on area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Adjust area threshold as needed
                    x, y, w, h = cv2.boundingRect(cnt)
                    detected.append((color_name, (x, y, x + w, y + h)))  # Store bounding box coordinates

    return detected


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if ret:
            # Mirror the frame horizontally
            mirrored_frame = cv2.flip(frame, 1)  # 1 for horizontal flip
            # Brighten the frame (adjust the factor as needed)
            brightened_frame = adjust_brightness(mirrored_frame,1.5)

            colors_to_detect = ['blue', 'yellow', 'green']
            detected_colors = detect_color(brightened_frame, colors_to_detect)

            for color_name, bbox in detected_colors:
                x1, y1, x2, y2 = bbox
                brightened_frame = cv2.rectangle(brightened_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(brightened_frame, color_name.capitalize(), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2, cv2.LINE_AA)

            # Display the instruction to press ESC key to close in black color
            cv2.putText(brightened_frame, "Press ESC key to close", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('frame', brightened_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Check if the key pressed is the Escape key (ASCII 27)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



