import cv2
import numpy as np

def find_circle_properties(vicinity_image):
    # Convert the vicinity image to grayscale
    gray = cv2.cvtColor(vicinity_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    cv2.imwrite("2.jpg", blurred)

    rows = blurred.shape[0]

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, rows / 8,
 param1=100, param2=30, minRadius=1, maxRadius=3000)


    results = []

    # If circles are detected
    if circles is not None:
        # Convert the coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")

        # Iterate over each detected circle
        for circle in circles:
            x, y, radius = circle

            # Calculate square and circumference
            square = np.pi * (radius ** 2)
            circumference = 2 * np.pi * radius

            results.append((radius, square, circumference))

    return results

if __name__ == "__main__":
    # Load the vicinity image (you should get it from the previous script)
    vicinity_image = cv2.imread("1.jpg")

    # Find circle properties
    circle_properties = find_circle_properties(vicinity_image)

    # Print the results
    if circle_properties:
        for i, (radius, square, circumference) in enumerate(circle_properties):
            print(f"Circle {i + 1}:")
            print("Radius:", radius)
            print("Square:", square)
            print("Circumference:", circumference)
            print()
    else:
        print("No circles detected.")
