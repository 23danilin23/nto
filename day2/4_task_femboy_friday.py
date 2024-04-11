import cv2
import numpy as np
import pandas as pd
import os

def find_circle_properties(vicinity_image):
    # Convert the vicinity image to grayscale
    gray = cv2.cvtColor(vicinity_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.blur(gray, (5, 5))

    rows = blurred.shape[0]

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp = 1.2, minDist = rows / 16,
 param1=100, param2=30,
 minRadius=1, maxRadius=3000)


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

            results.append((square, radius, circumference))
        return results
    return None


if __name__ == "__main__":
    # Create Pandas dataset
    df = pd.DataFrame(columns=["Viscinity file name", "Radius", "Square", "Circumference", "Octane number"])
    for dir in os.listdir("./output_data"):

        images_path = "./output_data/" + dir
        octane_number = dir

        for i, filename in enumerate(os.listdir(images_path)):
            # Load the vicinity image
            vicinity_image = cv2.imread(os.path.join(images_path, filename))

            # Find circle properties
            circle_properties = find_circle_properties(vicinity_image)

            # If circles are detected
            if circle_properties:
                
                data = pd.DataFrame({
                "Viscinity file name": filename,
                "Radius": radius,
                    "Square": square,
                    "Circumference": circumference,
                    "Octane number": octane_number  # Add your octane number here
                } for (radius, square, circumference) in circle_properties)
                print(data)
                # Append the properties to the DataFrame
                df = pd.concat([df, data], ignore_index=True)
            else:
                print(f"No circles detected in image {filename}.")

        

        # Save the DataFrame to a CSV file
        df.to_csv("output_data.csv", index=False)

        print("Circle properties saved to output_data.csv.")
