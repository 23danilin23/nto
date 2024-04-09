import cv2
import os

def get_file_names(folder_path):
    # Get all the file names in the specified folder
    files = os.listdir(folder_path)

    # Filter out only the file names (excluding directories)
    file_names = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]

    return file_names


def compress_image(input_image_path, output_image_path, target_resolution):

    if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

    image = cv2.imread(input_image_path)

    resized_image = cv2.resize(image, target_resolution)

    print(output_image_path)
    cv2.imwrite(output_image_path, resized_image)

    # print("Image compressed successfully.")

if __name__ == "__main__":
    input_images_path = input("specify folder path\n")
    output_images_path = input("specify folder where to exctract to\n")
    target_resolution = tuple(map(int, input("specify resolution\n").split()))

    for input_image_path in get_file_names(input_images_path):
        compress_image("./" + input_images_path + "/" + input_image_path, "./" + output_images_path +"/" + input_image_path, target_resolution)
