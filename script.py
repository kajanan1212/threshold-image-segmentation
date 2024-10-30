import cv2
import numpy as np


def load_image(file_name: str):
    image = cv2.imread(file_name)
    return image


def write_image(file_name: str, image):
    cv2.imwrite(file_name, np.clip(image, 0, 255).astype(np.uint8))


def rgb_to_gray_scale(image):
    RGB = 0.21, 0.72, 0.07
    weights = np.array(RGB).reshape(3, 1)

    gray_scale_image = np.dot(image, weights)

    return np.clip(gray_scale_image, 0, 255).reshape(image.shape[:2]).astype(np.uint8)


def find_threshold(image, tolerance, max_iterations):
    temp_threshold = image.mean()

    for i in range(max_iterations):
        r_1 = image[np.where(image >= temp_threshold)]
        r_2 = image[np.where(image < temp_threshold)]

        mean_1 = r_1.mean()
        mean_2 = r_2.mean()

        threshold = (mean_1 + mean_2) / 2

        threshold_difference = abs(temp_threshold - threshold)

        if threshold_difference < tolerance:
            print(f"Convergence achieved with a tolerance of {tolerance} | THRESHOLD DIFFERENCE: {threshold_difference} | ITERATION: {i}")
            return threshold

        temp_threshold = threshold
    else:
        print(f"Maximum iterations ({max_iterations}) have been reached | LAST THRESHOLD DIFFERENCE: {threshold_difference}")

    return temp_threshold


def apply_threshold_segmentation(image, threshold):
    segmented_image = np.zeros(image.shape, dtype=np.uint8)

    segmented_image[np.where(image >= threshold)] = np.uint8(255)

    return segmented_image


if __name__ == "__main__":
    image_num = input("enter the image number: ")

    # 1.Load an image from the disk
    image = load_image(file_name=f"images/{image_num}/01-image.jpg")
    print(f"original image's shape: {image.shape}")

    # 2.Convert the image to gray-scale (8bpp format)
    gray_scale = rgb_to_gray_scale(image=image)
    print(f"gray scale image's shape: {gray_scale.shape}")
    write_image(f"images/{image_num}/02-gray-scale.jpg", gray_scale)

    # 3.Find threshold using inter-means algorithm
    threshold = find_threshold(
        image=gray_scale, tolerance=1e-4, max_iterations=1000
    )
    print(f"threshold: {threshold}")

    # 4.Segment the image using the threshold
    segmented_image = apply_threshold_segmentation(
        image=gray_scale, threshold=threshold
    )
    print(f"segmented image's shape: {segmented_image.shape}")
    write_image(f"images/{image_num}/03-segmented-image.jpg", segmented_image)
