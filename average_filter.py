import cv2


def make_integeral(image):
    integral = cv2.integral(image)
    return integral


def image_padding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded_image = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    return padded_image


# return blurred image
def avg_integral(image, integral_image):
    height = integral_image.shape[0]
    width = integral_image.shape[1]
    # filtered_image = image.astype(np.float32)
    for row in range(2, height - 2):
        for col in range(2, width - 2):
            top_left = integral_image[row - 2][col - 2]
            top_right = integral_image[row - 2, col + 1]
            bottom_left = integral_image[row + 1][col - 2]
            bottom_right = integral_image[row + 1][col + 1]
            image[row][col] = ((bottom_right + top_left) - (top_right + bottom_left)) / 9

    return image
