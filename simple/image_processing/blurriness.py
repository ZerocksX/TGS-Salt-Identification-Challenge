import cv2
import numpy as np
import png

# ID = "1d9bc2591e"
ID = "0a1742c740"
PNG_PATH = "data/train/images/%s.png" % ID
DEPTHS = {ID: 594}


def _get_grayscale(r, g, b):
    return round(0.299 * r + 0.587 * g + 0.114 * b, 4)


def variance_of_lapacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_blurriness_for_pixel(pixels, window_width=10, window_height=10, transform=lambda x: x):
    # window_width, window_height = 10, 10
    blurriness_map = {}
    width = len(pixels[0])
    height = len(pixels)
    for x in range(height - window_height):
        for y in range(width - window_width):
            window = [[pixels[x + j][y + i] for i in range(window_height)] for j in range(window_width)]
            blurriness = variance_of_lapacian(np.array(window))
            for j in range(window_width):
                for i in range(window_height):
                    curr_x = x + j
                    curr_y = y + i
                    counter = 1
                    tmp_bluriness = blurriness
                    if (curr_x, curr_y) in blurriness_map:
                        tmp_bluriness += blurriness_map[(curr_x, curr_y)][0]
                        counter += blurriness_map[(curr_x, curr_y)][1]
                    blurriness_map[(curr_x, curr_y)] = (tmp_bluriness, counter)
    image = [[0 for i in range(width)] for j in range(height)]
    for (x, y), (blurriness, counter) in blurriness_map.items():
        print((x, y), blurriness / counter)
        image[x][y] = transform(blurriness / counter)
    return np.array(image)


def _get_grayscale_image(path):
    file = open(path, 'rb')
    reader = png.Reader(file=file)
    width, height, pixel_map, metadata = reader.read()
    file.close()
    print(width, height, metadata)
    pixels = []
    for pixel_info in pixel_map:
        pixels.append([_get_grayscale(*(pixel_info[i + 0], pixel_info[i + 1], pixel_info[i + 2])) for i in
                       range(0, len(pixel_info), 3)])

    return pixels


if __name__ == '__main__':
    image = cv2.imread(PNG_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray)
    print(variance_of_lapacian(gray))
    mask = get_blurriness_for_pixel(gray, transform=lambda x: 0 if x > 50 else 255)
    cv2.imwrite('mask.png', mask)
    mask = cv2.imread('mask.png')
    cv2.imshow("Mask", mask)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
