import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

sc = ShapeContext()

def get_contour_bounding_rectangles(gray):
    '''
        return ->   Getting all 2nd level bouding boxes based on
                    contour detection algorithm.
    '''
    cnts, aux = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return res

def parse(sc, path):
    '''
        return ->   Return an array of array of descriptors of every shape
                    found in the img referred by path
    '''
    img = cv2.imread(path, 0)

    # invert image colors
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # making shapes fat for better contour detection
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # getting our shapes one by one
    rois = get_contour_bounding_rectangles(img)
    grayd = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv2.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        nums.append((r[0], r[1], r[2], r[3]))

    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
    descs = []
    for i, r in enumerate(nums):
        if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
            continue
        points = sc.canny_edge_shape(img[r[1]:r[3], r[0]:r[2]])

        aux = img[r[1]:r[3], r[0]:r[2]]
        descriptor = sc.compute(points)

        descs.append(descriptor)
    return np.array(descs)

def match(base, current):

    abecedario = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    costes = []
    for b in base:
        costes.append(sc.cost(b, current))

    #char = str(np.argmin(costes))

    char = abecedario[np.argmin(costes)]

    #if char == '10':
    #    char = "/"

    return char

base_0123456789 = parse(sc, '../resources/sc/ABC.png')
recognize = parse(sc, '../resources/sc/AM.png')
res = ""
for r in recognize:
    res += match(base_0123456789, r)

base = cv2.imread('../resources/sc/ABC.png')
img = cv2.imread('../resources/sc/AM.png')
plt.imshow(base)
plt.show()
plt.imshow(img)
plt.show()
print(res)
