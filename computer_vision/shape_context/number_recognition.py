import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

sc = ShapeContext()

def get_contour_bounding_rectangles(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """
    cnts, aux = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return res

def parse_nums(sc, path):
    img = cv2.imread(path, 0)

    # invert image colors
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # making numbers fat for better contour detectiion
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # getting our numbers one by one
    rois = get_contour_bounding_rectangles(img)
    grayd = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv2.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        #plt.imshow(grayd)
        #plt.show()
        nums.append((r[0], r[1], r[2], r[3]))
    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
    descs = []
    for i, r in enumerate(nums):
        if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
            continue
        points = sc.canny_edge_shape(img[r[1]:r[3], r[0]:r[2]])
        #points, _ = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 50, 15)

        aux = img[r[1]:r[3], r[0]:r[2]]
        #plt.imshow(aux)
        #plt.show()
        #print(len(points))
        for p in points:
            aux = cv2.circle(aux, (p[1], p[0]), 0 , 128)
        plt.imshow(aux)
        plt.show()
        descriptor = sc.compute(points).flatten()

        descs.append(descriptor)
    return np.array(descs)

def shape_context_cost(nh1, nh2):
            '''
                nh1, nh2 -> normalized histogram
                return cost of shape context of
                two given shape context of the shape.
            '''
            cost = 0
            if nh1.shape[0] > nh2.shape[0]:
                nh1, nh2 = nh2, nh1
            nh1 = np.hstack([nh1, np.zeros(nh2.shape[0] - nh1.shape[0])])
            for k in range(nh1.shape[0]):
                if nh1[k] + nh2[k] == 0:
                    continue
                cost += (nh1[k] - nh2[k])**2 / (nh1[k] + nh2[k])
            return cost / 2.0

def match(base, current):
    """
      Here we are using cosine diff instead of "by paper" diff, cause it's faster
    """
    costes = []
    for b in base:
        costes.append(shape_context_cost(b, current))

    char = str(np.argmin(costes))

    if char == '10':
        char = "/"
    return char

base_0123456789 = parse_nums(sc, '../resources/sc/base.png')
recognize = parse_nums(sc, '../resources/sc/telefono.png')
res = ""
for r in recognize:
    res += match(base_0123456789, r)

base = cv2.imread('../resources/sc/numbers.png')
img = cv2.imread('../resources/sc/telefono.png')
plt.imshow(base)
plt.show()
plt.imshow(img)
plt.show()
print(res)
