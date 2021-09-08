from numpy import ndarray
import cv2
import numpy as np
from typing import List, Tuple
import os
from pandas import DataFrame
import pandas as pd
from PIL import ImageTk, Image
import matplotlib.pylab as plt
import matplotlib.cm as cm


def auto_canny(image: ndarray, sigma: float = 0.33) -> ndarray:
    """applies edge detection
    https://www.pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, apertureSize=3, L2gradient=True)

    return edged


def contours(edges: ndarray) -> (List, ndarray):
    contours: List
    hierarchy: ndarray
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = dict()
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = cv2.contourArea(cont)
        if 20 < area and 20 < w and h > 10:
            contours_dict[(x, y, w, h)] = cont

    contours_filtered: List = sorted(contours_dict.values(), key=cv2.boundingRect)
    blank_background = np.zeros_like(edges)
    img_contours: ndarray = cv2.drawContours(blank_background, contours_filtered, -1, (255, 255, 255), thickness=2)

    return contours_filtered, img_contours


def is_overlapping_horizontally(box1: Tuple, box2: Tuple) -> bool:
    x1, _, w1, _ = box1
    x2, _, _, _ = box2
    if x1 > x2:
        return is_overlapping_horizontally(box2, box1)
    return (x2 - x1) < w1


def merge(box1: Tuple, box2: Tuple) -> Tuple:
    assert is_overlapping_horizontally(box1, box2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    w = max(x1 + w1, x2 + w2) - x
    y = min(y1, y2)
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)


def windows(contours: List) -> List:
    boxes = []
    for cont in contours:
        box: Tuple = cv2.boundingRect(cont)
        if not boxes:
            boxes.append(box)
        else:
            if is_overlapping_horizontally(boxes[-1], box):
                last_box = boxes.pop()
                merged_box = merge(box, last_box)
                boxes.append(merged_box)
            else:
                boxes.append(box)
    return boxes


def save_pixels(boxes: List, img_file: str, len_boxes: int, img: ndarray) -> None:
    # create directory for all pixels
    main_folder = "./images_data"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    # create folder inside for the image
    if not os.path.exists(os.path.join(main_folder, img_file.replace(".jpg", ''))):
        os.makedirs(os.path.join(main_folder, img_file.replace(".jpg", '')))
    for n in range(len_boxes):
        x, y, w, h = boxes[n]
        plt.axis('off')

        if y < 10:
            y = 10
        if x < 10:
            x = 10

        roi = img[y - 10:y + h + 5, x - 10:x + w + 5]
        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        a = str(n + 1)
        cv2.imwrite(os.path.join(main_folder, img_file.replace(".jpg", ''), 'basamak' + a + '.png'), roi)

    columnNames = list()

    for i in range(1024):
        pixel = 'pixel'
        pixel += str(i)
        columnNames.append(pixel)

    train_data = pd.DataFrame(columns=columnNames)

    for n in range(len_boxes):
        a = str(n + 1)
        b = 'basamak' + a + '.png'
        img = Image.open(os.path.join(main_folder, img_file.replace(".jpg", ''), b))
        rawData = img.load()
        data = []
        for y in range(32):
            for x in range(32):
                data.append(rawData[x, y])
        k = 0
        train_data.loc[0] = [data[k] for k in range(1024)]
        train_data = train_data.div(255)
        train_data.to_csv(os.path.join(main_folder, img_file.replace(".jpg", ''), "train_converted" + a + ".csv"),
                          index=False)
