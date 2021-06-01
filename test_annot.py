import os
import cv2 as cv
from PIL import Image

annot_file = './data/train_annotation.txt'
colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]

with open(annot_file, 'r') as f:
    lines = f.readlines()
line = lines[12]

folder = 'C:/Users/HP/work/data/image/japan_coins/data/obj'
parts = line.split(' ')
fname = parts[0]

img = cv.imread(os.path.join(folder, fname))

boxs = parts[1:]
# breakpoint()
for box in boxs:
    xmin, ymin, xmax, ymax, label = list(map(int, box.split(',')))
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), colors[label], 2)
# breakpoint()
img = Image.fromarray(img)
img.show()