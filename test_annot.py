import os
import cv2 as cv
from PIL import Image

annot_file = './data/test_annotation.txt'
colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]

with open(annot_file, 'r') as f:
    lines = f.readlines()
# line = lines[30]
# line = "20190619_090121.jpg 187,134,290,195,0 290,149,409,221,1 153,193,272,278,2"
line = "20506_main.jpg 133,1,234,47,0 2,27,105,104,0 1,175,45,259,0 59,209,170,299,0 89,95,195,177,0"

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