import os
import glob
import cv2 as cv

test_file = 'C:/Users/HP/work/data/image/japan_coins/data/test.txt'
train_file = 'C:/Users/HP/work/data/image/japan_coins/data/train.txt'

train_annotation = './data/train_annotation.txt'
test_annotation = './data/test_annotation.txt'

data_path = 'C:/Users/HP/work/data/image/japan_coins/data/obj'

def parse_file2line(base_name, img_path, annot_path):
    img = cv.imread(img_path, 0)
    H, W = img.shape
    output_line = basename
    with open(annot_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' ')
            label = parts[0]
            x, y, w, h = list(map(float, parts[1:]))
            x *= W
            y *= H
            w *= W
            h *= H
            xmin = x - w//2
            ymin = y - h//2
            xmax = xmin + w
            ymax = ymin + h
            box = map(int, [xmin, ymin, xmax, ymax])
            output_line += ' '
            for x in box:
                output_line += str(x)
                output_line += ','
            output_line += label
    output_line += '\n'
    return output_line

f_annot_test = open(test_annotation, 'w')
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        basename = os.path.basename(line)
        img_path = os.path.join(data_path, basename)
        annot_path = os.path.join(data_path, basename.split('.')[0]+'.txt')
        # breakpoint()
        if os.path.exists(img_path):
            if os.path.exists(annot_path):
                line = parse_file2line(basename, img_path, annot_path)
                f_annot_test.write(line)
f_annot_test.close()    

f_annot_train = open(train_annotation, 'w')
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        basename = os.path.basename(line)
        img_path = os.path.join(data_path, basename)
        annot_path = os.path.join(data_path, basename.split('.')[0]+'.txt')
        # breakpoint()
        if os.path.exists(img_path):
            if os.path.exists(annot_path):
                line = parse_file2line(basename, img_path, annot_path)
                f_annot_train.write(line)
f_annot_train.close()                

