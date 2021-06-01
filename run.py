from tensorflow.data import TextLineDataset

dataset = TextLineDataset(filenames='data/annot_train.txt')

for x in dataset.take(1):
    print(x)


