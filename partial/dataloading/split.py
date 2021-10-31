import numpy as np
import glob
import imageio
import csv 
import os


files = glob.glob("../datasets/images/*.png")
numFiles = len(files)
np.random.seed(0)
shuffledIndices = np.random.permutation(numFiles)

# Use 20% of the images for Training
numTrain = round(0.60 * numFiles)
trainingIdx = shuffledIndices[:numTrain]

# Use 20% of the images for validation
numVal = round(0.20 * numFiles)
valIdx = shuffledIndices[numTrain:numTrain+numVal]

# Use the rest for testing.
testIdx = shuffledIndices[numTrain+numVal:]
# 295 98 98
print(numFiles, numTrain, numVal, len(testIdx))
for i, img_name in enumerate(files):
    print(i, img_name)

    # img = imageio.imread(img_name)
    label_name = "../datasets/labels/SegmentationClass/wasp2/"+img_name.split("/")[-1]
    label_name = os.path.abspath(label_name)
    img_name = os.path.abspath(img_name)

    # label = imageio.imread(label_name)
    # print(img.shape)
    # print(label.shape)
    # break
    if i in trainingIdx:
        with open('../datasets/train/img_name1.txt', 'a') as the_file:
            the_file.write(img_name+"\n")
        with open('../datasets/train/label_name1.txt', 'a') as the_file_:
            the_file_.write(label_name+"\n")
            
    if i in valIdx:
        with open('../datasets/val/img_name1.txt', 'a') as the_file:
            the_file.write(img_name+"\n")
        with open('../datasets/val/label_name1.txt', 'a') as the_file_:
            the_file_.write(label_name+"\n")
            
    if i in testIdx:
        with open('../datasets/test/img_name1.txt', 'a') as the_file:
            the_file.write(img_name+"\n")
        with open('../datasets/test/label_name1.txt', 'a') as the_file_:
            the_file_.write(label_name+"\n")
            