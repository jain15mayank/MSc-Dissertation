import cv2
import os

outputDir = 'Reshaped Signs'
finRes = 100
dirCounter = dict()

oriDir = 'Cropped Signs'
if not os.path.exists(outputDir):
	os.makedirs(outputDir)

for filename in os.listdir(oriDir):
	img = cv2.imread(os.path.join(oriDir,filename))
	new_img = cv2.resize(img, (finRes,finRes))
	cv2.imwrite(outputDir+'/'+filename, new_img)