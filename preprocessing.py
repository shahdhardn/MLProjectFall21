import pandas as pd
import os
import pydicom as dicom
import numpy as np
from PIL import Image
from skimage import io
from skimage.io import imread
from skimage import color


def doPreprocessing():
	# Convert one hot vectors into categorical labels
	# Atypical=0, Indeterminate= 1, Negative=2, Typical=3
	df = pd.read_csv('RSNACOVID/train_study_level.csv')
	df.head()
	labels = []

	# #Atypical=0, Indeterminate= 1, Negative=2, Typical=3
	class_names = ['Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'Typical Appearance']
	for i, (index, row) in enumerate(df.iterrows()):
		for cls in class_names:
			if row[cls] == 1:
				labels.append(class_names.index(cls))

	df = pd.DataFrame({'ID': df['id']})

	df['Labels'] = labels

	# Images are grayscale. Create folder for each class
	os.makedirs('ML Pro Dataset grayscale', exist_ok=True)
	class_names = ['Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'Typical Appearance']
	for i in class_names:
		os.makedirs('ML Pro Dataset grayscale/' + i, exist_ok=True)

	# Atypical=0, Indeterminate= 1, Negative=2, Typical=3
	# Resize images to 224x224 and Save images in their respective folders
	for i in os.listdir('RSNACOVID/train/'):
		for j in os.listdir('RSNACOVID/train/' + i):
			for k in os.listdir('RSNACOVID/train/' + i + '/' + j):
				im = dicom.dcmread('RSNACOVID/train/' + i + '/' + j + '/' + k)
				# print('RSNACOVID/train/'+i+'/'+j+'/'+k)
				im.PhotometricInterpretation = 'YBR_FULL'
				im = im.pixel_array.astype(float)
				rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float image
				final_image = np.uint8(rescaled_image)  # int image
				final_image = Image.fromarray(final_image)  # PIL image
				final_image = final_image.resize((224, 224))
				if int(df.loc[df['ID'] == i + '_study']['Labels']) == 2:
					final_image.save('ML Pro Dataset grayscale/Negative for Pneumonia/' + i + '.png')
				elif int(df.loc[df['ID'] == i + '_study']['Labels']) == 0:
					final_image.save('ML Pro Dataset grayscale/Atypical Appearance/' + i + '.png')
				elif int(df.loc[df['ID'] == i + '_study']['Labels']) == 1:
					final_image.save('ML Pro Dataset grayscale/Indeterminate Appearance/' + i + '.png')
				elif int(df.loc[df['ID'] == i + '_study']['Labels']) == 3:
					final_image.save('ML Pro Dataset grayscale/Typical Appearance/' + i + '.png')

	# Create RGB version of images. create folder for each class
	os.makedirs('ML Pro Dataset RGB', exist_ok=True)
	class_names = ['Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'Typical Appearance']
	for i in class_names:
		os.makedirs('ML Pro Dataset RGB/' + i, exist_ok=True)

	# Transform gray scale images to RGB

	org_path = 'ML Pro Dataset grayscale'
	new_path = 'ML Pro Dataset RGB'
	for i in os.listdir(org_path):
		for j in os.listdir(org_path + '/' + i):
			image = io.imread(org_path + '/' + i + '/' + j)

		if len(image.shape) == 3:
			path = new_path + '/' + i + '/' + j
			io.imsave(path, image)
		elif len(image.shape) == 2:
			image = color.gray2rgb(image)
			path = new_path + '/' + i + '/' + j
			io.imsave(path, image)



if __name__=="__main__":
	doPreprocessing()
