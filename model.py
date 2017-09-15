import csv
import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D,Lambda, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import sklearn

def readCSV(csvPath):
	lines = []
	with open(csvPath) as csvfile:
		reader = csv.reader(csvfile)
		i = 0
		for line in reader:
			if(i > 0):
				lines.append(line)
			i+=1
	return lines

def crop(image):
    return image[60:-20, :, :]

def flipImg(image, angle):
	if(randint(0, 1)==1): #50% flip
		return cv2.flip(image,1), -angle
	else:
		return image, angle

def preprocess(image, width, height):
	return cv2.resize(crop(image),(width, height), interpolation = cv2.INTER_AREA)

def translateX(image, angle):
	transX = randint(-50, 50)
	angle += transX * 0.002
	transM = np.float32([[1, 0, transX],[0, 1, 0]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, transM, (width, height))
	return image, angle

def augment(image, width, height, angle):

	image, angle = flipImg(image, angle)
	#image, angle = translateX(image, angle)
	# preprocess image
	img = preprocess(image, width, height)
	return img, angle

def buildModel(width, height, channel):#, lr):

	model = Sequential()
	
	#NVIDIA
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(height, width, channel)))
	model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2), border_mode='same'))
	model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2), border_mode='same'))
	model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))	

	"""
	#COMMA.AI
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(height,width, channel)))
	model.add(Conv2D(16, 5, 5, activation='elu', subsample=(2, 2), border_mode='same'))
	model.add(Conv2D(32, 5, 5, activation='elu', subsample=(2, 2), border_mode='same'))
	model.add(Conv2D(64, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	"""

	model.compile(optimizer='adam', loss='mean_squared_error')

	return model

def generator(samples, batch_size, width, height):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	
            	# choose random center, left or right image with center in 2 out of 4 cases
            	x = randint(0,2) 

            	if(x==1): #left
            		name = 'data/IMG/'+batch_sample[1].split('/')[-1]
            		clr_image = plt.imread(name)
            		clr_angle = float(batch_sample[3]) + 0.2
            	elif(x==2): #right
            		name = 'data/IMG/'+batch_sample[2].split('/')[-1]
            		clr_image = plt.imread(name)
            		clr_angle = float(batch_sample[3]) - 0.2
            	else: #center
            		name = 'data/IMG/'+batch_sample[0].split('/')[-1]
            		clr_image = plt.imread(name)
            		clr_angle = float(batch_sample[3])

            	aug_img, aug_angle = augment(clr_image, width, height, clr_angle)

            	images.append(aug_img)
            	angles.append(aug_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

if __name__ == '__main__':
	
	# HYPERPARAMETERS
	#################
	WIDTH = 80
	HEIGHT = 20
	CHANNEL = 3
	BATCHSIZE = 128
	EPOCHS = 10
	#################

	# read csv file
	samples = readCSV('data/driving_log.csv')
	print('Done reading csv.')

	# split into train and val samples
	train_samples, validation_samples = train_test_split(samples, test_size=0.1)
	print('Training samples:', len(train_samples))
	print('Validation samples:', len(validation_samples))

	# building model
	model = buildModel(WIDTH, HEIGHT, CHANNEL)#, LR)
	print('Done building model.')

	# create generators
	train_generator = generator(train_samples, BATCHSIZE, WIDTH, HEIGHT)
	validation_generator = generator(validation_samples, BATCHSIZE, WIDTH, HEIGHT)

	# save multiple checkpoints
	filepath=str(WIDTH) + '-' + str(HEIGHT)+ '-' + str(CHANNEL) +'-model-{epoch:02d}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
	callbacks_list = [checkpoint]

	# fit model
	model.fit_generator(train_generator, samples_per_epoch=len(train_samples)
		, validation_data=validation_generator,nb_val_samples=len(validation_samples)
		, nb_epoch=EPOCHS, callbacks=callbacks_list)
