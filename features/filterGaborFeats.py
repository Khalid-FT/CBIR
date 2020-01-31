
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import numpy as np
from pandas import DataFrame 
from scipy.spatial.distance import euclidean
from methodes import normalize

# define gabor filter bank with different orientations and at different scales
def build_filters():
	filters = []
	ksize = 9
	#define the range for theta and nu
	for theta in np.arange(0, np.pi, np.pi / 8):
		for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
			kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv2.CV_32F)
			kern /= 1.5*kern.sum()
			filters.append(kern)
	return filters

#function to convolve the image with the filters
def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum
# extract gabor filter feature vector
def filtreGaborFeats(image):
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    feats = []
    filters = build_filters()
    f = np.asarray(filters)  
    #calculating the local energy for each convolved image
    for j in range(20):
        temp = 0
        res = process(image, f[j])
        for p in range(128) :
            for q in range(128):
                temp = temp + res[p][q]*res[p][q]
        feats.append(temp)
    #calculating the mean amplitude for each convolved image	
    for j in range(20):
        temp = 0
        res = process(image, f[j])
        for p in range(128) :
            for q in range(128):
                temp = temp + abs(res[p][q])
        feats.append(temp)
 	#feat matrix is the feature vector for the image
    feats = np.array(feats)
    return feats 

# extract gabor features from databse images
def extractFeaturesGaborDatabaseImg(dataImages):
    feats = []
    print('features extracting ...')
    for img in dataImages:
        features = filtreGaborFeats(img) 
        feats.append(features)
    df_gabor = DataFrame(feats)
    df_gabor.to_json(r'features/json/df_gabor.json' , orient='split')
    print('gabor features extracted -> path = features/json/df_gabor.json ')
    return df_gabor

# calculate euclidian distance for Gabor feats
def calcDistanceGabor( queryImage  ,  df_gabor) :
    featsVectors = df_gabor.values.tolist()
    distances = {}
    print('--------------- calculate distances for Gabor Feats  -------------------')
    for i in range(len(featsVectors)):
        queryFeatures = filtreGaborFeats(queryImage)
        imgFeatures = featsVectors[i]
        dist = euclidean(queryFeatures,imgFeatures)
        #dist = cv2.compareHist(queryFeatures, imgFeatures, cv2.HISTCMP_CHISQR)
        print('dist (query image , image %d )'%(i+1)+'-----> %f'%dist)
        distances[i] = dist       
    print('------------calculate distances for Gabor Feats completed ---------')
    distances = normalize(distances , 25)
    return distances




        