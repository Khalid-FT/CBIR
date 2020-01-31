# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import pandas as pd 
import numpy as np
from features.colorHistFeats import histFeats , calcDistanceHist , extractFeaturesHistDatabaseImg 
from features.colorDominantFeats import colorDominantFeats , calcDistanceColorDom , extractFeaturesColDomDatabaseImg  
from features.textureHaralickFeats import haralickTextureFeats , calcDistanceHaralickTexture , extractFeaturesHaralickDatabaseImg
from features.filterGaborFeats import filtreGaborFeats , calcDistanceGabor , extractFeaturesGaborDatabaseImg
from methodes import load_images_from_folder , getIndexImages , distanceTotal
from methodes import saveResults  , normalize , plotHist , plotColorDom
from scipy.spatial.distance import euclidean


#load data
dataImages = load_images_from_folder('static/dataset')

# extract histogram  features --> save to json file 'df_hist.json'
featuresHistDatabase = extractFeaturesHistDatabaseImg(dataImages)

# extract color dominant features --> save to json file 'df_colorDom.json'
nbreDominantColors = 1
featuresColDomDatabase = extractFeaturesColDomDatabaseImg( dataImages , nbreDominantColors)

# extract haralic texture features --> save to json file 'df_haralickTexture.json'
featuresHaralickDatabase = extractFeaturesHaralickDatabaseImg( dataImages)

# extract gabor features --> save to json file 'df_gabor.json'
featuresGaborDatabase = extractFeaturesGaborDatabaseImg(dataImages)


'''
# Test
# Hist
featuresHistDatabase = pd.read_json('features/database/df_hist.json' , orient='split')
distanceHist = calcDistanceHist(queryImage , featuresHistDatabase)

# ColorDom
featuresColDomDatabase= pd.read_json('features/database/df_colorDom.json' , orient='split')
nbreDominantColors = 1
distanceColDom = calcDistanceColorDom(queryImage  , nbreDominantColors  , featuresColDomDatabase)

# haralick texture
featuresHaralickDatabase= pd.read_json('features/database/df_haralick.json' , orient='split')
distanceHaralickTexture = calcDistanceHaralickTexture(queryImage , featuresHaralickDatabase)

# Gabor filtre
featuresGaborDatabase= pd.read_json('features/database/df_gabor.json' , orient='split')
distanceGabor = calcDistanceGabor(queryImage , featuresGaborDatabase)

distanceTotal = distanceTotal(distanceHist , distanceColDom , distanceHaralickTexture , distanceGabor)


indexImages = getIndexImages(distanceTotal , False)
saveResults(indexImages , 15)
cv2.imshow( filename , queryImage)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
