
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import mahotas as mt
from pandas import DataFrame 
from scipy.spatial.distance import euclidean
from methodes import normalize

# haralick
def haralickTextureFeats(image):
        #image = cv2.imread(filename)
        # calculate haralick texture features for 4 types of adjacency
        image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        textures = mt.features.haralick(image)
        # take the mean of it and return it
        feats = textures.mean(axis=0)
        return feats
    
# extract texture features from databse images
def extractFeaturesHaralickDatabaseImg(dataImages):
    feats = []
    print('features extracting ...')
    for img in dataImages:
        features = haralickTextureFeats(img) 
        feats.append(features)
    df_haralick = DataFrame(feats)
    df_haralick.to_json(r'features/json/df_haralick.json' , orient='split')
    print('haralick texture features extracted -> path = features/json/df_haralick.json ')
    return df_haralick   

# calculate euclidian distance for haralick feats 
def calcDistanceHaralickTexture(queryImage , df_haralick) :
    featsVectors = df_haralick.values.tolist()
    distances = {}
    print('--------------- calculate distances for Haralick Feats  -------------------')
    for i in range(len(featsVectors)):
        queryFeatures = haralickTextureFeats(queryImage)
        imgFeatures = featsVectors[i]
        #imgFeatures = cv2.normalize(imgFeatures, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        dist = euclidean(queryFeatures,imgFeatures)
        print('dist (query image , image %d )'%(i+1)+'-----> %f'%dist)
        distances[i] = dist
    print('------------calculate distances for HaralickTexture Feats completed ---------')
    distances = normalize(distances , 25) 
    return distances
 