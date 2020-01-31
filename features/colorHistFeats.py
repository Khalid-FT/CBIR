

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cv2
from pandas import DataFrame 
from scipy.spatial.distance import euclidean
from methodes import normalize


# calculate histogram
def histFeats(image):
        feats = []
        image = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
        chans = cv2.split(image)
        colors = ("h", "s", "v")
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            feats.extend(hist)
        return feats
    
# extract hsit features from databse images
def extractFeaturesHistDatabaseImg(dataImages):
    feats = []
    print('features extracting ...')
    for img in dataImages :
        features = histFeats(img) 
        feats.append(features)
    df_hist = DataFrame(feats)
    df_hist.to_json(r'features/json/df_hist.json' , orient='split')
    print('hist features extracted -> path = features/json/df_hist.json ')
    return df_hist

# calculate euclidian distance for hist feats 
def calcDistanceHist( queryImage  ,  df_hist) :
    featsVectors = df_hist.values.tolist()
    distances = {}
    print('--------------- calculate distances for Hist Feats --------------------')
    for i in range(len(featsVectors)):
        queryFeatures = histFeats(queryImage)
        imgFeatures = featsVectors[i]
        dist = euclidean(queryFeatures,imgFeatures)
        print('dist (query image , image %d )'%(i+1)+'-----> %f'%dist)
        distances[i] = dist
    print('------------calculate distances for Hist Feats completed ---------')
    distances = normalize(distances , 25)
    return distances





