import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
from pandas import DataFrame 
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from methodes import normalize


# color dominant
def colorDominantFeats(image , nbreDominantColors):
    
    img = image
    img = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    # Changement d'echelle, pour avoir moins d'exemples
    width  = 50 # largeur cible
    ratio  = img.shape[0]/img.shape[1]
    height = int(img.shape[1]*ratio)
    dim    = (width, height)
    img = cv2.resize(img, dim)
    # Paramètres d'apprentissage
    # Un triplet (B, G, R) par ligne 
    examples = img.reshape((img.shape[0] * img.shape[1], 3))
    # Groupement par la technique des KMEANS
    kmeans = KMeans(n_clusters = nbreDominantColors , random_state=42)
    kmeans.fit(examples) 
    # Les Centres des groupement représentent les couleurs dominantes (B, G, R)
    colors = kmeans.cluster_centers_.astype(int)
    return colors.flatten()

# extract color dom features from databse images
def extractFeaturesColDomDatabaseImg(dataImages , nbreDominantColors):
    feats = []
    print('features extracting ...')
    for img in dataImages:
        features = colorDominantFeats(img , nbreDominantColors ) 
        feats.append(features)
    df_colorDom = DataFrame(feats)
    df_colorDom.to_json(r'features/json/df_colorDom.json' , orient='split')
    print('colorDom features extracted -> path = features/json/df_colorDom.json ')
    return df_colorDom

# calculate euclidian distance for color Dom feats 
def calcDistanceColorDom( queryImage  , nbreDominantColors , df_colorDom):
    featsVectors = df_colorDom.values.tolist()
    distances = {}
    print('--------- calculate distances for color dominant  feats -----------')
    for i in range(len(featsVectors)):
        queryFeatures = colorDominantFeats(queryImage , nbreDominantColors)
        imgFeatures = featsVectors[i]
        dist = euclidean(queryFeatures,imgFeatures)
        print('dist (query image , image %d )'%(i+1)+'-----> %f'%dist)
        distances[i] = dist
    print('------------calculate distances for colorDom Feats completed ---------')
    distances = normalize(distances , 25)  
    return distances
