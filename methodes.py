# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import numpy as np
import os , os.path
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from collections import Counter 
from matplotlib import pyplot as plt

# read images 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# extract filenames img results
def filenames(folder):
    filenames = []
    for filename in os.listdir(folder):
        filenames.append(filename)
    return filenames   
 
# get index of image similair
def getIndexImages(results , asc):
    sortedResults = sorted(results.items(), key=lambda x: x[1] , reverse = asc )
    arr = []
    for j in range(len(results)): 
       a = sortedResults[j]
       index = a[0]
       arr.append(index)
    return arr        

# clear results folder
def clearResultsFolder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))

# save results in folder    
def saveResults(path_results , numberImages , indexImagesResults , path_dataImages):
    clearResultsFolder(path_results)
    images = load_images_from_folder(path_dataImages)
    for j in range(len(indexImagesResults)):
       if numberImages == j  : break
       filenames = os.listdir(path_dataImages)
       i = indexImagesResults[j]
       filename = os.path.join( path_results,'%d_'%(j+10)+filenames[i]) 
       cv2.imwrite( filename , images[i]  )
       
# normalize distances 
def normalize(distances , scale) :
       #vect = (vector - min(vector)) / (max(vector) - min(vector))*scale
       scaler = MinMaxScaler(feature_range = (0,scale))
       values = distances.values()   
       keys = distances.keys()    
       distances = np.array(list(values))
       distances = scaler.fit_transform(distances.reshape(-1 , 1 ))
       distances = dict(zip( keys , distances ))
       return distances

# get global distance 
def distanceTotal(distanceHist , distanceColDom , distanceHaralickTexture , distanceGabor) :
    # weights
    w_hist = 0.4
    w_ColDom= 0.4
    w_HaralickTexture = 0.1
    w_Gabor = 0.1
    
    for key in distanceHist:     
        distanceHist[key] *= w_hist
        
    for key in distanceColDom:     
        distanceColDom[key] *= w_ColDom
        
    for key in distanceHaralickTexture:     
        distanceHaralickTexture[key] *= w_HaralickTexture
        
    for key in distanceGabor:     
        distanceGabor[key] *= w_Gabor 
        
    distTotal = Counter(distanceHist) + Counter(distanceColDom) + Counter(distanceHaralickTexture) + Counter(distanceGabor)
    return distTotal

# save dist to json 
def saveToJson(filename ,distance) :
    df = pd.DataFrame(list(distance.items()),columns = ['image','dist']) 
    df.to_json(r'distances/' + filename[0:-4] +'.json' , orient='split')
    print(' distance saved ! ')

# modify dist
def feedback( filename , indexImage  , json_dist , value ) :
    dist = json_dist.set_index('image')['dist'].to_dict()
    dist[indexImage]=dist[indexImage]+ np.array(float(value))
    df = pd.DataFrame(list(dist.items()),columns = ['image','dist']) 
    df.to_json(r'' +filename , orient='split')
    print(' dist updated !')
   
# plot Hist    
def plotHist(image) :
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
 
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

# plot color dominant 
def plotColorDom(img , nbreDominantColors):
    cv2.imshow("Image originale", img)
    # Nombre de couleurs  
    nbreDominantColors = nbreDominantColors
        
    # Créer une image temopraire
    barColorW=75
    barColorH=50
    imgr = np.zeros((barColorH, barColorW*nbreDominantColors, 3), dtype=np.uint8)
        
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
    kmeans = KMeans(n_clusters = nbreDominantColors)
    kmeans.fit(examples)
    
    # Les Centres des groupement représentent les couleurs dominantes (B, G, R)
    colors = kmeans.cluster_centers_.astype(int)
      
    for i in range(0,nbreDominantColors):
        cv2.rectangle(imgr, (i*barColorW,0), ((i+1)*barColorW,barColorH), [int(x) for x in colors[i]],-1)

    str_="Lab02: "+str(nbreDominantColors)+" couleurs dominantes";  
    cv2.imshow(str_, imgr)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()