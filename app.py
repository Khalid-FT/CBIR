# -*- coding: utf-8 -*-
from flask import Flask, render_template, redirect, url_for, request
import cv2
import pandas as pd 
import numpy as np
import os
from features.colorHistFeats import calcDistanceHist
from features.colorDominantFeats import calcDistanceColorDom 
from features.textureHaralickFeats import calcDistanceHaralickTexture 
from features.filterGaborFeats import  calcDistanceGabor
from methodes import distanceTotal , getIndexImages , saveResults , filenames , saveToJson , feedback

STATIC_FOLDER = './static'
app = Flask(__name__)


UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  

@app.route('/')
def home():
    return render_template('index.html')

# features loading
featuresHistDatabase = pd.read_json('features/json/df_hist.json' , orient='split')
featuresColDomDatabase= pd.read_json('features/json/df_colorDom.json' , orient='split')
featuresHaralickDatabase= pd.read_json('features/json/df_haralick.json' , orient='split')
featuresGaborDatabase= pd.read_json('features/json/df_gabor.json' , orient='split')

path_results = './static/results/'
hh = os.listdir(path_results)
files = filenames(path_results)


@app.route("/output", methods=["GET", "POST"])
def upload_image():
    if request.method == 'POST':
        if request.files:
            #Reading the uploaded image
            image = request.files["image"]
            imageFile = image.filename
            filename = os.path.join('./static/query_img/', imageFile)
            image.save(filename)
            queryImage = cv2.imread(filename)
            filname_NoExten = imageFile[0:-4]
            json_file = 'distances/'+filname_NoExten + '.json'
            
            # check if dist result already exist
            if (os.path.exists(json_file)) :
                 json_dist= pd.read_json(json_file , orient='split')
                 dists = json_dist.set_index('image')['dist'].to_dict()
                 # get index retrieved images
                 indexImagesResults = getIndexImages(dists , False)
                 path_results = './static/results/'
                 numberImagesToRetrieve = 25
                 path_dataImages = './static/dataset/'
                 # save retrieved images to folder results
                 saveResults(path_results , numberImagesToRetrieve , indexImagesResults , path_dataImages)
                 # load retrieved images
                 similar_images = filenames(path_results)
                 return render_template("output.html",query_name=imageFile ,  query_image = filename , similar_images = zip (indexImagesResults ,similar_images)) 
                 print('Done !') 
            else :   
               # Hist distance
               distanceHist = calcDistanceHist(queryImage , featuresHistDatabase)

               # ColorDom distance
               nbreDominantColors = 1
               distanceColDom = calcDistanceColorDom(queryImage  , nbreDominantColors  , featuresColDomDatabase)
               # haralick texture distance
               distanceHaralickTexture = calcDistanceHaralickTexture(queryImage , featuresHaralickDatabase)  
               # Gabor filtre distance
               distanceGabor = calcDistanceGabor(queryImage , featuresGaborDatabase)
               # distance global
               distanceGlobal = distanceTotal( distanceHist , distanceColDom , distanceHaralickTexture , distanceGabor)
               # save distances to json file
               saveToJson(imageFile , distanceGlobal )
               # get index retrieved images
               indexImagesResults = getIndexImages(distanceGlobal , False)
               path_results = './static/results/'
               numberImagesToRetrieve = 25
               path_dataImages = './static/dataset/'
               # save retrieved images to folder results
               saveResults(path_results , numberImagesToRetrieve , indexImagesResults , path_dataImages)
               # load retrieved images
               similar_images = filenames(path_results)
               return render_template("output.html",query_name=imageFile ,  query_image = filename , similar_images = zip (indexImagesResults ,similar_images))
               print('Done !') 
        else :
            return redirect(url_for('home'))

@app.route('/do_something', methods = ['POST'])
def get_python_data():
    jsdata = request.get_json()
    name=jsdata['name_img']
    indexImage=jsdata['index']
    value=jsdata['value']
    query_filename = jsdata['queryimage']
    
    filename = 'distances/'+query_filename[19:-4] + '.json'
    print('filename : ', filename)
    print('index image ' , indexImage)
    print('value ' , value , type(value))
    json_dist = pd.read_json(filename, orient='split')
    feedback( filename , indexImage  , json_dist , value )
    
    return "ok"
           
if __name__ == '__main__':
   app.run()
        
            