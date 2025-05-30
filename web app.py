import os
import numpy as np #used for numerical analysis
from flask import Flask,request,render_template
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
#render_template- used for rendering the html pages
from tensorflow.keras.models import load_model#to load our trained model
from tensorflow.keras.preprocessing import image
import tensorflow

import numpy as np
import os


import librosa
import librosa.display
from IPython.display import Audio
import warnings

app=Flask(__name__)#our flask app
model=load_model('sser.h5')#loading the model
def extract_mfcc(filename):
     y, sr = librosa.load(filename, duration=3, offset=0.5)
     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
     return mfcc

@app.route("/") #default route
def about():
    return render_template("about.html")#rendering html page

@app.route("/about") #default route
def home():
    return render_template("about.html")#rendering html page

@app.route("/info") #default route
def information():
    return render_template("info.html")#rendering html page

@app.route("/upload") #default route
def test():
    return render_template("index6.html")#rendering html page


@app.route("/predict",methods=["GET","POST"]) #route for our prediction
def upload():
    if request.method=='POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        xmfcc=extract_mfcc(filepath)
        xmfcc
        type(xmfcc)
        X = [xmfcc]
        X = np.array(X)
        X.shape
        X = np.expand_dims(X, -1)
        X.shape
        test=model.predict(X)
        names=["fear" ,"neutral", "happy",  "disgust",  "angry ","sad"    ,  "ps  "]
        maxval=np.argmax(test)
        
        print("output is",names[maxval])
        
        
        return names[maxval]#resturing the result
    return None

#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=False)#running our app
    #app.run(host='0.0.0.0', port=8000)
            
            