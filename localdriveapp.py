''' This code is for facedetection api using the local drive space for storing and processing on images '''
import os, boto3, botocore
from botocore.client import Config
from flask import Flask, send_file, jsonify 
from flask import render_template, request, redirect, url_for, send_from_directory  
import cv2, urllib , urllib.request
import numpy as np 
from werkzeug.utils import secure_filename 


UPLOAD_FOLDER = '/home/kaustubh/facedetection_flask/static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)   
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/main/<filename>")
def processing(filename):
	data = {"success": False}
	path = '/home/kaustubh/facedetection_flask/static/' + str(filename)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	image = cv2.imread(path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.04, 6, minSize=(30,30))
	rects = [(int(x), int(y), int(x + w), int(y + h),int((y+h)+(x+w)),int(w),int(h)) for (x, y, w, h) in faces]
	data.update({"num_faces": len(rects), "faces": rects, "success": True})


	return jsonify(data)

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    data = {'success':True}
    if 'file' not in request.files:
        return ('upload a file')
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            print(filename)
            return redirect(url_for('processing',filename=filename))
    return ('Select an image file from drive')        

if __name__ == "__main__":

    app.run(debug=True)      
