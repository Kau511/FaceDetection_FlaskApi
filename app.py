''' Three query parametes to be passed to this APi
    1 - urls, 2- accuracy(coarse or fine), 3- file(from local drive option if urls not present)   '''
import numpy as np
import cv2, os
import json
from flask import Flask, jsonify, request, url_for, redirect
import urllib, urllib.request
from werkzeug.utils import secure_filename
from config import config
import pyrebase


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET','POST'])
def check_accuracy():
	if 'urls' and 'accuracy' not in request.args:
		if 'file' not in request.files and 'accuracy' not in request.args:
			return ("Enter image url or upload image from drive with accuracy parameter")
	if 'urls' in request.args and 'accuracy' in request.args:
		urls = (request.args['urls'])
		accuracy = (request.args['accuracy'])
		if urls == "" and accuracy == "":
			return ("Enter urls or upload an image")
		if accuracy == 'coarse':
			return redirect(url_for('harr_class',urls = urls))
		elif accuracy == 'fine':
			return redirect(url_for('processing',urls = urls))
	if 'file' in request.files and 'accuracy' in request.args:
		accuracy = (request.args['accuracy'])
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			storage.child("images/" +str(filename)).put(file)
			link = storage.child("images/" + str(filename)).get_url(None)
		if accuracy == 'coarse':
			return redirect(url_for('harr_class',link = link))
		if accuracy == 'fine':
			return redirect(url_for('processing',link = link))	

	return ("Enter both urls=image url and accuracy = fine or coarse")

@app.route("/coarse")
def harr_class():
	data = {'success':True}
	url = request.args.get('urls')
	link = request.args.get('link')
	if link != None:
		resp = urllib.request.urlopen(link)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	elif url != None:
		resp = urllib.request.urlopen(url)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	faces = face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(2,2))
	rects = [(int(x), int(y), int(x + w), int(y + h),int((y+h)+(x+w)),int(w),int(h)) for (x, y, w, h) in faces]
	data.update({"num_faces": len(rects), "faces": rects, "success": True})
	return jsonify(data)

@app.route("/fine")
def processing():
	url = request.args.get('urls')
	link = request.args.get('link')
	if link != None:
		resp = urllib.request.urlopen(link)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	elif url != None:
		resp = urllib.request.urlopen(url)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	faces = []  
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.4:
			data = {'success':False}
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			rect = [(int(startX), int(startY), int(endX), int(endY))]
			faces.append(rect)
	data = {"success":True,"num_faces":(len(faces)),"Faces":faces}
	return jsonify(data)

if __name__ == "__main__":
	app.run(debug=True)
