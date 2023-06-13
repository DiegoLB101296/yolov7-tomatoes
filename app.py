import io
from operator import truediv
import os
import json
from PIL import Image
import cv2
import pickle

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect, Response

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# finds the model inside your directory automatically - works only if there is one model
#def find_model():
#    for f  in os.listdir():
#        if f.endswith(".pt"):
#            return f
#    print("please place a model file in this directory!")
    
#model_name = find_model()
#model =torch.hub.load("WongKinYiu/yolov7", 'custom',model_name)
model = pickle.load(open('model.pkl', 'rb'))
model.eval()
#pickle.dump(model, open('model.pkl', 'wb'))
def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
            
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = 'image0.jpg'
        
        return render_template('result.html',result_image = filename,model_name = model)
        #return render_template('result.html',result_image = results,model_name = model_name)

    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def handle_video():
    # some code to be implemented later
    pass

#def gen_frames():
#    camera = cv2.VideoCapture(0)
#    while True:
#        success, frame = camera.read() 
#        if not success:
#            break
#        else:
#            ret, buffer = cv2.imencode('.jpg', frame)
#            frame = buffer.tobytes()
#            yield(b'--frame\r\n'
#                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

          
@app.route('/webcam', methods=['GET', 'POST'])

#@app.route('/video_feed')
#def video_feed():
#    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def web_cam():
    # TODO: some code to be implemented later
    pass

