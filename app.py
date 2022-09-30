from flask import Flask, render_template, request
from onnx_infer import DoInference
import cv2

app = Flask(__name__)

model = DoInference()

# @app.route('/')
# def hello_world():
#     return "This is Joel's new server-2 again"

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_2():
   if request.method == 'POST':
      f = request.files['file']
      f.save('uploads/input.png')

      image = cv2.imread('uploads/input.png', cv2.IMREAD_GRAYSCALE)
      pred_class = model.get_image_prediction(image)

      return f'Predicted Class is: {pred_class}'

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)#, ssl_context='adhoc')