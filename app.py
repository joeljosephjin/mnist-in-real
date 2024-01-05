# import syscd 
# sys.path.append('/home/ec2-user/.local/lib/python3.7/site-packages')

from flask import Flask, render_template, request, jsonify
from onnx_infer import DoInference
import cv2
import argparse
import base64


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

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
      save_img_path = 'uploads/input.png'
      f.save(save_img_path)

      image = cv2.imread(save_img_path, cv2.IMREAD_GRAYSCALE)
      pred_class = model.get_image_prediction(image)

      return f'Predicted Class is: {pred_class}'
   

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.json:
        # Get the image data from the request
        image_data = request.json['image']
        captured_image_path = 'data/uploaded_image.png'
        
        # Process the image data (save to a file, perform operations, etc.)
        # For example, to save the image data to a file
        image_data = image_data.split(',')[1]  # Remove 'data:image/png;base64,' from the beginning
        with open(captured_image_path, 'wb') as file:
            file.write(base64.b64decode(image_data))

        image = cv2.imread(captured_image_path, cv2.IMREAD_GRAYSCALE)
        pred_class = model.get_image_prediction(image)

        # return 
        return jsonify({"message": f'Predicted Class is: {pred_class}', "predicted_class": f'{pred_class}'})
        # return jsonify({"message": f'Predicted Class is: {pred_class}'})
        # return jsonify({"pred_class": f'Predicted Class is: {pred_class}'})
    else:
        return 'No image data found in the request', 400  # Return a bad request response if no image data is found



if __name__=='__main__':
    if args.debug:
      app.run(host='0.0.0.0', port=args.port, debug=True)
    else:
      app.run(host='0.0.0.0', port=args.port)#, ssl_context='adhoc')