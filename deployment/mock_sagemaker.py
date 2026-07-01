import sys
import os 

sys.path.append('/opt/ml/model/code')

import inference
import io
import torch
from flask import Flask, request, jsonify



app = Flask(__name__)


print("⏳ Loading model via inference.py...")
model = inference.model_fn('/opt/ml/model')
print("✅ Model loaded and ready.")

@app.route('/invocations', methods=['POST'])
def serve():
    
    image_bytes = request.data
    

    input_data = inference.input_fn(image_bytes, 'application/x-image')
    
   
    prediction = inference.predict_fn(input_data, model)
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)