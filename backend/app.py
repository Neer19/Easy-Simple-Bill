from flask import Flask, request,jsonify,send_file
from flask_cors import CORS
import datetime 
from PIL import Image
import base64
import io
x = datetime.datetime.now() 
from end_to_end_data_extraction import extractcsv
  
# Initializing flask app 
app = Flask(__name__) 
CORS(app)
  
# CORS(app, resources={r"/*": {"origins": "localhost:3000"}}, allow_headers=["Content-Type"], supports_credentials=True)
  
# Route for seeing a data 
@app.route('/uploadFile', methods=['GET', 'POST']) 
def get_time(): 
  
    # Returning an api for showing in  reactjs 
    directory = './uploadedImage'
    image_file = request.files['file']
    filename = image_file.filename
    image_file.save(filename)

    # Calling model to convert png into structured data
    extractcsv(f"./{filename}", 1)
    
    return send_file('./final.csv',
        as_attachment=True)
  
      
@app.route('/test', methods=['GET', 'POST'])
def get_time2():
    return jsonify({"text": "This is the test route"})

# Running app 
if __name__ == '__main__': 
    app.run(port=8000, debug=True)