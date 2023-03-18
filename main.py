import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from flask import jsonify
from urllib import request

app = FastAPI()

origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

model = pickle.load(open('xgb_pred_model.pkl','rb'))

@app.post('/accident_prediction')

def predict_accidents():
        
    data = request.get_json()

    # Extract the values for each feature
    category_code = data['category_code']
    acc_type_code = data['accident_type_code']
    year = data['year']
    month = data['month']

    input_list = [[category_code, acc_type_code, year, month]]

    prediction = model.predict(input_list)

    # Return the prediction as a JSON object
    response = {'predictions': prediction.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

     



