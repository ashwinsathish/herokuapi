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

class model_input(BaseModel):
    category_code : int
    acc_type_code : int
    year : int
    month : int

model = pickle.load(open('xgb_pred_model.pkl','rb'))

@app.post('/predict')

def predict_accidents(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    cat = input_dictionary['category_code']
    acc = input_dictionary['acc_type_code']
    year = input_dictionary['year']
    month = input_dictionary['month']

    input_list = [cat, acc, year, month]

    result = model.predict(input_list)
    return jsonify(result)



     



