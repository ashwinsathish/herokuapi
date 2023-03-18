import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from Flask import jsonify
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
    Category_code : int
    Acc_type_code : int
    Year : int
    Month : int

model = pickle.load(open('xgb_pred_model.pkl','rb'))

@app.post('/accident_prediction')

def predict_accidents(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    cat = input_dictionary['Category_code']
    acc = input_dictionary['Acc_type_code']
    year = input_dictionary['Year']
    month = input_dictionary['Month']

    input_list = [[cat, acc, year, month]]

    result = model.predict(input_list)
        
    response = {'predictions': result}
    return jsonify(response)



