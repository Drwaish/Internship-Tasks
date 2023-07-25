import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template,redirect,jsonify
from flask_smorest import Blueprint
from flask.views import MethodView

blp = Blueprint("items", __name__, description = "Operations in item" )
model = pickle.load(open("api_built.pkl", "rb"))
data = {  'c_Argentina': 0,
            'c_Canada': 0,
            'c_Estonia': 0,
            'c_Japan': 0,
            'c_Spain': 0,
            's_Kagglazon': 0,
            's_Kaggle Learn': 0,
            's_Kaggle Store': 0,
            'p_Using LLMs to Improve Your Coding': 0,
            'p_Using LLMs to Train More LLMs': 0,
            'p_Using LLMs to Win Friends and Influence People': 0,
            'p_Using LLMs to Win More Kaggle Competitions': 0,
            'p_Using LLMs to Write Better': 0 }
class Item( MethodView):
    def get(self):
        return jsonify(data)

    def post(self):
        data1 = request.get_json()
        country = data1["country"]
        store = data1['store']
        product = data1['product'] 
        print(country, store, product)

        data[country] = 1
        data[store] = 1
        data[product] = 1

        df=pd.DataFrame(data, index = [0])
        pred = model.predict(df)
        print(pred)
        return str(pred[0])