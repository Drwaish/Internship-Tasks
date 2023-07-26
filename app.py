"""Api to predic function of solvent"""
import json
from typing import Dict, Optional, Tuple
import pandas as pd
from flask import Flask, jsonify ,request
from flask_httpauth import HTTPBasicAuth

from prediction import predict


app = Flask(__name__)
auth = HTTPBasicAuth()

# @app.route('/rest-auth')
# @auth.login_required
# def get_response():
# 	return redirect("http://127.0.0.1:3000/prediction")

# @auth.verify_password
# def authenticate(username, password):
#     """
#     Authenticate 
#     """
#     if username and password:
#         if username == 'zain' and password == 'zain':
#             return True
#         else:
#             return False
#     return False

# model = pickle.load(open("api_built.pkl","rb"))

def validte_data(cas: Optional[str] = None, was: Optional[float] = None,
                category: Optional[str] = None) -> bool:
    """
    Validate data type and update status

    Parameters
    ---------
    cas
        cas is string variable of dataset
    was
        was is float variable of dataset
    category
        Category tells about the product
    
    Retrun
    ------
    bool
    """
    if type(cas) is str and type(was) is float and  type(category) is str:
        return True
    else:
        return False

def convert_prediction_ready(cas: Optional[str] = None, category: Optional[str] = None,
                            func : Optional[int] = None) -> Tuple:
    """
    Metadata convert into correspondence integer

    Parameters
    ---------
    cas
        cas is string variable of dataset
    func: 
        Predicted variable for inverse transform

    category
        Category tells about the product
    
    
    Retrun
    ------
    Tuple
    """
    try:
        file = open("meta_data1.json", encoding="Utf-8")
        meta_data=json.load(file)
        if func is None:
            cas_t = meta_data["cas"]
            category_t = meta_data["category"]
            cas_index = -1
            category_index = -1
            for cas_ele in cas_t:
                if cas_ele[0] == cas:
                    cas_index = cas_ele[1]
                    break
            for cat_ele in category_t:
                if cat_ele[0] == category:
                    category_index = cat_ele[1]
                    break
            return (cas_index, category_index)
        else:
            func_t = meta_data["function"]
            func_index = None
            for func_ele in func_t:
                if func_ele[1] == func:
                    func_index = func_ele[0]
                    break
            return (func_index)
    except FileNotFoundError: 
        print("File not Found")
    return (False)

@app.route("/prediction",methods = ["POST","GET"])
def prediction() -> Dict:
    """
    Prediction of data can perform here

    Parameters
    ----------
    None

    Return
    -----
    dict
    """
    try:
        if request.method =="POST":
            data = request.get_json()
            elements=["cas","was","category"]
            for ele in elements:
                if ele not in data:
                    return jsonify({"Status" : 400,
                                    "message": "Bad Params"})

            cas = data['cas']
            was = data['was']
            category = data["category"]

            print("Before Validate", cas +"-" + str(was) + "-" + category)
            predicted = -1
            if validte_data(cas, was, category):
                cas_x,category_x = convert_prediction_ready(cas=cas,category=category)
                print("In Validate", str(cas_x) +"-" + str(was) + "-" + str(category_x))

                df_dict = {"CAS" : cas_x,
                    "Max_WF" : was,
                    "category" : category_x
                    }
                data_frame = pd.DataFrame(df_dict, index=[0])
                print("dataframe for pred",data_frame)
                predicted = predict.predict(data_frame)
                print("predicition", predicted)
                funct = convert_prediction_ready(func = predicted)
                print(funct)
                return jsonify({"Status": 200,
                         "message" : "Your function is " + funct        
                })
            else:
                return jsonify({"Success" : 400,
                                "message" : "Bad params"})

    except RuntimeError:
        print("Runtime error")
if __name__ == "__main__":
    app.run(port=3000, debug=True)
        