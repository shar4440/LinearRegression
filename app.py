import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


#app starts
app = Flask(__name__)

##Load the pickle model

regmodel = pickle.load(open('linear_regression.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
  data = request.json['data']
  print(data)
  print(np.array(list(data.values())).reshape(1,-1))

  new_transfered_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

  output = regmodel.predict(new_transfered_data)
  print(output[0])
  return jsonify(output[0])




@app.route('/predict_form',methods=['POST'])
def predict_form():
  #creating the list for each form inputs
  data = [float(x) for x in request.form.values()]
  #this final output will contain the scaled data or input  values
  final_input = scalar.transform(np.array(data).reshape(1,-1))

  print(final_input)
  output = regmodel.predict(final_input)
  return render_template("home.html",prediction_text="The House prediction price is {} ".format(output))






if __name__ == "__main__":
  app.run(debug=True)