import joblib
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open(r"C:\Users\RUSHMITHA\Downloads\MENTEL HEALTH PREDICTION\model.pkl","rb"))
ct = joblib.load('feature_values')

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/home1', methods=["POST"])
def home1():
    return render_template("home.html")
@app.route('/pred1', methods=["POST"])
def pred1():
    return render_template("index.html")
@app.route('/pred', methods=["POST","GET"])
def predict():
    return render_template("index.html")
@app.route('/out',methods=["POST","GET"])
def output():
    Age = request.form["Age"]
    Gender = request.form["Gender"]
    self_employed = request.form["self_employed"]
    family_history = request.form["family_history"]
    work_interfere = request.form["work_interfere"]
    no_employees = request.form["no_employees"]
    remote_work = request.form["remote_work"]
    tech_company = request.form["tech_company"]
    benefits = request.form["benefits"]
    care_options = request.form["care_options"]
    wellness_program = request.form["wellness_program"]
    seek_help = request.form["seek_help"]
    anonymity = request.form["anonymity"]
    leave = request.form["leave"]
    mental_health_consequence = request.form["mental_health_consequence"]
    phys_health_consequence = request.form["phys_health_consequence"]
    coworkers = request.form["coworkers"]
    supervisor = request.form["supervisor"]
    mental_health_interview = request.form["mental_health_interview"]
    phys_health_interview = request.form["phys_health_interview"]
    mental_vs_physical = request.form["mental_vs_physical"]
    obs_consequence = request.form["obs_consequence"]


    data = {"Age":Age,"Gender":Gender,"self_employed":self_employed,"family_history":family_history,"work_interfere":work_interfere,"no_employees":no_employees,
             "remote_work":remote_work,"tech_company":tech_company,"benefits":benefits,"care_options":care_options,"wellness_program":wellness_program,
             "seek_help":seek_help,"anonymity":anonymity,"leave":leave,"mental_health_consequence":mental_health_consequence,"phys_health_consequence":phys_health_consequence,
            "coworkers":coworkers,"supervisor":supervisor,
            "mental_health_interview":mental_health_interview,"phys_health_interview":phys_health_interview,"mental_vs_physical":mental_vs_physical,"obs_consequence":obs_consequence}
    data=pd.DataFrame.from_dict([data])
    """
    f = ["Age","Gender","self_employed","family_history","work_interfere","no_employees",
         "remote_work","tech_company","benefits","care_options","wellness_program",
         "seek_help","anonymity","leave","mental_health_consequence","phys_health_consequence","coworkers","supervisor",
                    "mental_health_interview","phys_health_interview","mental_vs_physical","obs_consequence"]
    data=pd.DataFrame(data, columns=f)
    """
    print(data)

    pred = model.predict(data)
    print(pred)



    if pred==0:
        return render_template("output1.html",y="This person requires mental health treatment")
    else:
        return render_template("output.html",y="This person doesn't require mental health treatment")

if __name__ == '__main__':
    app.run(debug = True)



