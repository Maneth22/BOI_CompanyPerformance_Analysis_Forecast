import re
from datetime import datetime
from flask import Flask, render_template, redirect, jsonify, request, url_for, flash, send_file
import pymssql  
import pandas as pd
import mysql.connector
from Arima_exp import Arima_exp
from getData import getDataByCompanyRef
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/forcast")
def forcast():

    return render_template('forcasts.html', data_present =False, NP_data = None, CR_data = None, QR_data = None,AT_data =  None,DA_data = None)

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/ref_Input", methods =['GET', 'POST'])
def ref_Input():
    if request.method == 'POST':
        reference_no = request.form['ref']
        print(reference_no)

        my_file = open("ARIMA_model\\companies.txt", "r")

        # reading the file
        data = my_file.read()

        # replacing end of line('/n') with ' ' and
        # splitting the text it further when '.' is seen
        data_into_list = data.split("\n")
        for i in data_into_list:
            if i == reference_no:
                ArimaClass = Arima_exp()
                net_profit_plot, cur_rate_plot, quick_ratio_plot, AT_plot, debt_Asset_plot = ArimaClass.givePredictions(reference_no)
                return render_template("forcasts.html", NP_data = net_profit_plot, CR_data = cur_rate_plot, QR_data = quick_ratio_plot,AT_data =  AT_plot,DA_data = debt_Asset_plot)
            
        return render_template('forcasts.html', data_present =False, NP_data = None, CR_data = None, QR_data = None,AT_data =  None,DA_data = None)

    else:
        return render_template('forcasts.html', data_present =False, NP_data = None, CR_data = None, QR_data = None,AT_data =  None,DA_data = None)


@app.route("/existing_data")
def existing_data():
    return render_template("records.html", data = None)

@app.route("/displayTables", methods =['GET' , 'POST'])
def displayTables():
    if request.method == "POST":
        reference_no = request.form['ref']
        dataframe = getDataByCompanyRef(reference_no)
        print(dataframe)
        return render_template("records.html", data=dataframe.to_html(), ref = reference_no)

    return render_template("records.html")


if __name__ == '__main__':
   app.run(debug = True)

