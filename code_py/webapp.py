from flask import Flask, request, render_template, flash, redirect, url_for
from flask_wtf import FlaskForm

from wtforms import StringField, SelectField, DecimalField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange

import numpy as np
import pandas as pd
import json
from market_tool import MarketTool
from time import sleep
from pathlib import Path
from collections import OrderedDict

from bokeh_webapp import *
from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.resources import CDN
from jinja2 import Template


class ModelOptions(FlaskForm):
    option_wind = DecimalField("Wind Availability", validators=[DataRequired(), NumberRange(min=0, max=1)],
                               default=1)
    option_ex_de = IntegerField("Export: DK->DE", validators=[DataRequired(),  NumberRange(min=-10000, max=10000)],
                                default=1000)
    option_ex_se = IntegerField("Export: DK->SE", validators=[DataRequired(),  NumberRange(min=-10000, max=10000)],
                                default=-1000)
    option_ex_no = IntegerField("Export: DK->NO", validators=[DataRequired(),  NumberRange(min=-10000, max=10000)],
                                default=-1000)
    option_demand_dk1 = IntegerField("Demand DK1", validators=[DataRequired(),  NumberRange(min=0, max=50000)],
                                     default=1500)
    option_demand_dk2 = IntegerField("Demand DK1", validators=[DataRequired(),  NumberRange(min=0, max=50000)],
                                     default=2500)

    grid_config = SelectField('Grid Config.', choices=[('nodal', 'Nodal'), 
                                                       ('cbco_nodal', 'N-1'), 
                                                       ('dispatch', 'Dispatch')])
    submit = SubmitField('Run Model')

class DataOptions(FlaskForm):
    data_selector = SelectField('Select Data', choices=[('nodes', 'Nodes'), 
                                                        ('plants', 'Plants'), 
                                                        ('lines', 'Lines')], default="plants")
    submit_data = SubmitField('Look at Data')

app = Flask(__name__)
app.config["SECRET_KEY"] = '12309213099AFH819'

wdir = Path.cwd()

global mato
mato = MarketTool(options_file="profiles/dk.json", webapp=True)
mato.options["optimization"]["infeasibility"]["bound"] = 10000
mato.load_data('data_input\\dk_webapp.xlsx')

@app.route("/results", methods=['GET', 'POST'])
def return_results():
    global mato
    display_results = OrderedDict()
    if mato.data.results:
        df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=5e-2)
        df3, df4 = mato.data.results.overloaded_lines_n_0()

        slack_pos = mato.data.results.INFEAS_EL_N_POS.INFEAS_EL_N_POS.sum() 
        slack_neg = mato.data.results.INFEAS_EL_N_NEG.INFEAS_EL_N_NEG.sum()

        total_gen = mato.data.results.G.G.sum()
        total_demand = mato.data.demand_el.loc["t0001", :].sum()

        results = mato.data.result_attributes["objective"]
        display_results["Solve Status"] = results["Solve Status"]
        display_results["Objective Value"] = round(results["Objective Value"])
        display_results["Generation Cost"] = round(results["COST_G"] + results["COST_H"])
        display_results["Total Generation / Demand [MW]"] = f"{round(total_gen)} / {round(total_demand)}"
        display_results["Gen. Slack (+, -)"] = f"{round(slack_pos)} / {round(slack_neg)}" 
        display_results["Overloads (N-0)"] = df3["# of overloads"].sum()
        display_results["Overloads (N-1)"] = df1["# of overloads"].sum()

    # return render_template("results.html", results=display_results)
    return display_results

def run_model(options):
    global mato
    mato.load_data('data_input\\dk_webapp.xlsx')
    # flash("Running Model!", "warning")
    wind = options.option_wind.data
    export_de = options.option_ex_de.data
    export_se = options.option_ex_se.data
    export_no = options.option_ex_no.data

    demand_dk1 = options.option_demand_dk1.data
    demand_dk2 = options.option_demand_dk2.data

    grid_option = options.grid_config.data

    # Setting wind availability
    wind_plants = mato.data.plants.index[mato.data.plants.fuel == "wind"]
    mato.data.availability.loc[:, wind_plants] = wind

    # Setting Export
    de_nodes = mato.data.nodes.index[mato.data.nodes.zone == "DE"]
    mato.data.net_export[de_nodes] = -export_de/len(de_nodes)

    se_nodes = mato.data.nodes.index[mato.data.nodes.zone == "SE"]
    mato.data.net_export[se_nodes] = -export_se/len(se_nodes)

    no_nodes = mato.data.nodes.index[mato.data.nodes.zone == "NO"]
    mato.data.net_export[no_nodes] = -export_no/len(no_nodes)

    # setting demand
    dk1_nodes = mato.data.nodes.index[mato.data.nodes.zone == "DK1"]
    dk2_nodes = mato.data.nodes.index[mato.data.nodes.zone == "DK2"]

    demand_profile_dk1 = mato.data.demand_el.loc["t0001", dk1_nodes]
    demand_profile_dk1 *= 1/demand_profile_dk1.sum()
    demand_profile_dk2 = mato.data.demand_el.loc["t0001", dk2_nodes]
    demand_profile_dk2 *= 1/demand_profile_dk2.sum()

    mato.data.demand_el.loc["t0001", dk1_nodes] = demand_profile_dk1*demand_dk1
    mato.data.demand_el.loc["t0001", dk2_nodes] = demand_profile_dk2*demand_dk2

    mato.options["optimization"]["type"] = grid_option
    mato.create_grid_representation()
    mato.update_market_model_data()
    mato.run_market_model()


@app.route("/", methods=['GET', 'POST'])
@app.route("/main", methods=['GET', 'POST'])
def home():
    options = ModelOptions()
    # if options.validate_on_submit():
    if request.method == "POST":
        # print(request.form["submit_botton"])
        run_model(options)
    
    display_results = return_results()

    # if request.method == "GET":
    return render_template('main.html', form=options, results=display_results)

@app.route("/data", methods=['POST', 'GET'])
def data():
    global mato
    options = DataOptions()
    if mato:
        df = getattr(mato.data, options.data_selector.data)
    else: 
        print("Init Mato")
        df = pd.DataFrame(data=["model not initialized!!"])

    return render_template('data.html', 
                            tables=[df.to_html(classes='data', header="true")],
                            form=options)

@app.route('/plot')
def plot():
    global mato
    if mato:
        if mato.data.results:
            timestep = mato.data.result_attributes["model_horizon"][0]

            inj = mato.data.results.INJ
            inj = inj.INJ[inj.t == timestep].values

            flow_n_0 = mato.data.results.n_0_flow()
            flow_n_1 = mato.data.results.n_1_flow(sensitivity=5e-2)
               
            flow_n_0 = flow_n_0[timestep]
            flow_n_0 = flow_n_0.reindex(mato.data.lines.index)

            flow_n_1 = flow_n_1.drop("co", axis=1)
            flow_n_1[timestep] = flow_n_1[timestep].abs()
            flow_n_1 = flow_n_1.groupby("cb").max().reset_index()
            flow_n_1 = flow_n_1.set_index("cb").reindex(mato.data.lines.index)
            flow_n_1 = flow_n_1[timestep]
            flow_n_1 = flow_n_1.reindex(mato.data.lines.index)

            f_dc = mato.data.results.F_DC
            f_dc = f_dc[f_dc.t == timestep].drop("t", axis=1)
            f_dc.set_index("dc", inplace=True)
            f_dc = f_dc.reindex(mato.data.dclines.index)

        else:
            inj = np.array([0 for n in mato.data.nodes.index])

            flow_n_0 = pd.DataFrame(index=mato.data.lines.index)
            flow_n_0["t0001"] = 0
            flow_n_0 = flow_n_0["t0001"]

            flow_n_1 = pd.DataFrame(index=mato.data.lines.index)
            flow_n_1["t0001"] = 0
            flow_n_1 = flow_n_1["t0001"]

            f_dc = pd.DataFrame(index=mato.data.dclines.index)
            f_dc["t0001"] = 0
            f_dc = f_dc["t0001"]

        fig = create_plot(mato.data.lines, mato.data.nodes, mato.data.dclines, 
                          inj, flow_n_0, flow_n_1, f_dc)
        # print(json.dumps(json_item(fig, "myplot")))
        return json.dumps(json_item(fig, "grid_plot"))


@app.route('/stream')
def stream():
    def generate():
        with open(wdir.joinpath('logs\\market_tool_webapp.log')) as f:
            while True:
                log = f.read()
                yield "\n".join([line[:70] for line in log.split("\n")[-1-100:-1]])
                sleep(2)
    return app.response_class(generate(), mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True)
