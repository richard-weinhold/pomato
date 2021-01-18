# pylint: disable-msg=E1102

import json
import threading

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash_daq import BooleanSwitch
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash.dependencies import Input, Output
from flask import request

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.GRID, 'https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowY': 'scroll', 
        'padding': '20px'
    }
}

def display_hover_data(data):
    return json.dumps(data, indent=2)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def page_0():
    layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col(html.Button('Update Results', id='results-botton', n_clicks=0), style={"padding": "15px"}),
                dbc.Col(dcc.Dropdown(id='results-dropdown'), style={"padding": "15px"})
                ]),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown("""**Result summary**"""),
                    html.Pre(id='results-summary', style=styles['pre'])], width=6),
                dbc.Col([
                    dcc.Graph(id='installed-capacity-figure'),
                    dcc.Graph(id='generation-pie-chart')
                    ], width=6)
                ])
            ])                   
    return layout

def page_1():

    layout = dbc.Container(
        [
            dbc.Row(
                [    
                    dbc.Col(html.P("Click or select nodes to see plants and generation.", className="control_label"),
                            width={"size": 4, "offset": 0}),
                    dbc.Col(html.P(" ", className="control_label"),
                            width={"size": 4, "offset": 2}, style={"padding": "15px"}),
                ]),
                dbc.Row(
                [ 
                    dbc.Col(dcc.Graph(id='geo-figure-generation'), width={"size": 6}),
                    dbc.Col(dcc.Graph(id='generation-figure'), width={"size": 6})
                ]),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([dcc.Markdown("""**Hover Data**"""),
                                    html.Pre(id='hover-geo-generation')]), width=4),
                    dbc.Col(
                        html.Div([dcc.Markdown("""**Hover Plant Table**"""),
                            dash_table.DataTable(id='plant-table')]), width=4),
                    dbc.Col(
                        html.Div([
                            dcc.Markdown("""**Selection Data**"""),
                            html.Pre(id='selection-geo-generation')]), width=4)
                    ]),
            ], fluid=True)
                                    
    return layout

def page_2():
    layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col([html.Div(id='timestep-display'),
                             dcc.Slider(id='timestep-selector', min=0, step=1)],
                             width={"size": 5, "offset": 0.5}, style={"padding": "15px"}),
                    dbc.Col([dcc.Markdown("""**Lines** (use clicks or dropdown for selection) """),
                                dcc.Dropdown(id='line-selector', multi=True)],
                                            width={"size": 4, "offset": 1}, style={"padding": "15px"}),
                    ]),
            dbc.Row(
                [   
                    dbc.Col(
                        [
                            html.P("Line Colors:", className="control_label"),
                            dcc.RadioItems(id='flow-option',
                                options=[
                                    {'label': 'N-0 Flows', 'value': 0},
                                    {'label': 'N-1 Flows', 'value': 1},
                                    {'label': 'Voltage Levels', 'value': 2}],
                                value=0),
                            html.P("Click Mode:", className="control_label",
                                    style={"margin-top": "15px"}),
                            dcc.RadioItems(id='line-click-option',
                                options=[
                                    {'label': 'Add to Line Plot', 'value': 0},
                                    {'label': 'Show LODF', 'value': 1}],
                                value=0),
                            html.P("Show Prices:", className="control_label",
                                    style={"margin-top": "15px"}),
                            BooleanSwitch(id='toggle-prices', on=False),
                            html.P("Show Redispatch:", className="control_label",
                                    style={"margin-top": "15px"}),
                            BooleanSwitch(id='toggle-redispatch', on=False),
                            ], width={"size": 0.5}, style={"padding": "15px"}),
                    
                    dbc.Col(dcc.Graph(id='geo-figure-lines'), 
                            width={"size": 5}, style={"padding": "15px"}),
                    dbc.Col(dcc.Graph(id='lines-figure'), 
                            width={"size": 5}, style={"padding": "15px"})
                ], className="h-75"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([dcc.Markdown("""**Hover Data**"""),
                                    html.Pre(id='hover-geo-lines')]), width=4),
                    dbc.Col(
                        html.Div([dcc.Markdown("""**Click Data**"""),
                                    dash_table.DataTable(id='node-table')]), width=4),
                    dbc.Col(
                        html.Div([
                            dcc.Markdown("""**Selection Data**"""),
                            html.Pre(id='selection-geo-lines')]), width=4)
                ], className="h-25"),
            ], 
        fluid=True)
    return layout

class Dashboard():
    def __init__(self, pomato_instance):
        self.pomato_instance = pomato_instance
        self.app = None
        self.init_app()      
        self.dash_thread = threading.Thread(target=self.run, args=())
        # self.start()
        
    def start(self):
        self.dash_thread.start()
    
    def run(self):
        self.app.run_server(debug=True, use_reloader=False,
                            host='0.0.0.0')

    def join(self):
        """Close the locally hosted dash plot"""
        print("Teardown Dash Server.")
        if self.dash_thread.is_alive():
            print("Joining Dash Thread.")

            requests.post('http://127.0.0.1:8050/shutdown')
            self.dash_thread.join()
        self.dash_thread = self.dash_thread = threading.Thread(target=self.run, args=())
    
    def init_app(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        # self.app.layout = self.create_layout()
        self.app.layout = html.Div([
            dcc.Tabs([
                dcc.Tab(label='Overview', children=[page_0()]),
                dcc.Tab(label='Generation', children=[page_1()]),
                dcc.Tab(label='Transmission', children=[page_2()]),
                ])
            ])
        # Callbacks
        # Page 1: Summary 
        self.app.callback([dash.dependencies.Output('results-dropdown', 'options'),
                           dash.dependencies.Output('results-dropdown', 'value')],
                          [#dash.dependencies.Input('results-dropdown', 'value'),
                           dash.dependencies.Input('results-botton', 'n_clicks')])(self.update_result_selection)
                
        # Print Result Attributes 
        self.app.callback(dash.dependencies.Output('results-summary', 'children'),
                          [dash.dependencies.Input('results-dropdown', 'value')])(self.display_result_summary)

        # Generation Pie Chart 
        self.app.callback(dash.dependencies.Output('installed-capacity-figure', 'figure'),
            dash.dependencies.Input('results-dropdown', 'value'))(self.update_installed_capacity_figure)
        
        # Generation Pie Chart 
        self.app.callback(dash.dependencies.Output('generation-pie-chart', 'figure'),
            dash.dependencies.Input('results-dropdown', 'value'))(self.update_generation_pie_chart)
    
        # Update All components that have options based on the result
        self.app.callback([dash.dependencies.Output('timestep-selector', 'max'),
                           dash.dependencies.Output('timestep-selector', 'marks'),
                           dash.dependencies.Output('timestep-selector', 'value'),
                           dash.dependencies.Output('line-selector', 'options'),
                           dash.dependencies.Output('toggle-redispatch', 'disabled')],
                          [dash.dependencies.Input('results-dropdown', 'value')])(self.update_components_lines)
                
        # Page 2: Generation
        # Generation 
        self.app.callback(dash.dependencies.Output('generation-figure', 'figure'),
            [dash.dependencies.Input('results-dropdown', 'value'),
             dash.dependencies.Input('geo-figure-generation', 'selectedData')])(self.update_graph_generation)
        
        # Geoplot
        self.app.callback(dash.dependencies.Output('geo-figure-generation', 'figure'),
            [dash.dependencies.Input('results-dropdown', 'value')])(self.update_geo_plot)

        # Hover Geoplot
        self.app.callback(dash.dependencies.Output('hover-geo-generation', 'children'),
            [dash.dependencies.Input('geo-figure-generation', 'hoverData')])(display_hover_data)
        
        # Click Plant Table
        self.app.callback([dash.dependencies.Output('plant-table', 'columns'),
                           dash.dependencies.Output('plant-table', 'data')],
                          [dash.dependencies.Input('results-dropdown', 'value'),
                           dash.dependencies.Input('geo-figure-generation', 'clickData')])(self.display_plant_data)
                
        # Select Geoplot
        self.app.callback(dash.dependencies.Output('selection-geo-generation', 'children'),
            [dash.dependencies.Input('geo-figure-generation', 'selectedData')])(display_hover_data)
        

        ### Page 3: Lines 
        # Geoplot 
        self.app.callback(dash.dependencies.Output('geo-figure-lines', 'figure'),
            [dash.dependencies.Input('results-dropdown', 'value'),
             dash.dependencies.Input('toggle-redispatch', 'on'),
             dash.dependencies.Input('toggle-prices', 'on'),
             dash.dependencies.Input('flow-option', 'value'),
             dash.dependencies.Input('timestep-selector', 'value')])(self.update_geo_plot)
        
        # Click Lines Geoplot
        self.app.callback(dash.dependencies.Output('line-selector', 'value'),
            [dash.dependencies.Input('results-dropdown', 'value'),
              dash.dependencies.Input('geo-figure-lines', 'clickData'),
              dash.dependencies.State('line-selector', 'value')])(self.click_lines)
        
        # Lineflow plot
        self.app.callback(dash.dependencies.Output('lines-figure', 'figure'),
            [dash.dependencies.Input('results-dropdown', 'value'),
             dash.dependencies.Input('line-selector', 'value')])(self.update_graph_lines)

        # Timestep display
        self.app.callback(dash.dependencies.Output('timestep-display', 'children'),
            [dash.dependencies.Input('results-dropdown', 'value'),
             dash.dependencies.Input('timestep-selector', 'value')])(self.display_timestep)

        # Hover Geoplot 
        self.app.callback(dash.dependencies.Output('hover-geo-lines', 'children'),
            [dash.dependencies.Input('geo-figure-lines', 'hoverData')])(display_hover_data)
    
        self.app.callback([dash.dependencies.Output('node-table', 'columns'),
                           dash.dependencies.Output('node-table', 'data')],
                          [dash.dependencies.Input('results-dropdown', 'value'),
                           dash.dependencies.Input('geo-figure-lines', 'clickData')])(self.display_node_data)
        # Select Geoplot
        self.app.callback(dash.dependencies.Output('selection-geo-lines', 'children'),
            [dash.dependencies.Input('geo-figure-lines', 'selectedData')])(display_hover_data)

        self.app.server.route('/shutdown', methods=['POST'])(shutdown)
                
    def update_components_lines(self, result_name):
        result = self.pomato_instance.data.results[result_name]
        slider_max = len(result.model_horizon)
        number_labels = 10
        slider_steps = int(len(result.model_horizon)/number_labels)
        marks = {x : result.model_horizon[x] for x in range(0, slider_max) if (x%slider_steps == 0)}
        options = [{"label": x, "value": x} for x in result.data.lines.index]
        
        disable_redispatch_toggle = not (result.result_attributes["is_redispatch_result"] and 
                                         isinstance(result.result_attributes["corresponding_market_result_name"], str))
        
        return slider_max, marks, 0, options, disable_redispatch_toggle
        
    def display_plant_data(self, result_name, click_data):
        result = self.pomato_instance.data.results[result_name]
        if click_data:
            nodes = []
            for point in click_data["points"]:
                if point["customdata"][0] in result.data.nodes.index:
                    nodes.append(point["customdata"][0])
        else: 
            nodes = []
        plants = result.data.plants[result.data.plants.node.isin(nodes)]
        columns = ["index", "g_max", "mc_el", "plant_type", "fuel"]
        return ([{"name": i, "id": i} for i in columns], 
                plants.reset_index()[columns].to_dict("records"))
    
    def display_node_data(self, result_name, click_data):
        result = self.pomato_instance.data.results[result_name]
        if click_data and click_data["points"][0]["customdata"][0] in result.data.nodes.index:
            node = click_data["points"][0]["customdata"][0]
            node_data = result.data.nodes.loc[node].reset_index()
            node_data.columns = ["", node]
        else: 
            node_data = pd.DataFrame(columns= ["", "node"])
        
        return ([{"name": i, "id": i} for i in node_data.columns], 
                node_data.to_dict("records"))
                
    def display_result_summary(self, result_name):
        result = self.pomato_instance.data.results[result_name]
        result_info = dict(result.result_attributes)
        del result_info["model_horizon"]
        del result_info["variables"]
        del result_info["dual_variables"]
        del result_info["infeasibility_variables"]
        return json.dumps(result_info, indent=2)
        
    def display_timestep(self, result_name, timestep):
        result = self.pomato_instance.data.results[result_name]
        return "Select timestep with slider - currently selected: " + result.model_horizon[timestep]
        
    def update_result_selection(self, n_clicks):
        options = []
        for result in self.pomato_instance.data.results:
            value = result
            label = self.pomato_instance.data.results[result].result_attributes["title"]
            options.append({"label": label, "value": value})
            
        value = next(r for r in self.pomato_instance.data.results)
        return options, value
    
    def click_lines(self, result_name, click_data, current_value):
        result = self.pomato_instance.data.results[result_name]
        if not isinstance(current_value, list):
            current_value = []
        lines = []
        if click_data:
            for point in click_data["points"]:
                if point["customdata"][0] in result.data.lines.index:
                    lines.append(point["customdata"][0])
        return current_value + lines
    
    def update_installed_capacity_figure(self, result_name):
        result = self.pomato_instance.data.results[result_name]
        fig = self.pomato_instance.visualization.create_installed_capacity_plot(result, show_plot=False)
        fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
        return fig
    
    def update_generation_pie_chart(self, result_name):
        result = self.pomato_instance.data.results[result_name]
        fig = self.pomato_instance.visualization.create_generation_overview(result, show_plot=False)
        fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
        return fig

    def update_graph_lines(self, result_name, lines):
        result = self.pomato_instance.data.results[result_name]
        if not isinstance(lines, list):
            lines = [result.data.lines.index[0]]
        fig =  self.pomato_instance.visualization.create_lineflow_plot(result, lines, show_plot=False)
        fig.update_layout(margin={"r":0,"t":25,"l":0,"b":0})
        return fig
    
    def update_graph_generation(self, result_name, selection_data):
        result = self.pomato_instance.data.results[result_name]
        if selection_data:
            nodes = []
            for point in selection_data["points"]:
                if point["customdata"][0] in result.data.nodes.index:
                    nodes.append(point["customdata"][0])
        else: 
            nodes = None
        fig = self.pomato_instance.visualization.create_generation_plot(result, nodes=nodes, show_plot=False)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
    def update_geo_plot(self, result_name, show_redispatch=False, show_prices=False, line_color_option=0, 
                        timestep=None):
        result = self.pomato_instance.data.results[result_name]
        vis = self.pomato_instance.visualization
        fig =  vis.create_geo_plot(result, 
                                   show_redispatch=show_redispatch, 
                                   show_prices=show_prices,
                                   line_color_option=line_color_option,
                                   timestep=timestep, 
                                   show_plot=False)
        fig.update_layout(uirevision = True)
        return fig



