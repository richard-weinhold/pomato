# pylint: disable-msg=E1102

import json
import threading

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
import numpy as np
import pandas as pd
import itertools
import requests
from dash.dependencies import Input, Output, State
from flask import request
from pomato.tools import add_default_values_to_dict
from pomato.visualization import FBDomainPlots
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.GRID, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
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

def basic_layout(naming_suffix, multi=False):
    return dbc.Row(
        [
            dbc.Col(html.Button('Update Results', id=f'results-botton-{naming_suffix}', n_clicks=0), style={"padding": "15px"}),
            dbc.Col(dcc.Dropdown(id=f'results-dropdown-{naming_suffix}', multi=multi), style={"padding": "15px"})
        ])

def page_overview():
    layout = dbc.Container(
        [
            basic_layout("overview", multi=True),
            dbc.Row([
                dbc.Col([dcc.Graph(id='installed-capacity-figure', style={"height": "100%"})], width=3),
                dbc.Col([dcc.Graph(id='generation-overview-figure', style={"height": "100%"})], width=6),
                dbc.Col([dcc.Graph(id='cost-overview-figure', style={"height": "100%"})], width=3),
                ], className="h-75")
            ], style={"height": "100vh"}, fluid=True)                   
    return layout

def page_generation():

    layout = dbc.Container(
        [
            basic_layout("generation"),
            dbc.Row(
                [   dbc.Col(html.P("Click or select nodes to see plants and generation.", className="control_label"),
                            width={"size": 2, "offset": 0}, style={"padding": "15px"}),
                    dbc.Col(dcc.Dropdown(id="generation-node-dropdown", multi=True),
                            width={"size": 2, "offset": 0}, style={"padding": "15px"}),
                    ]),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Form(
                                    [
                                        html.P("Display Options:", className="control_label"),
                                        html.Div(id='lineloading-display-generation',
                                                style={"font-size": "small", "margin-top": "10px"}),
                                        dbc.Input(
                                            id="input-lineloading-generation", type="number", 
                                            placeholder="Lineloading", value=0, debounce=True,
                                            min=0, max=100, step=1),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Show Prices:", "value": 1},
                                                {"label": "Show Redispatch", "value": 2, "disabled": True},
                                                {"label": "Show Curtailment", "value": 3}],
                                            value=[],
                                            id="switches-generation",
                                            switch=True,
                                            style={"margin-top": "10px"}),
                                    ])
                                )
                            ), width={"size": 2}, style={"padding": "15px"}
                        ),
                    dbc.Col(dcc.Graph(id='geo-figure-generation', style={"height": "100%"}), width={"size": 5}),
                    dbc.Col(dcc.Graph(id='generation-figure', style={"height": "100%"}), width={"size": 5})
                ],  className="h-50"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([html.P("Click nodes to see plants"),
                        dash_table.DataTable(id='plant-table')]), width=4),
                ]),
            ], style={"height": "100vh"}, fluid=True)
                                    
    return layout

def page_transmission():
    layout = dbc.Container(
        [
            basic_layout("transmission"),
            dbc.Row(
                [
                    dbc.Col([html.Div(id='timestep-display-transmission'),
                             dcc.Slider(id='timestep-selector-transmission', min=0, step=1)],
                             width={"size": 4, "offset": 2}, style={"padding": "15px"}),
                    dbc.Col([dcc.Markdown("""**Lines** (use clicks or dropdown for selection) """),
                                dcc.Dropdown(id='line-selector', multi=True, persistence=True, persistence_type="local")],
                                            width={"size": 5, "offset": 1}, style={"padding": "15px"}),
                    ]),
            dbc.Row(
                [   dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Form(
                                    [
                                        html.P("Display Options:", className="control_label"),
                                        dbc.RadioItems(id='flow-option',
                                            options=[
                                                {'label': 'N-0 Flows', 'value': 0},
                                                {'label': 'N-1 Flows', 'value': 1},
                                                {'label': 'Voltage Levels', 'value': 2}],
                                            value=0),
                                        html.Div(id='lineloading-display-transmission',
                                                style={"font-size": "small", "margin-top": "10px"}),
                                        dbc.Input(
                                            id="input-lineloading-transmission", type="number", 
                                            placeholder="Lineloading", value=0, debounce=True,
                                            min=0, max=100, step=1),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Show Prices:", "value": 1},
                                                {"label": "Show Redispatch", "value": 2, "disabled": True},
                                                {"label": "Show Curtailment:", "value": 1}],
                                            value=[],
                                            id="switches-transmission",
                                            switch=True,
                                            style={"margin-top": "10px"}),
                                    ]))), width={"size": 2}, style={"padding": "15px"}),
                    dbc.Col(dcc.Graph(id='geo-figure-lines', style={"height": "100%"}), 
                            width={"size": 5}, style={"padding": "15px"}),
                    dbc.Col(dcc.Graph(id='lines-figure', style={"height": "100%"}), 
                            width={"size": 5}, style={"padding": "15px"})
                ], className="h-50"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([dcc.Markdown("""**Click Data**"""),
                                    dash_table.DataTable(id='node-table')]), width=4),
                ]),
        ], style={"height": "100vh"}, fluid=True)
    return layout

def page_fbmc():
    layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Button('Update Results', id='results-botton-fbmc', n_clicks=0),
                    ], width={"size": 2}, style={"padding": "15px"}),
                dbc.Col([
                    html.Div("Select Basecase:"),
                    dcc.Dropdown(id='results-dropdown-fbmc')
                    ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("Select GSK"),
                    dcc.Dropdown(id='gsk-dropdown',
                    options=[
                        {'label': 'gmax', 'value': "gmax"},
                        {'label': 'dynamic', 'value': "dynamic"},
                        {'label': 'flat', 'value': "flat"}],
                        value="gmax"),
                    ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("minRAM [%]:"),
                    dbc.Input(
                        id="input-minram", type="number", 
                        value=40, min=0, max=100, step=1),
                    ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("FRM/FAV [%]:"),
                    dbc.Input(
                        id="input-frm", type="number", 
                        value=0, min=0, max=100, step=1),
                    ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("CNE sensitivity [%]:"),
                    dbc.Input(
                        id="input-cne-sensitivity", type="number", 
                        value=5, min=0, max=100, step=0.1),
                    ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("Contingency sensitivity [%]:"),
                    dbc.Input(
                        id="input-cnec-sensitivity", type="number", 
                        value=20, min=0, max=100, step=0.1),
                    ], style={"padding": "15px"}),
                ], className="h-10"),
            dbc.Row([
                dbc.Col([
                    html.Div(id='timestep-display-fbmc-market'),
                    dcc.Slider(id='timestep-selector-fbmc-market', min=0, step=1, persistence=True),
                ],  width={"size": 4}, style={"padding": "15px"}),
                dbc.Col([
                    html.Div("Select Domain X:"),
                    dcc.Dropdown(id='domain-x-dropdown')
                ], style={"padding": "15px"}),
                dbc.Col([
                    html.Div("Select Domain Y"),
                    dcc.Dropdown(id='domain-y-dropdown'),
                ], style={"padding": "15px"}),
                dbc.Col([
                    html.Br(),
                    html.Button('Calculate FB Domain', id='botton-fb-domains', n_clicks=0)
                ], style={"padding": "15px"}),
                ], className="h-10"),
            dbc.Row([
                dbc.Col([
                    html.Div("Select Market Result:"),
                    dcc.Dropdown(id='results-dropdown-fbmc-market'),
                    ], style={"padding": "15px"}, width={"size": 3}),
                dbc.Col([
                    html.Div("Correct FB Domain for NEX in Market Result:"),
                    daq.BooleanSwitch(id='switch-fb-domain-nex-correction', on=True),
                    ], style={"padding": "15px"}, width={"size": 3}),
                dbc.Col([
                    html.Div("Enforce FB Domain to include NTCs with value:"),
                    daq.BooleanSwitch(id='switch-fb-domain-ntc', on=False),
                    ], style={"padding": "15px"}, width={"size": 3}),
                dbc.Col([
                    html.Br(),
                    dbc.Input(id="input-fb-domain-ntc", type="number", value=500), 
                    ], style={"padding": "15px"}, width={"size": 1}),                                        
                ], className="h-10"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='fb-geo-figure', style={"height": "100%"}), 
                        width={"size": 6}, style={"padding": "15px"}),
                dbc.Col(dcc.Graph(id='fb-domain-figure', style={"height": "100%"}), 
                        width={"size": 6}, style={"padding": "15px"}),
                ], className="h-50")
        ], style={"height": "100vh"}, fluid=True)
    return layout

class Dashboard():
    """Dashboard of Pomato"""
    def __init__(self, pomato_instance, **kwargs):

        self.pomato_instance = pomato_instance
        for result in self.pomato_instance.data.results:
            self.pomato_instance.data.results[result].create_result_data()
        self.app = None
        self.init_app()      
        self.dash_thread = threading.Thread(target=self.run, kwargs=kwargs)
        self._fb_domain = None
        
    def start(self):
        self.dash_thread.start()
    
    def run(self,  **kwargs):
        default_options = {"debug": True, 
                           "use_reloader": False,
                           "port": "8050", 
                           "host": "127.0.0.1"}

        server_args = add_default_values_to_dict(kwargs, default_options)
        self.app.run_server(**server_args)

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
        # self.app.config['suppress_callback_exceptions'] = True
        self.app.layout = html.Div([
            dcc.Location(id='url'),
            dcc.Store(id='viewport-container'),
            dcc.Tabs(id='pomato-tabs', children=[
                dcc.Tab(label='Overview', children=[page_overview()]),
                dcc.Tab(label='Generation', children=[page_generation()]),
                dcc.Tab(label='Transmission', children=[page_transmission()]),
                dcc.Tab(label='FBMC', children=[page_fbmc()]),
            ]),
        ])
        self.app.server.route('/shutdown', methods=['POST'])(shutdown)
        
        self.app.clientside_callback(
            """
            function(href) {
                var w = window.innerWidth;
                var h = window.innerHeight;
                return {'height': h, 'width': w};
            }
            """,
            Output('viewport-container', 'data'),
            Input('url', 'href')
        )

        for page in ["overview", "generation", "transmission", "fbmc"]:
            self.app.callback(
                [Output(f'results-dropdown-{page}', 'options'),
                 Output(f'results-dropdown-{page}', 'value')],
                Input(f'results-botton-{page}', 'n_clicks'))(self.update_result_selection)
        
        self.app.callback(
            [Output('results-dropdown-fbmc-market', 'options'),
            Output('results-dropdown-fbmc-market', 'value')],
            Input('results-botton-fbmc', 'n_clicks'))(self.update_result_selection)
        
        for page in ["transmission", "fbmc-market"]:
            self.app.callback([Output(f'timestep-selector-{page}', 'max'),
                               Output(f'timestep-selector-{page}', 'marks'),
                               Output(f'timestep-selector-{page}', 'value')],
                              [Input(f"results-dropdown-{page}", "value"),
                               Input("viewport-container", "data")],
                               State(f'timestep-selector-{page}', 'value'))(self.update_timestep_slider)

            self.app.callback(Output(f'timestep-display-{page}', 'children'),
                             [Input(f'results-dropdown-{page}', 'value'),
                              Input(f'timestep-selector-{page}', 'value')])(self.display_timestep)

        self.add_overview_callbacks()
        self.add_generation_callbacks()
        self.add_tranmission_callbacks()
        self.add_flowbased_callbacks()
        
    def add_flowbased_callbacks(self):
        # FB Parameters
        self.app.callback([Output('domain-x-dropdown', 'options'),
                           Output('domain-y-dropdown', 'options')],
                           Input("results-dropdown-fbmc", "value"))(self.update_domain_dropdown)                         
        
        self.app.callback(Output("fb-domain-figure", "figure"),
                          Input("botton-fb-domains", "n_clicks"),
                          [State("results-dropdown-fbmc", "value"),
                           State("gsk-dropdown", "value"),
                           State("input-minram", "value"),
                           State("input-frm", "value"),
                           State("input-cne-sensitivity", "value"),
                           State("input-cnec-sensitivity", "value"),
                           State('results-dropdown-fbmc-market', 'value'),
                           State('timestep-selector-fbmc-market', 'value'),
                           State('domain-x-dropdown', 'value'),
                           State('domain-y-dropdown', 'value'),
                           State("switch-fb-domain-nex-correction", "on"),
                           State("switch-fb-domain-ntc", "on"),
                           State("input-fb-domain-ntc", "value"),
                           ])(self.update_domain_plot)
        
        self.app.callback(Output("fb-geo-figure", "figure"),
                          [Input("botton-fb-domains", "n_clicks"),
                           Input("fb-domain-figure", "clickData")],
                          [State('timestep-selector-fbmc-market', 'value'),
                           State('results-dropdown-fbmc-market', 'value')])(self.update_fb_geo_plot)
        
    def add_overview_callbacks(self):
        # Page 1: Summary 
        # Print Result Attributes 
        self.app.callback(Output('installed-capacity-figure', 'figure'),
                          Input('results-dropdown-overview', 'value'))(self.update_installed_capacity_figure)

        # Overview Generation
        self.app.callback(Output('generation-overview-figure', 'figure'),
                          Input('results-dropdown-overview', 'value'))(self.update_generation_overview)
        
        # Overview Cost
        self.app.callback(Output('cost-overview-figure', 'figure'),
                          Input('results-dropdown-overview', 'value'))(self.update_cost_overview)

    def add_generation_callbacks(self):
        # Page 2: Generation
        # Update All components that have options based on the result
        self.app.callback(Output('switches-generation', 'options'),
                          Output("generation-node-dropdown", 'options'),
                          Input('results-dropdown-generation', 'value'))(self.update_components_generation)

        # Generation 
        self.app.callback(Output('generation-figure', 'figure'),
                         [Input('results-dropdown-generation', 'value'),
                          Input('geo-figure-generation', 'selectedData')])(self.update_graph_generation)
        
        # Geoplot
        self.app.callback(Output('geo-figure-generation', 'figure'),
                          [Input('results-dropdown-generation', 'value'),
                           Input('switches-generation', 'value'),
                           Input('input-lineloading-generation', 'value'),
                           Input('generation-node-dropdown', 'value')])(self.update_generation_geo_plot)
        # Lineloading display
        self.app.callback(Output('lineloading-display-generation', 'children'),
                          Input('input-lineloading-generation', 'value'))(self.display_lineloading)
        # Click Plant Table
        self.app.callback([Output('plant-table', 'columns'),
                           Output('plant-table', 'data')],
                          [Input('results-dropdown-generation', 'value'),
                           Input('geo-figure-generation', 'clickData')])(self.display_plant_data)

    def add_tranmission_callbacks(self):
        ### Page 3: Lines 
        # Update All components that have options based on the result
        self.app.callback([Output('line-selector', 'options'),
                           Output('switches-transmission', 'options')],
                           Input('results-dropdown-transmission', 'value'))(self.update_components_transmission)

        # Geoplot 
        self.app.callback(Output('geo-figure-lines', 'figure'),
                          [Input('results-dropdown-transmission', 'value'),
                           Input('switches-transmission', 'value'),
                           Input('flow-option', 'value'),
                           Input('timestep-selector-transmission', 'value'),
                           Input('input-lineloading-transmission', 'value'),
                           Input('line-selector', 'value')])(self.update_transmission_geo_plot)
        
        # Click Lines Geoplot
        self.app.callback(Output('line-selector', 'value'),
                          [Input('results-dropdown-transmission', 'value'),
                           Input('geo-figure-lines', 'clickData')],
                           State('line-selector', 'value'))(self.click_lines)
        
        # Lineflow plot
        self.app.callback(Output('lines-figure', 'figure'),
                          [Input('results-dropdown-transmission', 'value'),
                           Input('line-selector', 'value')])(self.update_graph_lines)

        # Lineloading display
        self.app.callback(Output('lineloading-display-transmission', 'children'),
                          Input('input-lineloading-transmission', 'value'))(self.display_lineloading)
        # Display Node data    
        self.app.callback([Output('node-table', 'columns'),
                           Output('node-table', 'data')],
                          [Input('results-dropdown-transmission', 'value'),
                           Input('geo-figure-lines', 'clickData')])(self.display_node_data)

    def update_fb_geo_plot(self, botton, click_data, timestep, result_name):
        if not result_name:
            return {}
        else:
            if click_data:
                if click_data:
                    lines = click_data["points"][0]["customdata"]
            else:
                lines = None
            result = self.pomato_instance.data.results[result_name]
            vis = self.pomato_instance.visualization
            fig = vis.create_zonal_geoplot(result, timestep=timestep, highlight_lines=lines, show_plot=False)
            fig.update_layout(uirevision = True)
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            return fig

    def update_domain_plot(self, click, basecase, gsk, minram, frm, cne_sensitivity, cnec_sensitivity,
                           result_name, timestep, domain_x, domain_y, correct_for_nex, include_ntc, ntc_value):
        if not basecase or (domain_x == domain_y) or not result_name: 
            return {}
        basecase = self.pomato_instance.data.results[basecase]
        result = self.pomato_instance.data.results[result_name]
        self.pomato_instance.data.ntc = self.pomato_instance.grid_model.create_ntc(ntc_value)
        if isinstance(timestep, int):
            timestep = basecase.model_horizon[timestep]
        self.pomato_instance.options["grid"]["minram"] = minram/100
        self.pomato_instance.options["grid"]["capacity_multiplier"] = 1 - frm/100
        fb_parameter_function = self.pomato_instance.fbmc.create_flowbased_parameters
        fb_parameters = fb_parameter_function(basecase, gsk, timesteps=[timestep],
                                              cne_sensitivity=cne_sensitivity/100, 
                                              lodf_sensitivity=cnec_sensitivity/100,
                                              enforce_ntc_domain=include_ntc)
        fb_domain = FBDomainPlots(self.pomato_instance.data, fb_parameters)
        domain_x, domain_y = domain_x.split("-"), domain_y.split("-")
        if correct_for_nex:
            domain_plot = fb_domain.generate_flowbased_domain(domain_x, domain_y, timestep=timestep, 
                                                              commercial_exchange=result.EX)
        else: 
            domain_plot = fb_domain.generate_flowbased_domain(domain_x, domain_y, timestep=timestep)
       
        fig = self.pomato_instance.visualization.create_fb_domain_plot(domain_plot, show_plot=False)
    
        commercial_exchange = result.EX[result.EX.t == timestep].copy()
        commercial_exchange.set_index(["z", "zz"], inplace=True)
        nex_x = commercial_exchange.loc[tuple(domain_x), "EX"] - commercial_exchange.loc[tuple(domain_x[::-1]), "EX"]
        nex_y = commercial_exchange.loc[tuple(domain_y), "EX"] - commercial_exchange.loc[tuple(domain_y[::-1]), "EX"]
        fig.add_trace(go.Scatter(x=[nex_x], y=[nex_y], mode="markers", name="Market Outcome"))

        if include_ntc:
            ntc_domain = [(ntc_value, ntc_value), (-ntc_value, ntc_value), 
                          (-ntc_value, -ntc_value), (ntc_value, -ntc_value), 
                          (ntc_value, ntc_value)]
            fig.add_trace(go.Scatter(x=np.asarray(ntc_domain)[:, 0], 
                                    y=np.asarray(ntc_domain)[:, 1],
                                    line=dict(color='blue', width=1, dash='dash'),
                                    name="NTC-Domain"))

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    def update_domain_dropdown(self, basecase):
        if not basecase:
            raise PreventUpdate
        else:
            basecase = self.pomato_instance.data.results[basecase]
            fb_region = self.pomato_instance.options["grid"]["flowbased_region"]
            if not fb_region:
                fb_region = list(self.pomato_instance.data.zones.index)
            option_domain = [{"label": "-".join(x), "value": "-".join(x)} for x in itertools.combinations(fb_region, 2)]
            return option_domain, option_domain

    def update_timestep_slider(self, result_name, size, value):
        if not result_name:
            raise PreventUpdate
        if not value:
            value = 0
        result = self.pomato_instance.data.results[result_name]
        slider_max = len(result.model_horizon)
        number_labels = int(size["width"] * (4/12) / 50)
        slider_steps = int(len(result.model_horizon)/number_labels)
        marks = {x : result.model_horizon[x] for x in range(0, slider_max) if (x%slider_steps == 0)}
        return slider_max, marks, value

    def update_components_transmission(self, result_name):
        result = self.pomato_instance.data.results[result_name]
        options_lineflow = [{"label": x, "value": x} for x in result.data.lines.index]
        enable_redispatch = not (result.result_attributes["is_redispatch_result"] and 
                                         isinstance(result.result_attributes["corresponding_market_result_name"], str))
        switches = [
            {"label": "Show Prices:", "value": 1},
            {"label": "Show Redispatch", "value": 2, "disabled": enable_redispatch},
            {"label": "Show Curtailment:", "value": 3}]
        return options_lineflow, switches

    def update_components_generation(self, result_name):
        result = self.pomato_instance.data.results[result_name]      
        enable_redispatch = not (result.result_attributes["is_redispatch_result"] and 
                                         isinstance(result.result_attributes["corresponding_market_result_name"], str))

        switches = [
            {"label": "Show Prices:", "value": 1},
            {"label": "Show Redispatch", "value": 2, "disabled": enable_redispatch},
            {"label": "Show Curtailment:", "value": 3}]
        nodes_options = [{"label": node, "value": node} for node in result.data.nodes.index]
        return switches, nodes_options
        
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
                
    def display_timestep(self, result_name, timestep):
        result = self.pomato_instance.data.results[result_name]
        return "Select timestep with slider - currently selected: " + result.model_horizon[timestep]
    
    def display_lineloading(self, lineloading):
        return "Line loading >" + str(lineloading) + "% ."
        
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
    
    def update_installed_capacity_figure(self, result_names):
        if not result_names:
            raise PreventUpdate

        if not isinstance(result_names, list):
            result_names = [result_names]
        result = self.pomato_instance.data.results[result_names[0]]
        fig = self.pomato_instance.visualization.create_installed_capacity_plot(result.data, show_plot=False)
        fig.update_layout(margin={"r":10,"t":10,"l":10,"b":10})
        return fig
    
    def update_generation_overview(self, result_names):
        if not result_names:
            raise PreventUpdate
        if not isinstance(result_names, list):
            result_names = [result_names]
        results = [self.pomato_instance.data.results[r] for r in result_names]
        fig = self.pomato_instance.visualization.create_generation_overview(results, show_plot=False)
        fig.update_layout(margin={"r":10,"t":110,"l":10,"b":10})
        return fig

    def update_cost_overview(self, result_names):
        if not result_names:
            raise PreventUpdate
        if not isinstance(result_names, list):
            result_names = [result_names]
        results = [self.pomato_instance.data.results[r] for r in result_names]
        fig = self.pomato_instance.visualization.create_cost_overview(results, show_plot=False)
        fig.update_layout(margin={"r":10,"t":10,"l":10,"b":10})
        return fig

    def update_graph_lines(self, result_name, lines):
        result = self.pomato_instance.data.results[result_name]
        if not isinstance(lines, list):
            lines = [result.data.lines.index[0]]
        fig =  self.pomato_instance.visualization.create_lineflow_plot(result, lines, show_plot=False)
        # fig.update_layout(margin={"r":0,"t":25,"l":0,"b":25})
        return fig
    
    def update_graph_generation(self, result_name, selection_data):
        result = self.pomato_instance.data.results[result_name]
        if selection_data:
            nodes = []
            for point in selection_data["points"]:
                if point["customdata"][0] in result.data.nodes.index:
                    nodes.append(point["customdata"][0])
            nodes = list(set(nodes))
        else: 
            nodes = None
        fig = self.pomato_instance.visualization.create_generation_plot(result, nodes=nodes, show_plot=False)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
    def update_transmission_geo_plot(self, result_name, switches, 
                                     line_color_option, timestep, threshold, lines):
        show_prices = True if 1 in switches else False
        show_redispatch = True  if 2 in switches else False
        show_curtailment = True  if 3 in switches else False
        result = self.pomato_instance.data.results[result_name]
        vis = self.pomato_instance.visualization
        if len(lines) == 0:
            lines = None
        fig =  vis.create_geo_plot(result, 
                                   show_redispatch=show_redispatch, 
                                   show_prices=show_prices,
                                   show_curtailment=show_curtailment,
                                   line_color_option=line_color_option,
                                   highlight_lines=lines, 
                                   timestep=timestep, 
                                   threshold=threshold,
                                   show_plot=False)
        fig.update_layout(uirevision = True)
        return fig

    def update_generation_geo_plot(self, result_name, switches, threshold, nodes):
        show_prices = True if 1 in switches else False
        show_redispatch = True  if 2 in switches else False
        show_curtailment = True  if 3 in switches else False
        result = self.pomato_instance.data.results[result_name]
        vis = self.pomato_instance.visualization
        fig =  vis.create_geo_plot(result, 
                                   show_redispatch=show_redispatch, 
                                   show_prices=show_prices,
                                   show_curtailment=show_curtailment,
                                   line_color_option=0,
                                   timestep=None, 
                                   highlight_nodes=nodes,
                                   threshold=threshold,
                                   show_plot=False)
        fig.update_layout(uirevision = True)
        return fig

