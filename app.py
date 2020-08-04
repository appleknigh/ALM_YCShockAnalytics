#%%
import matplotlib.pyplot as plt
from CSModules import nssmodel
import copy
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash
from CSModules import ALM_kit
import numpy as np
from scipy.optimize import minimize
import Shock_testscript as BE
import dash_daq as daq
import plotly.graph_objects as go


#%%
#Extract and fit yc data
df_ycfit = BE.getfit(t1='2020-01-02', t2='2020-12-02')

#%%
#Get cashflow for asset and liability
x = BE.Cashflow_AL(bond_weight=3*50*np.array([1.8, 0.2, 2.5]))
cf_liability, t_liability = x.Gen_Liability()
cf_asset, t_asset = x.Gen_Asset()

#%%
#Factor analysis of the PV movement
fit_par_sim = np.empty([2, 2])
fit_par_sim = np.tile(df_ycfit['fit_par'].iloc[0], [2, 1])

test_asset, test_liabiltiy, dur_asset_t1, dur_liabilities_t1 = ALM_kit.FactorAnalysis(
    fit_par_sim, t1=0, t2=1,
    cf_asset=cf_asset, t_asset=t_asset,
    cf_liability=cf_liability, t_liability=t_liability
)

#%%
tPar = np.append(np.linspace(0, 0.95, 20), np.linspace(1, 30, 59))
f = BE.graph_shock(test_asset, test_liabiltiy)
f.Movement_text()
f.trace_figures(fit_par_sim, t_asset, cf_asset,
                t_liability, cf_liability, tPar=tPar)
f1, f2 = f.plot_figures()


#%%
#Dash

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

Pannel_Paramettric = [
    dbc.Row(dbc.Col(html.H5('Long-term level'))),
    dbc.Row(
        [
            dbc.Col(html.H6('beta0 = x', id='beta0'), width=6),
            dbc.Col(html.H6('30y: base'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Slider(id='slider_level',
                               min=0, max=5, value=fit_par_sim[1, 0], step=0.01), width=6),
            dbc.Col(html.H6('x%'), id='val_level_base', width=3),
            dbc.Col(html.H6('x%'), id='val_level_diff', width=2)
        ], justify='center'),
    dbc.Row(dbc.Col(html.H5('Inversion Slope'))),
    dbc.Row(
        [
            dbc.Col(html.H6('beta1 = x', id='beta1'), width=6),
            dbc.Col(html.H6('(10y-3m): base'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Slider(id='slider_slope',
                               min=-2, max=2, value=-fit_par_sim[1, 1], step=0.01), width=6),
            dbc.Col(html.H6('x%'), id='val_slope_base', width=3),
            dbc.Col(html.H6('x%'), id='val_slope_diff', width=2)
        ], justify='center'),
    dbc.Row(dbc.Col(html.H5('Humped Curve'))),
    dbc.Row(
        [
            dbc.Col(html.H6('beta2 = x', id='beta2'), width=6),
            dbc.Col(html.H6('(5y-3y): base'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Slider(id='slider_curve',
                               min=-2, max=2, value=fit_par_sim[1, 2], step=0.01), width=6),
            dbc.Col(html.H6('x%'), id='val_curve_base', width=3),
            dbc.Col(html.H6('x%'), id='val_curve_diff', width=2)
        ], justify='center')]

drop_terms = [
    {'label': '1 mo', 'value': 0.08},
    {'label': '2 mo', 'value': 0.17},
    {'label': '3 mo', 'value': 0.25},
    {'label': '6 mo', 'value': 0.5},
    {'label': '1 yr', 'value': 1},
    {'label': '2 yr', 'value': 2},
    {'label': '3 years', 'value': 3},
    {'label': '5 years', 'value': 5},
    {'label': '7 years', 'value': 7},
    {'label': '10 years', 'value': 10},
    {'label': '20 years', 'value': 20},
    {'label': '30 years', 'value': 30}]

Pannel_KeyDuration = [
    dbc.Row(
        [
            dbc.Col(html.H6('Key Duration #1'), width=6),
            dbc.Col(html.H6('rate'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Dropdown(
                id='drop_dur1', options=drop_terms, placeholder='Please pick a term'), width=6),
            dbc.Col(html.H6('x%'), id='val_dur1_base', width=3),
            dbc.Col(dcc.Input(id='val_dur1_diff',value='0',
                              style={'width': 80}), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(html.H6('Key Duration #2'), width=6),
            dbc.Col(html.H6('rate'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Dropdown(
                id='drop_dur2', options=drop_terms, placeholder='Please pick a term'), width=6),
            dbc.Col(html.H6('x%'), id='val_dur2_base', width=3),
            dbc.Col(dcc.Input(id='val_dur2_diff',value='0',
                              style={'width': 80}), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(html.H6('Key Duration #3'), width=6),
            dbc.Col(html.H6('rate'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Dropdown(
                id='drop_dur3', options=drop_terms, placeholder='Please pick a term'), width=6),
            dbc.Col(html.H6('x%'), id='val_dur3_base', width=3),
            dbc.Col(dcc.Input(id='val_dur3_diff',value='0',
                              style={'width': 80}), width=2)
        ], justify='center')]

graph_2 = dbc.Card([
    dbc.Row(dbc.Col(dcc.Graph(id='graph_YC', figure=f1))),
    dcc.Tabs(id='shock_tabs',value='tab-1',children=[
        dcc.Tab(label='Key Duration',value='tab-1', children=Pannel_KeyDuration),
        dcc.Tab(label='Parametric', value='tab-2',children=Pannel_Paramettric)
    ]),
    dbc.Row(dbc.Col(html.Button('Submit', id='submit-button-state', n_clicks=0)))
])

graph_1 = dbc.Card([
    dbc.Row(dbc.Col(dcc.Graph(id='graph_ALM', figure=f2))),
    dbc.Row(
        [
            dbc.Col(dcc.Upload(html.Button(
                'Upload asset & liability cashflows'), id='upload_ALM'), width=6)
        ], justify='center'),
    dbc.Row(html.Hr()),
    dbc.Row(dbc.Col(html.H5('Base scenario'))),
    dbc.Row(dbc.Col(html.Div('Asset duration: {:.2f}'.format(
        dur_asset_t1)), width={'offset': 1})),
    dbc.Row(dbc.Col(html.Div('Liability duration: {:.2f}'.format(
        dur_liabilities_t1)), width={'offset': 1}))
])

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(graph_2),
                dbc.Col(graph_1)
            ]
        )
    ]
)

app.layout = body

# @app.callback(Output('output-data-upload', 'children'),
#               [Input('upload_ALM', 'contents')],
#               [State('upload_ALM', 'filename'),
#                State('upload_ALM', 'last_modified')])
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children

@app.callback(
    [
        Output(component_id='beta0', component_property='children'),
        Output(component_id='val_level_base', component_property='children'),
        Output(component_id='val_level_diff', component_property='children'),
        Output(component_id='beta1', component_property='children'),
        Output(component_id='val_slope_base', component_property='children'),
        Output(component_id='val_slope_diff', component_property='children'),
        Output(component_id='beta2', component_property='children'),
        Output(component_id='val_curve_base', component_property='children'),
        Output(component_id='val_curve_diff', component_property='children'),
        Output(component_id='graph_YC', component_property='figure'),
        Output(component_id='graph_ALM', component_property='figure')],

    [Input('submit-button-state', 'n_clicks')],
    [
        State(component_id='slider_level', component_property='value'),
        State(component_id='slider_slope', component_property='value'),
        State(component_id='slider_curve', component_property='value'),
        State(component_id='shock_tabs', component_property='value'),
        State(component_id='drop_dur1', component_property='value'),
        State(component_id='drop_dur2', component_property='value'),
        State(component_id='drop_dur3', component_property='value'),
        State(component_id='val_dur1_diff', component_property='value'),
        State(component_id='val_dur2_diff', component_property='value'),
        State(component_id='val_dur3_diff', component_property='value')]
)

def update_figure(nclick,slider_level, slider_slope, slider_curve,tabs,keydur1,keydur2,keydur3,sdur1,sdur2,sdur3):
    # define parameters
    fit_par_sim = np.tile(df_ycfit['fit_par'].iloc[0], [2, 1])

    if tabs == 'tab-1':
        # original rates        
        t = df_ycfit['tact']
        y2 = copy.deepcopy(df_ycfit['y'][0])

        i_1 = t==keydur1
        i_2 = t==keydur2
        i_3 = t==keydur3

        #Shocked rates
        y2[i_1] += float(sdur1)/100
        y2[i_2] += float(sdur2)/100
        y2[i_3] += float(sdur3)/100

        fit_par_sim[1,:] = ALM_kit.ns_par(t,y2)
        print(y2)
        print(fit_par_sim)
    else:
        fit_par_sim[1, 0] = slider_level
        fit_par_sim[1, 1] = -slider_slope
        fit_par_sim[1, 2] = slider_curve

    #Calculate factor analysis
    test_asset, test_liabiltiy, _, _ = ALM_kit.FactorAnalysis(
        fit_par_sim, t1=0, t2=1,
        cf_asset=cf_asset, t_asset=t_asset,
        cf_liability=cf_liability, t_liability=t_liability
    )
  
    #Genderate figures
    tPar = np.append(np.linspace(0, 0.95, 20), np.linspace(1, 30, 30))
    f = BE.graph_shock(test_asset, test_liabiltiy)
    f.Movement_text()
    f.trace_figures(fit_par_sim, t_asset, cf_asset,
                    t_liability, cf_liability, tPar=tPar)
    f1, f2 = f.plot_figures()

    if tabs == 'tab-1':
        f1.add_trace(go.Scatter(x=df_ycfit['tact'],y=y2,mode="markers",marker=dict(color="Red"),name='shock rates'))
        f1.add_trace(go.Scatter(x=df_ycfit['tact'],y=df_ycfit['y'][0],mode="markers",marker=dict(color="MediumPurple"),name='raw rates'))
    else:
        f1.add_trace(go.Scatter(x=df_ycfit['tact'],y=df_ycfit['y'][0],mode="markers",marker=dict(color="MediumPurple"),name='raw rates'))

    # Stats for tab 2
    i_30yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 30.00
    i_10yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 10.00
    i_5yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 5.00
    i_3yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 3.00
    i_3mos = np.round(f.trace['scatter_ycbase']['x'], 2) == 0.05

    level_base = f.trace['scatter_ycbase']['y'][i_30yrs][0]
    level_shock = f.trace['scatter_ycshock']['y'][i_30yrs][0]
    slope_base = f.trace['scatter_ycbase']['y'][i_10yrs][0] - \
        f.trace['scatter_ycbase']['y'][i_3mos][0]
    slope_shock = f.trace['scatter_ycshock']['y'][i_10yrs][0] - \
        f.trace['scatter_ycshock']['y'][i_3mos][0]
    curve_base = f.trace['scatter_ycbase']['y'][i_5yrs][0] - \
        f.trace['scatter_ycbase']['y'][i_3yrs][0]
    curve_shock = f.trace['scatter_ycshock']['y'][i_5yrs][0] - \
        f.trace['scatter_ycshock']['y'][i_3yrs][0]

    text_level_beta = 'beta0 = ' + str(np.round(slider_level, 1))
    text_level_base = str(np.round(level_base, 2)) + ' %'
    text_level_shock = str(np.round((level_shock-level_base)*100, 0)) + ' bps'

    text_slope_beta = 'beta1 = ' + str(np.round(slider_slope, 1))
    text_slope_base = str(np.round(slope_base, 2)) + ' %'
    text_slope_shock = str(np.round((slope_shock-slope_base)*100, 0)) + ' bps'

    text_curve_beta = 'beta2 = ' + str(np.round(slider_curve, 1))
    text_curve_base = str(np.round(curve_base, 2)) + ' %'
    text_curve_shock = str(np.round((curve_shock-curve_base)*100, 0)) + ' bps'

    # Stats for tab 1

    return [
        text_level_beta, text_level_base, text_level_shock,
        text_slope_beta, text_slope_base, text_slope_shock,
        text_curve_beta, text_curve_base, text_curve_shock,
        f1, f2]

if __name__ == "__main__":
	app.run_server(debug=True)

# %%
