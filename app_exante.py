# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:57:02 2020

@author: 330411836
"""
    
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash
from CSModules import ALM_kit
import numpy as np
from scipy.optimize import minimize
import utility_exante as BE
#Dash

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Demo",
    brand_href="#",
    sticky="top",
)

body = dbc.Container(
    [dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(id='graph_YcReturnAlloc', figure=BE.fig)
                ]
            )
        ]
    ),
        dbc.Row([dbc.Col([
            html.Div(id='time_range_select', children='Time Range: ?, ?'),
            dcc.RangeSlider(
                 id='time_range',
                 min=0,
                 max=BE.fit_par.shape[0]-1,
                 step=1,
                 value=[1, 100]
                 )
        ]),
            dbc.Col([
                html.Div([
                    html.Div(
                        id='Asset_mix', children='Bond Mix: [Long, Median, Short]'),
                    html.Div(dcc.Input(id='asset_mix_input',
                                       type='text',value='1 1 1'))
                ], style={'columnCount': 2}
                ),
                html.Button('Duration match', id='Op'),
                html.Button('Submit', id='button'),
                html.Div(id='asset_dur', children=''),
                html.Div(id='liability_dur', children='')
            ]
        )
        ])
    ]
)

app.layout = html.Div([navbar, body])


@app.callback(
    [Output(component_id='graph_YcReturnAlloc', component_property='figure'),
     Output(component_id='asset_dur', component_property='children'),
     Output(component_id='liability_dur', component_property='children'),
     Output(component_id='asset_mix_input', component_property='value'),
     Output(component_id='time_range_select', component_property='children')],
    [Input(component_id='time_range', component_property='value'),
     Input(component_id='button', component_property='n_clicks'),
     Input(component_id='Op', component_property='n_clicks')],
    [State(component_id='asset_mix_input', component_property='value')]
)

def update_figure(time_range, n_clicks_submit, n_clicks_duration, Asset_text):

    #Inputs
    #Input for asset mixes
    asset_mix = np.fromstring(Asset_text, dtype=float, sep=' ')

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    #Input for yield t1 and t2
    t1 = time_range[0]
    t2 = time_range[1]

    time_range_t1t2 = 'Range Slider: ' + ' y1 <' + str(BE.t_cal.iloc[t1]) + '> y2 <' + str(BE.t_cal.iloc[t2]) + '>'

#Recalculate
    #Extract new points for Yield Curve plot
    y1 = BE.y[t1]
    y2 = BE.y[t2]
    fit_par_t1 = BE.fit_par[t1]
    fit_par_t2 = BE.fit_par[t2]

    #Recalculate values for tables
    #Duration recalculate
    PV_Lt1, dur_liabilities_t1 = ALM_kit.PV_cashflow(
        BE.cf_liability, BE.t_liability, fit_ns=BE.fit_par[t1])

    if button_id == 'Op':
        def res_func(x): return ALM_kit.optimize_duration(
            x, BE.cf_bonds, BE.t_bond_L, BE.fit_par[t1], dur_liabilities_t1)
        res = minimize(res_func, asset_mix, method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': True})
        asset_mix = abs(res.x)
        cf_weights = abs(res.x)  # Bond weight (Duration matched)
    else:
        cf_weights = asset_mix

    cf_bond_single = np.dot(cf_weights, BE.cf_bonds)
    PV_At1, _ = ALM_kit.PV_cashflow(
        cf_bond_single, BE.t_bond_L, fit_ns=BE.fit_par[t1])
    nBond = PV_Lt1/PV_At1

    cf_bond = cf_bond_single*nBond
    _, dur_asset_t1 = ALM_kit.PV_cashflow(
        cf_bond, BE.t_bond_L, fit_ns=BE.fit_par[t1])
    _, dur_liabilities_t2 = ALM_kit.PV_cashflow(
        BE.cf_liability, BE.t_liability, fit_ns=BE.fit_par[t2])
    _, dur_asset_t2 = ALM_kit.PV_cashflow(
        cf_bond, BE.t_bond_L, fit_ns=BE.fit_par[t2])

    #Asset mixes selection
    duration_asset_text = "Dur_asset @ y1: {:.2f}, y2: {:.2f}".format(
        dur_asset_t1, dur_asset_t2)
    duration_liability_text = "Dur_liability @ y1: {:.2f}, y2: {:.2f}".format(
        dur_liabilities_t1, dur_liabilities_t2)

    #Asset allocation recalculate
    test_asset, test_liabiltiy, _, _ = ALM_kit.FactorAnalysis(BE.fit_par, t1, t2,
                                                              cf_asset=(
                                                                  cf_bond), t_asset=BE.t_bond_L,
                                                              cf_liability=BE.cf_liability, t_liability=BE.t_liability)
    #Table values recalculate

    test_asset_perc = np.append(
        np.exp(np.diff(np.log(test_asset)))-1, test_asset[-1]/test_asset[0]-1)*100
    test_liabiltiy_perc = np.append(np.exp(
        np.diff(np.log(test_liabiltiy)))-1, test_liabiltiy[-1]/test_liabiltiy[0]-1)*100

    PV_asset_t1t2 = [test_asset[0], test_asset[-1]]
    PV_liabilities_t1t2 = [test_liabiltiy[0], test_liabiltiy[-1]]
    PV_ALMis = [test_asset[0]-test_liabiltiy[0],
                test_asset[-1]-test_liabiltiy[-1]]

    PV_asset_t1t2_text = [str('${:,.2f}'.format(PV_asset_t1t2[i]))
                          for i in range(len(PV_asset_t1t2))]
    PV_liabilities_t1t2_text = [str('${:,.2f}'.format(
        PV_liabilities_t1t2[i])) for i in range(len(PV_liabilities_t1t2))]
    PV_ALMis_text = [str('${:,.2f}'.format(PV_ALMis[i]))
                     for i in range(len(PV_ALMis))]

    movement_asset = [str(round(test_asset_perc[i]*100, 1)) +
                      ' bps' for i in range(len(test_asset_perc))]
    movement_liability = [str(round(test_liabiltiy_perc[i]*100, 1)) +
                          ' bps' for i in range(len(test_liabiltiy_perc))]

    table_u = BE.fig.data[-2]
    table_b = BE.fig.data[-1]

    with BE.fig.batch_update():
        BE.fig.data[0].y = ALM_kit.yfit_beta(BE.tPar, fit_par_t1)
        BE.fig.data[1].y = ALM_kit.yfit_beta(BE.tPar, fit_par_t2)
        BE.fig.data[2].y = y1
        BE.fig.data[3].y = y2
        BE.fig.data[4].y = cf_bond

        table_u.cells.values = [['Yield 1', 'Yield 2'],
                                PV_asset_t1t2_text,
                                PV_liabilities_t1t2_text,
                                PV_ALMis_text]

        table_b.cells.values = [['Level', 'Slope', 'Curvature', 'Tau', 'TOTAL'],
                                movement_asset,
                                movement_liability]

    return BE.fig, duration_asset_text, duration_liability_text,str(np.round(cf_weights/np.sum(cf_weights), 4)).strip('[]'), time_range_t1t2

if __name__ == "__main__":
	app.run_server(debug=True)