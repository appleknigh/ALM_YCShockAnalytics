#%%
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
fit_par_sim = np.tile(df_ycfit['fit_par'].iloc[1], [2, 1])

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

graph_2 = dbc.Card([
    dbc.Row(dbc.Col(dcc.Graph(id='graph_YC', figure=f1))),
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
            dbc.Col(html.H6('(30y-10y): base'), width=3),
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
            dbc.Col(html.H6('(30y-3m): base'), width=3),
            dbc.Col(html.H6('Δshock'), width=2)
        ], justify='center'),
    dbc.Row(
        [
            dbc.Col(dcc.Slider(id='slider_curve',
                               min=-2, max=2, value=fit_par_sim[1, 2], step=0.01), width=6),
            dbc.Col(html.H6('x%'), id='val_curve_base', width=3),
            dbc.Col(html.H6('x%'), id='val_curve_diff', width=2)
        ], justify='center'),
    dbc.Row(dbc.Col(html.H5('r(t) = beta0 + (beta1)f(slope) + (beta2)f(curve)',
                            style={'text-align': 'center'})))
])

graph_1 = dbc.Card([
    dbc.Row(dbc.Col(dcc.Graph(id='graph_ALM', figure=f2))),
    dbc.Row(
        [
            dbc.Col(dcc.Upload(html.Button('Upload asset & liability cashflows'), id='upload_ALM'),width=6)
        ], justify='center'),
    dbc.Row(html.Hr()),
    dbc.Row(dbc.Col(html.H5('Base scenario'))),
    dbc.Row(dbc.Col(html.Div('Asset duration: {:.2f}'.format(dur_asset_t1)),width={'offset':1})),
    dbc.Row(dbc.Col(html.Div('Liability duration: {:.2f}'.format(dur_liabilities_t1)),width={'offset':1})),
    dbc.Row(html.Hr()),
    dbc.Row(dbc.Col(html.H5('Shock scenario'))),
    dbc.Row(dbc.Col(html.Div('Asset duration: x',id='dur_asset_shock'),width={'offset':1})),
    dbc.Row(dbc.Col(html.Div('Liability duration: x',id='dur_liability_shock'),width={'offset':1}))
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
        Output(component_id='graph_ALM', component_property='figure'),
        Output(component_id='dur_asset_shock', component_property='children'),
        Output(component_id='dur_liability_shock', component_property='children')],
    [
        Input(component_id='slider_level', component_property='value'),
        Input(component_id='slider_slope', component_property='value'),
        Input(component_id='slider_curve', component_property='value')]
)
def update_figure(slider_level, slider_slope, slider_curve):
    fit_par_sim = np.tile(df_ycfit['fit_par'].iloc[1], [2, 1])
    fit_par_sim[1, 0] = slider_level
    fit_par_sim[1, 1] = -slider_slope
    fit_par_sim[1, 2] = slider_curve

    test_asset, test_liabiltiy, dur_asset_t2, dur_liabilities_t2 = ALM_kit.FactorAnalysis(
        fit_par_sim, t1=0, t2=1,
        cf_asset=cf_asset, t_asset=t_asset,
        cf_liability=cf_liability, t_liability=t_liability
    )

    dur_asset_text = 'Asset duration: {:.2f}'.format(dur_asset_t2)
    dur_liability_text = 'Liability duration: {:.2f}'.format(dur_liabilities_t2)

    tPar = np.append(np.linspace(0, 0.95, 20), np.linspace(1, 100, 100))
    f = BE.graph_shock(test_asset, test_liabiltiy)
    f.Movement_text()
    f.trace_figures(fit_par_sim, t_asset, cf_asset,
                    t_liability, cf_liability, tPar=tPar)
    f1, f2 = f.plot_figures()

    #Stats
    i_30yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 100.00
    i_10yrs = np.round(f.trace['scatter_ycbase']['x'], 2) == 10.00
    i_3mos = np.round(f.trace['scatter_ycbase']['x'], 2) == 0.05

    level_base = f.trace['scatter_ycbase']['y'][i_30yrs][0]
    level_shock = f.trace['scatter_ycshock']['y'][i_30yrs][0]
    slope_base = f.trace['scatter_ycbase']['y'][i_30yrs][0] - \
        f.trace['scatter_ycbase']['y'][i_10yrs][0]
    slope_shock = f.trace['scatter_ycshock']['y'][i_30yrs][0] - \
        f.trace['scatter_ycshock']['y'][i_10yrs][0]
    curve_base = f.trace['scatter_ycbase']['y'][i_30yrs][0] - \
        f.trace['scatter_ycbase']['y'][i_3mos][0]
    curve_shock = f.trace['scatter_ycshock']['y'][i_30yrs][0] - \
        f.trace['scatter_ycshock']['y'][i_3mos][0]

    text_level_beta = 'beta0 = ' + str(np.round(slider_level,1))
    text_level_base = str(np.round(level_base, 2)) + ' %'
    text_level_shock = str(np.round((level_shock-level_base)*100,0)) + ' bps'

    text_slope_beta = 'beta1 = ' + str(np.round(slider_slope,1))
    text_slope_base = str(np.round(slope_base, 2)) + ' %'
    text_slope_shock = str(np.round((slope_shock-slope_base)*100, 0)) + ' bps'

    text_curve_beta = 'beta2 = ' + str(np.round(slider_curve,1))
    text_curve_base = str(np.round(curve_base, 2)) + ' %'
    text_curve_shock = str(np.round((curve_shock-curve_base)*100, 0)) + ' bps'

    return text_level_beta, text_level_base, text_level_shock, text_slope_beta, text_slope_base, text_slope_shock, text_curve_beta, text_curve_base, text_curve_shock, f1, f2, dur_asset_text, dur_liability_text 


if __name__ == "__main__":
	app.run_server(debug=True)

# %%
