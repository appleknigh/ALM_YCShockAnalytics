# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:34:50 2020

@author: 330411836
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:41:38 2020

@author: 330411836
"""
# Added 1) YTM calculator and 2) duration optimizer

# To dos:
# 1) change the table to reflect movements in returns (asset allocation)
# 2) Add key duration labels to the chart (key duration movements)
# 3) add a button to do auto duration match (optimization)
# 4) quantify the YTM rate of the asset (return rate)
# 5) Add cashflow tab

# %% Import functions and data

#%%




from CSModules import ALM_kit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import quandl
def getfit(t1='2000-01-02', t2='2020-12-02'):
    '''
    Extract US Treasury data from quandl, and fitting NS parameters to it.
    t1 and t2 specify the time range to extract and fit.
    '''

    # Load data
    df_YieldC = quandl.get(
        "USTREASURY/YIELD", authtoken="4_zrDSANo7hMt_uhyzQy")
    df_YieldC.reset_index(level=0, inplace=True)
    df_YieldC['Date'] = pd.to_datetime(df_YieldC['Date'], format="%m/%d/%Y")

    # NS Cure fit
    t_cal = df_YieldC['Date']
    i_range = np.where((t_cal > t1) & (t_cal < t2))

    t = np.array([0.08333333, 0.16666667, 0.25,
                  0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.array(df_YieldC.iloc[:, 1:]).astype(float)[i_range]
    fit_par = pd.DataFrame(np.apply_along_axis(
        lambda x: ALM_kit.ns_par(t, x), axis=1, arr=y))
    return {'df_YieldC': df_YieldC, 't_cal': t_cal.iloc[i_range], 'tact': t, 'y': y, 'fit_par': fit_par}


# %%
class Cashflow_AL:
    def __init__(self, bond_weight=np.array([1.8, 0.2, 2.5]), liability_csv='Mx_2019.csv'):
        self.bond_weight = bond_weight
        self.liability_csv = liability_csv

    def Gen_Asset(self):
        '''Generating 3 assets by specifying bond structure
        bond_weight: composition of the three bonds in the portfolio
        '''
        # Generate cashflows for three bonds
        cf_bond_L, t_bond_L = ALM_kit.bond_cashflow(
            1000, 30, 2.5, 1)  # par, maturity, coupons, freq
        cf_bond_M, t_bond_M = ALM_kit.bond_cashflow(1000, 10, 2, 1)
        cf_bond_S, t_bond_S = ALM_kit.bond_cashflow(1000, 2, 1, 1)

        # Genrate porfolio of assets
        cf_bonds = np.array([cf_bond_L,
                             np.append(cf_bond_M, np.repeat(
                                 0, cf_bond_L.shape[0]-cf_bond_M.shape[0])),
                             np.append(cf_bond_S, np.repeat(0, cf_bond_L.shape[0]-cf_bond_S.shape[0]))])
        cf_singleasset = np.dot(self.bond_weight, cf_bonds)

        return cf_singleasset, t_bond_L

    def Gen_Liability(self):
        '''Read cashflow from liability csv file (currently formated only for Mx_2019.csv)
        '''
        df_mort = pd.read_csv(self.liability_csv, sep=',')
        cf_liability, t_liability = ALM_kit.liability_cashflow(
            df_mort.loc[60:]['Total'], 3000*12)

        return cf_liability, t_liability

#%%
class graph_shock():
    def __init__(self, asset_mov, liability_mov):
        self.asset_mov = asset_mov
        self.liability_mov = liability_mov
        self.trace = None

    def Movement_text(self):
        # Display overall movements in Asset and Liabitliy
        PV_asset_t1t2 = [self.asset_mov[0], self.asset_mov[-1]]
        self.PV_asset_t1t2_text = [str('${:,.2f}'.format(PV_asset_t1t2[i]))
                                   for i in range(len(PV_asset_t1t2))]

        PV_liabilities_t1t2 = [self.liability_mov[0], self.liability_mov[-1]]
        self.PV_liabilities_t1t2_text = [str('${:,.2f}'.format(
            PV_liabilities_t1t2[i])) for i in range(len(PV_liabilities_t1t2))]

        #Mismatch between asset and libility
        PV_ALMis = np.subtract(PV_asset_t1t2, PV_liabilities_t1t2)
        self.PV_ALMis_text = [str('${:,.2f}'.format(PV_ALMis[i]))
                              for i in range(len(PV_ALMis))]

        # Convert factored movements to percentage change
        test_asset_perc = np.append(
            np.exp(np.diff(np.log(self.asset_mov)))-1,  # movement step-by-step
            self.asset_mov[-1]/self.asset_mov[0]-1)*100  # overall movement
        self.movement_asset_text = [str(round(test_asset_perc[i]*100, 2)) +
                                    ' bps' for i in range(len(test_asset_perc))]

        test_liabiltiy_perc = np.append(np.exp(
            np.diff(np.log(self.liability_mov)))-1,
            self.liability_mov[-1]/self.liability_mov[0]-1)*100
        self.movement_liability_text = [str(round(test_liabiltiy_perc[i]*100, 2)) +
                                        ' bps' for i in range(len(test_liabiltiy_perc))]

        return self.PV_asset_t1t2_text, self.PV_liabilities_t1t2_text, self.PV_ALMis_text, self.movement_asset_text, self.movement_liability_text

    def trace_figures(self, fit_par_sim, t_asset, cf_asset, t_liability, cf_liability, tPar=np.linspace(0, 30, 50)):
        self.trace = {'scatter_ycbase': go.Scatter(x=tPar,
                                                   y=ALM_kit.yfit_beta(
                                                       tPar, fit_par_sim[0]),
                                                   mode='lines', line=dict(color='black'), showlegend=False),
                      'scatter_ycshock': go.Scatter(x=tPar,
                                                    y=ALM_kit.yfit_beta(
                                                        tPar, fit_par_sim[1]),
                                                    mode='lines', line=dict(color='blue',dash='dash'), showlegend=False),
                      'scatter_cashflow_aseset': go.Scatter(x=tPar,
                                                            y=ALM_kit.yfit_beta(
                                                                tPar, fit_par_sim[1]),
                                                            mode='lines', line=dict(color='red'), showlegend=False),
                      'scatter_cashflow_liability': go.Scatter(x=t_liability, y=cf_liability,
                                                               visible=False, line=dict(color='black'),
                                                               name='Cashflow - Liability'),
                      'table_overall': go.Table(header=dict(values=list(['PV', 'Asset', 'Liability', 'AL-Mismatch']),
                                                            fill_color='paleturquoise',
                                                            align='left'),
                                                cells=dict(values=[['Base', 'Shock'],
                                                                   self.PV_asset_t1t2_text,
                                                                   self.PV_liabilities_t1t2_text,
                                                                   self.PV_ALMis_text],
                                                           fill_color='lavender',
                                                           align='left')),

                      'table_factor': go.Table(header=dict(values=list(['%Movement <br>', 'Asset', 'Liability']),
                                                           fill_color='paleturquoise',
                                                           align='left'),
                                               cells=dict(values=[['Level', 'Slope', 'Curvature', 'Tau', '<b>Total</b>'],
                                                                  self.movement_asset_text,
                                                                  self.movement_liability_text],
                                                          fill_color='lavender',
                                                          align='left'))
                      }
    def plot_figures(self):
        f_scatter = go.Figure()
        f_scatter.add_trace(self.trace['scatter_ycbase'])
        f_scatter.add_trace(self.trace['scatter_ycshock'])

        f_table = make_subplots(
            rows=3, cols=1,
            specs=[
                [{"type": "table"}],
                [{"type": "table", "rowspan": 2}],
                [None]
            ]
        )

        f_table.add_trace(self.trace['table_overall'],row=1,col=1)
        f_table.add_trace(self.trace['table_factor'],row=2,col=1)


        return f_scatter, f_table

#%%


# # Button switch
# fig.update_layout(
#     autosize=False,
#     width=1200,
#     height=500,
#     updatemenus=[
#         dict(
#             buttons=list([
#                 dict(label="Yield Curve",
#                      method="update",
#                      args=[{'visible': [True, True, True, True, False, False]}
#                            ]
#                      ),
#                 dict(label="Cash flow",
#                      method="update",
#                      args=[
#                          {'visible': [False, False, False, False, True, True]}
#                      ]
#                      )
#             ]
#             ), yanchor="top", xanchor="left", y=1.25, x=0
#         )
#     ])


# #%%
# #Extract and fit yc data
# df_ycfit = getfit(t1='2020-01-02', t2='2020-12-02')

# #%%
# #Get cashflow for asset and liability
# x = Cashflow_AL(bond_weight=3*50*np.array([1.8, 0.2, 2.5]))
# cf_liability, t_liability = x.Gen_Liability()
# cf_asset, t_asset = x.Gen_Asset()

# #%%
# #Factor analysis of the PV movement
# fit_par_sim = np.empty([2, 2])
# fit_par_sim = np.tile(df_ycfit['fit_par'].iloc[1], [2, 1])
# fit_par_sim[1] = fit_par_sim[1]+0.2

# test_asset, test_liabiltiy, dur_asset_t2, dur_liabilities_t2 = ALM_kit.FactorAnalysis(
#     fit_par_sim, t1=0, t2=1,
#     cf_asset=cf_asset, t_asset=t_asset,
#     cf_liability=cf_liability, t_liability=t_liability
# )

# #%%
# #Graph and plotting
# f = graph_shock(test_asset, test_liabiltiy)
# f.Movement_text()
# f.trace_figures(fit_par_sim, t_asset, cf_asset, t_liability, cf_liability)
# f1, f2 = f.plot_figures()
# f2