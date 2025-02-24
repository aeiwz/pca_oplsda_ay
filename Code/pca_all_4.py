
# All

# Import the required python packages including 
# the custom Chemometric Model objects
import numpy as np



from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from pyChemometrics.ChemometricsScaler import ChemometricsScaler

# Use to obtain same values as in the text
np.random.seed(350)

import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn import decomposition
from sklearn.preprocessing import scale
from pca_ellipse import confidence_ellipse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from lingress.unipair import Unipair


df = pd.read_csv('/Volumes/CAS9/Aeiwz/Project/KOI_CC/Code/dataset_4.csv')
#Drop QC samples
df = df.drop(df[df['Technique'] == 'QC'].index)

Group="Technique"
plot_name = "All_4"


#Make directory
# path folder
report_path = '/Volumes/CAS9/Aeiwz/Project/KOI_CC/Report'
project = 'PCA_report_all_4'
# Create directories if they don't exist
os.makedirs('{}/{}'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/HTML'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/PNG'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/Scores'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/Loading'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/R2'.format(report_path, project), exist_ok=True)
os.makedirs('{}/{}/Trajectory'.format(report_path, project), exist_ok=True)





# path folder
PCA_result_path = '{}/{}'.format(report_path, project)
HTML_save = '{}/HTML'.format(PCA_result_path)
PNG_save = '{}/PNG'.format(PCA_result_path)
Scores_save = '{}/Scores'.format(PCA_result_path)
Loading_save = '{}/Loading'.format(PCA_result_path)
R2_save = '{}/R2'.format(PCA_result_path)
Trajectory_save = '{}/Trajectory'.format(PCA_result_path)

# Import the datasets from the /data directory
# X for the NMR spectra and Y for the 2 outcome variables
test_gr = df

X = test_gr.iloc[:, 113:]
#fill nan with 0
X = X.fillna(0)
meta = test_gr.iloc[:, :113]
Y = test_gr[Group]
Y1 = pd.Categorical(Y).codes
ppm = list(np.ravel(X.columns).astype(float))
# Use pandas Categorical type to generate the dummy enconding of the Y vector (0 and 1) 



import time

from tqdm import tqdm

T1 = time.time()


    # Select the scaling options: 
# Here we are generating 3 scaling objects to explore the effect of scaling in PCA:

# Unit-Variance (UV) scaling:

scale__ = 'Par'
scale_power_ = 0.5

# Mean Centering (MC):
#scaling_object_mc = ChemometricsScaler(scale_power=0)

# Pareto scaling (Par):
# scaling_object_par = ChemometricsScaler(scale_power=0.5)


model_scaler = ChemometricsScaler(scale_power=scale_power_)
model_scaler.fit(X)
model_X = model_scaler.transform(X)

pca_model = decomposition.PCA(n_components=2)
pca_model.fit(model_X)

scores_ = pca_model.transform(model_X)
df_scores_ = pd.DataFrame(scores_, columns=['PC1', 'PC2'])
df_scores_.index = test_gr.index

df2_scores_ = pd.concat([df_scores_, Y], axis=1)

#save PCA score to csv
df2_scores_.to_csv(Scores_save+'/PCA_scores_'+ plot_name +'.csv')

loadings_ = pca_model.components_.T
df_loadings_ = pd.DataFrame(loadings_, columns=['PC1', 'PC2'], index=np.ravel(ppm))
df_loadings_.to_csv(Loading_save + '/Loading_scores ' + plot_name + '.csv')

explained_variance_ = pca_model.explained_variance_ratio_
explained_variance_

explained_variance_ = np.insert(explained_variance_, 0, 0)

cumulative_variance_ = np.cumsum(np.round(explained_variance_, decimals=3))

pc_df_ = pd.DataFrame(['','PC1', 'PC2'], columns=['PC'])
explained_variance_df_ = pd.DataFrame(explained_variance_, columns=['Explained Variance'])
cumulative_variance_df_ = pd.DataFrame(cumulative_variance_, columns=['Cumulative Variance'])

df_explained_variance_ = pd.concat([pc_df_, explained_variance_df_, cumulative_variance_df_], axis=1)
df_explained_variance_.to_csv(R2_save + '/R2 ' + plot_name + '.csv')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=357)
X_test = model_scaler.transform(X_test)
X_test_pca = pca_model.transform(X_test)

# Inverse transform the test set from the PCA space
X_test_reconstructed = pca_model.inverse_transform(X_test_pca)


# Calculate Q2 score for the test set
q2_test = r2_score(X_test, X_test_reconstructed)
        

# Plot

# https://plotly.com/python/bar-charts/

fig = px.bar(df_explained_variance_, 
            x='PC', y='Explained Variance',
            text='Explained Variance',
            width=800, height=600,
            title='Explained Variance ({} scaling)'.format(scale__))
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(size=15))
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#fig.show()
fig.write_image(PNG_save + "/Explained Variance " + plot_name + ".png")
fig.write_html(HTML_save + "/Explained Variance " + plot_name + ".html")

# https://plotly.com/python/creating-and-updating-figures/
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_explained_variance_['PC'],
        y=df_explained_variance_['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen"),
        name='R<sup>2</sup>X (Cum)'
    ))

fig.add_trace(
    go.Bar(
        x=df_explained_variance_['PC'],
        y=df_explained_variance_['Explained Variance'],
        marker=dict(color="RoyalBlue"),
        name='R<sup>2</sup>X',
        text=np.round(df_explained_variance_['Explained Variance'], decimals=3)
    ))
fig.update_layout(width=800, height=600,
                title='Explained Variance and Cumulative Variance ' + plot_name)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

#fig.show()
fig.write_image(PNG_save + "/Explained Variance + Cumulative Variance " + plot_name + ".png")
fig.write_html(HTML_save + "/Explained Variance + Cumulative Variance " + plot_name + ".html")


colour_dict = {
                "Off pump CABG (OPCAB)": "#E91E63",        
                "On pump CABG (ONCAB)": "#FF9800",
                "custodiol": "#FFEB3B",       
                "del Nido": "#9C27B0",
                "st. thomas": "#03A9F4",
                "No med": "#4CAF50",        
                "7": "#B30000",
                "8": "#3F51B5"
                }





# PCA plot
pca_label = df2_scores_.index


fig = px.scatter(df2_scores_, x='PC1', y='PC2', symbol=meta['Time point'],
                    
                    symbol_map={'Pre-op': 'circle',
                        'Op': 'diamond',
                        'Post-op1': 'square',
                        'Post-op2': 'triangle-up',
                        'Post-op3': 'triangle-down'},
                    
                color=Group,
                color_discrete_map=colour_dict, 
                title='<b>PCA Scores Plot ({} Scaling)<b>'.format(scale__), 
                height=900, width=1300,
                labels={"PC1": "PC1 R<sup>2</sup>X: {} %".format(np.round(df_explained_variance_.iloc[1,1]*100, decimals=2)),
                        "PC2": "PC2 R<sup>2</sup>X: {} %".format(np.round(df_explained_variance_.iloc[2,1]*100, decimals=2))})

#fig.add_annotation(yref = 'paper', y = -1.06, xref = 'paper', x=1.06 , text='Q2' +' = {}'.format(np.round(df_explained_variance_.iloc[2,2], decimals=2)))
#fig.update_annotations(font = {
#    'size': 20}, showarrow=False)

#set data point fill alpha with boarder in each color
fig.update_traces(marker=dict(size=35, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')))

fig.add_annotation(dict(font=dict(color="black",size=20),
                        #x=x_loc,
                        x=1.0,
                        y=0.05,
                        showarrow=False,
                        text='<b>R<sup>2</sup>X (Cum): {}%<b>'.format(np.round(df_explained_variance_.iloc[2,2]*100, decimals=2)),
                        textangle=0,
                        xref="paper",
                        yref="paper"),
                        # set alignment of text to left side of entry
                        align="left")

fig.add_annotation(dict(font=dict(color="black",size=20),
                        #x=x_loc,
                        x=1.0,
                        y=0.01,
                        showarrow=False,
                        text='<b>Q<sup>2</sup>X (Cum): {}%<b>'.format(np.round(q2_test*100, decimals=2)),
                        textangle=0,
                        xref="paper",
                        yref="paper"),
                        # set alignment of text to left side of entry
                        align="left")



fig.update_traces(marker=dict(size=20))
#fig.update_traces(textposition='top center') #Text label position

#fig.update_traces(marker=dict(size=12, color=Y1_color, marker=Y2_marker))
fig.add_shape(type='path',
            path=confidence_ellipse(df2_scores_['PC1'], df2_scores_['PC2']))



fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(
    title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(size=20))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

#fig.show()
fig.write_image(PNG_save + "/PCA " + plot_name + ".png")
fig.write_html(HTML_save + "/PCA " + plot_name + ".html")


# Loading plot
loadings_label = df_loadings_.index


fig = px.line(df_loadings_, x=loadings_label, y=['PC1', 'PC2'],
                height=600, width=1800,
                title='Loadings ' + plot_name
                )

fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_layout(title={'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                font=dict(size=20))

fig.update_layout(scene={'xaxis': {'autorange': 'reversed'}})
        
fig.update_traces(marker=dict(size=1))
fig.update_layout(xaxis_title="𝛿<sub>H</sub> in ppm")
#fig.show()

fig.write_image(PNG_save + "/Loading " + plot_name + ".png")
fig.write_html(HTML_save + "/Loading " + plot_name + ".html")


#Time trajectory
traject_df = df2_scores_.copy()
traject_df['Time point'] = meta['Time point']
med_df = traject_df.groupby(['Time point', Group]).median()
err_df = traject_df.groupby(['Time point', Group]).sem()

med_df = med_df.reset_index()
err_df = err_df.reset_index()


med_df.sort_values(by=['Time point'], inplace=True, key=lambda x: x.map({"Pre-op": 0, "Op": 1, "Post-op1": 2, "Post-op2": 3, "Post-op3": 4}))
med_df.sort_values(by=[Group], inplace=True)
med_df.reset_index(drop=True, inplace=True)

err_df.sort_values(by=['Time point'], inplace=True, key=lambda x: x.map({"Pre-op": 0, "Op": 1, "Post-op1": 2, "Post-op2": 3, "Post-op3": 4}))
err_df.sort_values(by=[Group], inplace=True)
err_df.reset_index(drop=True, inplace=True)


list_label = med_df[Group].unique()
#line 1
if list_label[0] == 'Off pump CABG (OPCAB)': 
    colour1 = '#E91E63'
elif list_label[0] == 'On pump CABG (ONCAB)':
    colour1 = '#FF9800'
elif list_label[0] == 'custodiol':
    colour1 = '#FFEB3B'
elif list_label[0] == 'del Nido':
    colour1 = '#9C27B0'
elif list_label[0] == 'st. thomas':
    colour1 = '#03A9F4'
elif list_label[0] == 'No med':
    colour1 = '#4CAF50'

#line 2
if list_label[1] == 'Off pump CABG (OPCAB)':
    colour2 = '#E91E63'
elif list_label[1] == 'On pump CABG (ONCAB)':
    colour2 = '#FF9800'
elif list_label[1] == 'custodiol':
    colour2 = '#FFEB3B'
elif list_label[1] == 'del Nido':
    colour2 = '#9C27B0'
elif list_label[1] == 'st. thomas':
    colour2 = '#03A9F4'
elif list_label[1] == 'No med':
    colour2 = '#4CAF50'
    



fig = px.line(med_df, x='PC1', y='PC2', line_group='Time point', error_x=err_df["PC1"], error_y=err_df["PC2"],
                color=Group, 
                symbol='Time point',
                color_discrete_map=colour_dict,
                symbol_map={'Pre-op': 'circle',
                        'Op': 'diamond',
                        'Post-op1': 'square',
                        'Post-op2': 'triangle-up',
                        'Post-op3': 'triangle-down'}, 
                title='<b>Principle component analysis (Pareto scaling)<b>', 
                height=900, width=1300,
                labels={"PC1": "PC1 R<sup>2</sup>X: {} %".format(24.3),
                        "PC2": "PC2 R<sup>2</sup>X: {} %".format(15.0)})


# create a new trace for the connecting line
fig.add_trace(go.Scatter(
    x=med_df.loc[0:4, "PC1"], # x-coordinates of the line
    y=med_df.loc[0:4, "PC2"], # y-coordinates of the line
    mode='lines', # specify the trace type as lines
    line=dict(color=colour1, width=2), # set the color and width of the line
    showlegend=False # hide the trace from the legend
))


# create a new trace for the connecting line
fig.add_trace(go.Scatter(
    x=med_df.loc[5:9, "PC1"], # x-coordinates of the line
    y=med_df.loc[5:9, "PC2"], # y-coordinates of the line
    mode='lines', # specify the trace type as lines
    line=dict(color=colour2, width=2), # set the color and width of the line
    showlegend=False # hide the trace from the legend
))
'''
# create a new trace for the connecting line
fig.add_trace(go.Scatter(
    x=df_score_mean.iloc[4:8, 0], # x-coordinates of the line
    y=df_score_mean.iloc[4:8, 2], # y-coordinates of the line
    mode='lines', # specify the trace type as lines
    line=dict(color='#84CC56', width=2), # set the color and width of the line
    showlegend=False # hide the trace from the legend
))


# create a new trace for the connecting line
fig.add_trace(go.Scatter(
    x=df_score_mean.iloc[8:12, 0], # x-coordinates of the line
    y=df_score_mean.iloc[8:12, 2], # y-coordinates of the line
    mode='lines', # specify the trace type as lines
    line=dict(color='#CA83CC', width=2), # set the color and width of the line
    showlegend=False # hide the trace from the legend
))

# create a new trace for the connecting line
fig.add_trace(go.Scatter(
    x=df_score_mean.iloc[8:12, 0], # x-coordinates of the line
    y=df_score_mean.iloc[8:12, 2], # y-coordinates of the line
    mode='lines', # specify the trace type as lines
    line=dict(color='#6AE022', width=2), # set the color and width of the line
    showlegend=False # hide the trace from the legend
))

'''





fig.add_annotation(dict(font=dict(color="black",size=20),
                        #x=x_loc,
                        x=1.0,
                        y=0.05,
                        showarrow=False,
                        text='<b>R<sup>2</sup>X (Cum): {}%<b>'.format(np.round(df_explained_variance_.iloc[2,2]*100, decimals=2)),
                        textangle=0,
                        xref="paper",
                        yref="paper"),
                        # set alignment of text to left side of entry
                        align="left")

fig.add_annotation(dict(font=dict(color="black",size=20),
                        #x=x_loc,
                        x=1.0,
                        y=0.01,
                        showarrow=False,
                        text='<b>Q<sup>2</sup>X (Cum): {}%<b>'.format(np.round(q2_test*100, decimals=2)),
                        textangle=0,
                        xref="paper",
                        yref="paper"),
                        # set alignment of text to left side of entry
                        align="left")

fig.update_traces(marker=dict(size=20))
#fig.update_traces(textposition='top center') #Text label position

#fig.update_traces(marker=dict(size=12, color=Y1_color, marker=Y2_marker))
fig.add_shape(type='path',
            path=confidence_ellipse(med_df['PC1'],med_df['PC2']))



#update axis as scitifics
fig.update_xaxes(tickformat=".1e")
fig.update_yaxes(tickformat=".1e")



fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(size=20))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
#fig.show()

fig.write_image("{}/Save_PCA_trajectory_".format(Trajectory_save) + plot_name + ".png")
fig.write_html("{}/Save_PCA_trajectory_".format(Trajectory_save) + plot_name + ".html")




T2 = time.time()

print('{} Done /n Time taken: {} seconds'.format(plot_name, T2-T1))


