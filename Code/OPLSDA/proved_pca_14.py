
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


mac_path = '/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC5502014/'
windows_path = 'C:/Users/theer/OneDrive - Khon Kaen University/KKUPC'


platform_ = mac_path

df = pd.read_csv('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Dataset/KKUPC6602014_dataset_preprocessed.csv')
#Drop QC samples
df = df.drop(df[df['Group'] == 'QC'].index)

#test group
unique_gr = df['Intervention'].unique()

gr1 = df[df['Intervention'] == unique_gr[0]]
gr2 = df[df['Intervention'] == unique_gr[1]]
gr3 = df[df['Intervention'] == unique_gr[2]]
gr4 = df[df['Intervention'] == unique_gr[3]]
gr5 = df[df['Intervention'] == unique_gr[4]]
gr6 = df[df['Intervention'] == unique_gr[5]]
gr7 = df[df['Intervention'] == unique_gr[6]]


#Combind group
c_0 = df
c_1 = pd.concat([gr1,gr2])
c_2 = pd.concat([gr1,gr3])
c_3 = pd.concat([gr1,gr4])
c_4 = pd.concat([gr1,gr5])
c_5 = pd.concat([gr1,gr6])
c_6 = pd.concat([gr1,gr7])

c_8 = pd.concat([gr2,gr3])
c_9 = pd.concat([gr2,gr4])
c_10 = pd.concat([gr2,gr5])
c_11 = pd.concat([gr2,gr6])
c_12 = pd.concat([gr2,gr7])

c_14 = pd.concat([gr3,gr4])
c_15 = pd.concat([gr3,gr5])
c_16 = pd.concat([gr3,gr6])
c_17 = pd.concat([gr3,gr7])

c_19 = pd.concat([gr4,gr5])
c_20 = pd.concat([gr4,gr6])
c_21 = pd.concat([gr4,gr7])

c_23 = pd.concat([gr5,gr6])
c_24 = pd.concat([gr5,gr7])

c_26 = pd.concat([gr6,gr7])


c_list = [c_0, c_1,c_2,c_3,c_4,c_5,c_6,
          c_8,c_9,c_10,c_11,c_12,
          c_14,c_15,c_16,c_17,
          c_19,c_20,c_21,
          c_23,c_24,
          c_26]


# File name
name = ["All", 
        "Gr1 vs Gr2", "Gr1 vs Gr3", "Gr1 vs Gr4", "Gr1 vs Gr5", "Gr1 vs Gr6", "Gr1 vs Gr7",
        "Gr2 vs Gr3", "Gr2 vs Gr4", "Gr2 vs Gr5", "Gr2 vs Gr6", "Gr2 vs Gr7",
        "Gr3 vs Gr4", "Gr3 vs Gr5", "Gr3 vs Gr6", "Gr3 vs Gr7",
        "Gr4 vs Gr5", "Gr4 vs Gr6", "Gr4 vs Gr7",
        "Gr5 vs Gr6", "Gr5 vs Gr7",
        "Gr6 vs Gr7",]

#Make directory
# path folder
report_path = '/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process'

# Create directories if they don't exist
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process/HTML', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process/PNG', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process/Scores', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process/Loading', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process/R2', exist_ok=True)


for i in range(len(c_list)):

    plot_name = name[i]

    # path folder
    PCA_result_path = '/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/PCA_result_process'
    HTML_save = '{}/HTML'.format(PCA_result_path)
    PNG_save = '{}/PNG'.format(PCA_result_path)
    Scores_save = '{}/Scores'.format(PCA_result_path)
    Loading_save = '{}/Loading'.format(PCA_result_path)
    R2_save = '{}/R2'.format(PCA_result_path)

    # Import the datasets from the /data directory
    # X for the NMR spectra and Y for the 2 outcome variables
    test_gr = c_list[i]

    X = test_gr.iloc[:, 43:]
    #fill nan with 0
    X = X.fillna(0)
    Y = test_gr['Intervention']
    Y1 = pd.Categorical(Y).codes
    ppm = list(np.ravel(X.columns).astype(float))
    # Use pandas Categorical type to generate the dummy enconding of the Y vector (0 and 1) 

   

    import time

    from tqdm import tqdm

    T1 = time.time()


     # Select the scaling options: 
    # Here we are generating 3 scaling objects to explore the effect of scaling in PCA:

    # Unit-Variance (UV) scaling:
    
    scale__ = 'UV'
    scale_power_ = 1

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
    df2_scores_.to_csv('{}/PCA_scores_{}.csv'.format(Scores_save, name[i]))

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



    # PCA plot
    pca_label = df2_scores_.index


    fig = px.scatter(df2_scores_, x='PC1', y='PC2',
                    color='Intervention',
                    color_discrete_map={
                                        "Corn oil": "#E91E63",        
                                        "D-galactose": "#FF9800",
                                        "AB extract 500 mg/kg": "#FFEB3B",       
                                        "AP extract 250 mg/kg": "#9C27B0",
                                        "Vitamin E": "#03A9F4",
                                        "AP extract 500 mg/kg": "#4CAF50",        
                                        "AB extract 250 mg/kg": "#B30000",
                                        "0.5% SCMC": "#3F51B5"
                                        }, 
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



    fig.update_traces(marker=dict(size=35))
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

    T2 = time.time()

    print('{} Done /n Time taken: {} seconds'.format(plot_name, T2-T1))
