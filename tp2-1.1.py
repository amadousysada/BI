#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn. cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("eurostat/eurostat-2013.csv")

#diviser les valeurs de la colonne('tsc00004 (2012)') par la population
data['tsc00004 (2012)'] = data['tsc00004 (2012)'].divide(data['tps00001 (2013)'])

#Supprimer les données correspondant à la population.
data =data.drop(['tps00001 (2013)'], axis=1)

standardscaler = preprocessing.StandardScaler()

X = data.iloc[:,2:11]
#y = (0.3-np.mean(data['tec00115 (2013)']))/np.std(data['tec00115 (2013)'][0])
Y= ["axe1","axe2","axe3","axe4","axe5","axe6","axe7","axe8","axe9"]

X_norm = standardscaler.fit_transform(X)

pca = PCA()

X_pca =pca.fit_transform(X_norm)

axes = pd.DataFrame(data=X_pca, columns = Y)


cmap = cm.get_cmap('gnuplot')
df = pd.concat([axes, data[['Code']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('axe1', fontsize = 15)
ax.set_ylabel('axe2', fontsize = 15)
ax.set_title('2 first components PCA', fontsize = 20)
targets = df['Code'].values
for target in targets:
    
    indicesToKeep = df['Code'] == target
    plt.annotate(target,
                     xy=(
                         df.loc[indicesToKeep, 'axe1'],
                         df.loc[indicesToKeep, 'axe2']
                     )
                )
    ax.scatter(df.loc[indicesToKeep, 'axe1']
               , df.loc[indicesToKeep, 'axe2']
              , cmap = cmap
               , s = 50)
ax.grid()
plt.savefig("2 facteurs principaux de l'ACP")
plt.close()

#teilm (M dec 2013) --> tec00118 (2013)
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1) 

ax1.set_xlabel('axe3', fontsize = 15)
ax1.set_ylabel('axe4', fontsize = 15)
ax1.set_title('3th and 4th components PCA', fontsize = 20)

for target in targets:
    
    indicesToKeep = df['Code'] == target
    plt.annotate(target,
                     xy=(
                         df.loc[indicesToKeep, 'axe3'],
                         df.loc[indicesToKeep, 'axe4']
                     )
                )
    ax1.scatter(df.loc[indicesToKeep, 'axe3']
               , df.loc[indicesToKeep, 'axe4']
              , cmap = cmap
               , s = 50)
ax1.grid()
plt.savefig("3iem et 4iem facteurs de l'ACP")
plt.close()

n = np.size(X_norm, 0)
p = np.size(X_norm, 1)
eigval = float(n-1)/n*pca.explained_variance_

sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((p,p))
for k in range(p):
    corvar [:, k] = pca.components_[k,:]*sqrt_eigval[k]
    print(corvar)
#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

correlation_circle(data,9,0,1)

# Plot clusters
lst_kmeans = [KMeans(n_clusters=n) for n in range(3,6)]
titles = [str(x)+' clusters 'for x in range(3,6)]
fignum = 1
scaler = MinMaxScaler()

X_norm = scaler.fit_transform(X)

'''for kmeans in lst_kmeans:
    fig = plt. figure (fignum, figsize =(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    kmeans.fit(X_norm)
    labels = kmeans.labels_
    ax.scatter(X,c=labels.astype(np.float), edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels ([])
    ax.set_xlabel('mass')
    ax.set_ylabel('width')
    ax.set_title ( titles [fignum - 1])
    ax.dist = 12
    plt.savefig ('k-means_'+str(2+fignum)+'_clusters')
    fignum = fignum + 1
    plt.close( fig )
'''
# print centroids associated with several countries
lst_countries=['EL','FR','DE','US']
# centroid of the entire dataset
# est: KMeans model fit to the dataset
'''print (est.cluster_centers_)
for name in lst_countries:
    num_cluster = est.labels_[y.loc[y==name].index][0]
    print ('Num cluster for '+name+': '+str(num_cluster))
    print ('\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values))
    print ('\tcentroid: '+str(est.cluster_centers_[num_cluster]))'''

