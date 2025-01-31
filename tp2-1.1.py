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
from R_square_clustering import r_square
from scipy.cluster import hierarchy

data = pd.read_csv("eurostat/eurostat-2013.csv")

#diviser les valeurs de la colonne('tsc00004 (2012)') par la population
data['tsc00004 (2012)'] = data['tsc00004 (2012)'].divide(data['tps00001 (2013)'])
data['tet00002 (2013)'] = data['tet00002 (2013)'].divide(data['tps00001 (2013)'])

#Supprimer les données correspondant à la population.
data =data.drop(['tps00001 (2013)'], axis=1)

standardscaler = preprocessing.StandardScaler()

X = data.iloc[:,2:11]

Y= ["axe1","axe2","axe3","axe4","axe5","axe6","axe7","axe8","axe9"]

X_norm = standardscaler.fit_transform(X)

pca = PCA()

X_pca =pca.fit_transform(X_norm)

axes = pd.DataFrame(data=X_pca, columns = Y)


cmap = cm.get_cmap('gnuplot')
df = pd.concat([axes, data[['Code']]], axis = 1)

'''
   l'affichage des instances étiquetées par le code du pays suivant les 2 facteurs
principaux de l'ACP, puis suivant les facteurs 3 et 4 de l'ACP.
'''
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

correlation_circle(data,9,2,3)

#question 5
lst_k=range(2,8)
lst_rsq = []
for k in lst_k:
    est=KMeans(n_clusters=k)
    est.fit (X_norm)
    lst_rsq.append(r_square(X_norm, est.cluster_centers_,est.labels_,k))
fig = plt. figure ()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ')
plt.title ('The Elbow Method showing the optimal k')
plt.savefig('r_square')
plt.close(fig)


est = KMeans(n_clusters=5)

est.fit(X)

# print centroids associated with several countries
lst_countries=['EL','FR','DE','US']
# centroid of the entire dataset
# est: KMeans model fit to the dataset
y=data['Code']

print (est.cluster_centers_)
for name in lst_countries:
    num_cluster = est.labels_[y.loc[y==name].index][0]
    print ('Num cluster for '+name+': '+str(num_cluster))
    print ('\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values))
    print ('\tcentroid: '+str(est.cluster_centers_[num_cluster]))


Z = hierarchy.linkage(X,'ward')
lst_labels = map(lambda pair: pair[0], zip( data['Code'].values, data.index))
plt.figure()
dn = hierarchy.dendrogram(Z,color_threshold=0,labels=lst_labels)
plt.savefig ('dendogramme')
plt.close( fig )