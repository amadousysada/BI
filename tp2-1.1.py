#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

data = pd.read_csv("eurostat\eurostat-2013.csv")

#diviser les valeurs de la colonne('tsc00004 (2012)') par la population
data['tsc00004 (2012)'] = data['tsc00004 (2012)'].apply(lambda x: x/data.loc[lambda df: df['tsc00004 (2012)']==38637]["tps00001 (2013)"])

#Supprimer les données correspondant à la population.
data =data.drop(['tps00001 (2013)'], axis=1)

standardscaler = preprocessing.StandardScaler()

X = data.iloc[:,2:11]
#y = (0.3-np.mean(data['tec00115 (2013)']))/np.std(data['tec00115 (2013)'][0])
Y= X.columns

X_norm = standardscaler.fit_transform(X)

pca = PCA()

X_pca =pca.fit_transform(X_norm)

pca1_pca2 = pd.DataFrame(data=X_pca, columns = Y)

from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
df = pd.concat([pca1_pca2, data[['Code']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('tec00115 (2013)', fontsize = 15)
ax.set_ylabel('teilm (F dec 2013)', fontsize = 15)
ax.set_title('2 first components PCA', fontsize = 20)
targets = finalDf['Code'].values
colors = ['r', 'g', 'b']
for target in targets:
    
    indicesToKeep = df['Code'] == target
    plt.annotate(target,
                     xy=(
                         df.loc[indicesToKeep, 'tec00115 (2013)'],
                         df.loc[indicesToKeep, 'teilm (F dec 2013)']
                     )
                )
    ax.scatter(df.loc[indicesToKeep, 'tec00115 (2013)']
               , df.loc[indicesToKeep, 'teilm (F dec 2013)']
              , cmap = cmap
               , s = 50)
ax.legend(targets)
ax.grid()
plt.savefig("2 facteurs principaux de l'ACP")
plt.close()

#teilm (M dec 2013)	tec00118 (2013)
fig = plt.figure(figsize = (8,8))
ax.set_xlabel('teilm (M dec 2013)', fontsize = 15)
ax.set_ylabel('tec00118 (2013)', fontsize = 15)
ax.set_title('3th and 4th components PCA', fontsize = 20)
targets = finalDf['Code'].values
for target in targets:
    
    indicesToKeep = df['Code'] == target
    plt.annotate(target,
                     xy=(
                         df.loc[indicesToKeep, 'tec00118 (2013)'],
                         df.loc[indicesToKeep, 'teilm (M dec 2013)']
                     )
                )
    ax.scatter(df.loc[indicesToKeep, 'tec00118 (2013)']
               , df.loc[indicesToKeep, 'teilm (M dec 2013)']
              , cmap = cmap
               , s = 50)
ax.legend(targets)
ax.grid()
plt.savefig("3iem et 4iem facteurs de l'ACP")
plt.close()
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



# print centroids associated with several countries
lst_countries=['EL','FR','DE','US']
# centroid of the entire dataset
# est: KMeans model fit to the dataset
print (est.cluster_centers_)
for name in lst_countries:
    num_cluster = est.labels_[y.loc[y==name].index][0]
    print ('Num cluster for '+name+': '+str(num_cluster))
    print ('\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values))
    print ('\tcentroid: '+str(est.cluster_centers_[num_cluster]))

