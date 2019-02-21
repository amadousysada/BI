import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
data = pd.read_csv("eurostat\eurostat-2013.csv")
print (data.head())

print (data.shape)

print (data.dtypes)

print (data.groupby('tps00001 (2013)').size())
#fruits=fruits.astype(float)
#my_scatter = pd.plotting.scatter_matrix(fruits, diagonal="kde")

# Save the figure (this can also be a path). As it stands now it will save in this codes directory.
#plt.savefig(r"figure_1.png")
#print (fruits[['fruit_name', 'fruit_subtype', 'mass', 'color_score']])

cmap = cm.get_cmap('gnuplot')
pd.plotting.scatter_matrix(
        data[['teilm (F dec 2013)', 'teilm (M dec 2013)', 'tec00118 (2013)',
       'teimf00118 (dec 2013)', 'tsdsc260(2013)', 'tet00002 (2013)',
       'tsc00001 (2011)', 'tsc00004 (2012)']],
        c= data['tps00001 (2013)'],
        marker = 'o',
        s=40,
        hist_kwds={'bins':15},
        figsize=(9,9),
        cmap = cmap)

for attr in ['tps00001 (2013)', 'tec00115 (2013)','teilm (F dec 2013)', 'teilm (M dec 2013)', 'tec00118 (2013)','teimf00118 (dec 2013)', 'tsdsc260(2013)', 'tet00002 (2013)','tsc00001 (2011)', 'tsc00004 (2012)']:
	pd.DataFrame({k: v for k, v in data .groupby('tps00001 (2013)')[attr]}).plot. hist(stacked=True)
	plt . suptitle (attr)
	plt . savefig ('fruits_histogram_'+attr)
