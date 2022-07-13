import kmapper as km
import pandas as pd
import sklearn
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

statistics = "C:/Code/src/Mapper_for_TDA/statistics.csv"
merged_df = pd.read_csv(statistics, sep=",")
merged_df = merged_df.drop(["Index", "clustering_coeff", "graph_diameter", "motif1", "motif2", "cliques", "components"], axis=1) #drop the index column from the dataframe
df2 = merged_df.groupby(['dataset']).apply(lambda grp: grp.sample(n=187)) #mutag has 188 graphs

y = df2[['dataset']] #select dataset column
M = df2[df2.columns[1:]].apply(pd.to_numeric)
M = M.drop(["graphlabel"], axis=1)

Xfilt = M
cls = len(pd.unique(merged_df.iloc[:,0])) #select unique elements of the first column
mapper = km.KeplerMapper()
scaler = MinMaxScaler(feature_range=(0, 1))
print(list(M.columns))
Xfilt = scaler.fit_transform(Xfilt)
lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE(verbose=1)) #dimensionality reduction of Xfilt
print(" mapper started with "+str(len(pd.DataFrame(Xfilt).index))+" data points,"+str(cls)+" clusters")

graph = mapper.map(
    lens,
    Xfilt,
    clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
    cover=km.Cover(n_cubes=10, perc_overlap=0.6)
) # Create dictionary called 'graph' with nodes, edges and meta-information
print("mapper ended")
print(str(len(y))+" "+str(len(Xfilt)))

df2['dataset_graphlabel'] = df2['dataset'] + "-" + df2['graphlabel'].astype(str)
df2 = df2.drop(["dataset","graphlabel"], axis=1)
y_visual = df2.dataset_graphlabel

html = mapper.visualize(
    graph,
    path_html="C:/Code/src/Mapper_for_TDA/mapperunsampled.html",
    title="mapper data",
    custom_tooltips=y_visual) # Visualize the graph

#webbrowser.open('mapper.html')