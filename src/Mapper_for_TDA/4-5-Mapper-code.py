import sklearn
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
import webbrowser
import pandas as pd

if __name__ == "__main__":

    metricFile = "C:/Code/Mapper/statistics.csv"

    try:
        dataset = pd.read_csv(metricFile, sep=",")

        X = pd.read_csv(metricFile, sep=",")

        yfilt = X.dataset

        Xfilt = X.drop(columns=['dataset'])
        mapper = km.KeplerMapper()
        scaler = MinMaxScaler(feature_range=(0, 1))

        Xfilt = scaler.fit_transform(Xfilt)
        lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
        cls = 5  # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.

        graph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
            cover=km.Cover(n_cubes=10, perc_overlap=0.6)  # TODO: Playing with this parameter
        )
        print(str(len(yfilt)) + " " + str(len(Xfilt)))
        html = mapper.visualize(graph,
                         title="Graph Analysis",
                         custom_tooltips=yfilt,
                         path_html="C:/Code/Mapper/stats.html",

                         color_values=(yfilt.values.reshape(-1, 1)),)


        print("Mapper finished")
    except Exception as e:
        print(str(e))

webbrowser.open('stats.html')