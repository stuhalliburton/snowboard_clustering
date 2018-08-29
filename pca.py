import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from lib.data_formatter import create_dataset, data_clusters
from lib.plotter import plot_clusters

scaler = MinMaxScaler()
X, y = create_dataset(scaler=scaler)

pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(X))

powder, park, all_mountain = data_clusters(transformed, y)

plot_clusters(powder, park, all_mountain)
