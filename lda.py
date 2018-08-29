import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lib.data_formatter import create_dataset, data_clusters
from lib.plotter import plot_clusters

scaler = MinMaxScaler()
X, y = create_dataset(scaler=scaler)

lda = LDA(n_components=2)
transformed = pd.DataFrame(lda.fit_transform(X, y))

powder, park, all_mountain = data_clusters(transformed, y)

plot_clusters(powder, park, all_mountain)
