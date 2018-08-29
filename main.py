import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

FILE_PATH = './data/board-specs.csv'

dataset = pd.read_csv(FILE_PATH, index_col=False)

feature_columns = ['length', 'set back', 'waist width', 'min weight', 'max weight',
        'sidecut', 'effective edge', 'nose width', 'tail width', 'stance width']

X = dataset[feature_columns]
y = dataset['type']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# PCA 3 Dimensions
# pca = PCA(n_components=3)
# transformed = pd.DataFrame(pca.fit_transform(X))
#
# powder = transformed[y=='powder']
# park = transformed[y=='park']
# all_mountain = transformed[y=='all mountain']
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(powder[0], powder[1], powder[2], label='Powder', c='red')
# ax.scatter(park[0], park[1], park[2], label='Park', c='blue')
# ax.scatter(all_mountain[0], all_mountain[1], all_mountain[2], label='All Mountain', c='lightgreen')

# LDA 2 Dimensions
lda = LDA(n_components=2)
transformed = pd.DataFrame(lda.fit_transform(X, y))

powder = transformed[y=='powder']
park = transformed[y=='park']
all_mountain = transformed[y=='all mountain']

plt.scatter(powder[0], powder[1], label='Powder', c='red')
plt.scatter(park[0], park[1], label='Park', c='blue')
plt.scatter(all_mountain[0], all_mountain[1], label='All Mountain', c='lightgreen')

# Show plot
plt.legend()
plt.show()
