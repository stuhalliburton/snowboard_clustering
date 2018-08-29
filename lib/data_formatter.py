import pandas as pd

FILE_PATH = './data/board-specs.csv'
FEATURE_COLUMNS = ['length', 'set back', 'waist width', 'min weight', 'max weight',
        'sidecut', 'effective edge', 'nose width', 'tail width', 'stance width']
LABEL_COLUMN = 'type'

def create_dataset(scaler=None):
    dataset = pd.read_csv(FILE_PATH, index_col=False)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[LABEL_COLUMN]

    if scaler:
        X = scaler.fit_transform(X)

    return X, y

def data_clusters(data, labels):
    powder = data[labels=='powder']
    park = data[labels=='park']
    all_mountain = data[labels=='all mountain']

    return powder, park, all_mountain
