import pandas as pd

def ds2df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns = X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name = 'target')

    return x, y