import pandas as pd

def dset_2_dframe( data_loader ):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame( X_data, columns = X_columns )

    y_data = data_loader.target
    y = pd.Series( y_data, name = 'target')

    return x, y

def remove_Categories( x, y ):
  x = x.drop(x.select_dtypes('category'),axis=1).copy()

  if y.dtype == 'category':
    category_1 = y.dtype.categories[0]
    category_2 = y.dtype.categories[1]
    y = y.replace([category_1, category_2],[0, 1])
    print('Target 1: %s , Target 2: %s\nAre now 0 and 1 respectively' % (category_1, category_2))
  
  return x, y