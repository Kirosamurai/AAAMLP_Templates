from sklearn.feature_selection import VarianceThreshold

data = ...

var_thresh = VarianceThreshold(0.1)
transformed_data = var_thresh.fit_transform(data)

