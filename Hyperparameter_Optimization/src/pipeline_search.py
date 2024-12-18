import pandas as pd
from sklearn import metrics, model_selection, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quadratic_weighted_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")

train = pd.read_csv('../input/train.csv')
train = train.drop('id', axis=1)
y = train.relevance.values

traindata = list(train.apply(lambda x: '%s %s' % (x['text1'], x['text2']), axis=1))
tfv = TfidfVectorizer(
    min_df=3, 
    max_features=None, 
    strip_accents='unicode', 
    analyzer='word', 
    token_pattern=r'\w{1,}', 
    ngram_range=(1, 3), 
    use_idf=1, 
    smooth_idf=1, 
    sublinear_tf=1, 
    stop_words='english'
)

tfv.fit(traindata)
X = tfv.transform(traindata)

svd = TruncatedSVD()
scl = StandardScaler()
svm_model = SVC()

clf = pipeline.Pipeline([
    ('svd', svd),
    ('scl', scl),
    ('svm', svm_model)
])

param_grid = {
    'svd__n_components': [200, 300],
    'svm__C': [10, 12]
}

kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

model = model_selection.GridSearchCV(
    estimator=clf, 
    param_grid=param_grid, 
    scoring=kappa_scorer, 
    verbose=10, 
    n_jobs=-1, 
    refit=True, 
    cv=5
)

model.fit(X, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

best_model = model.best_estimator_
best_model.fit(X, y)
