from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from ml_pipeline.dataset import create_windows, preprocess
from ml_pipeline.featureset  import make_summary_stats_input

def make_binary_target(y):
    # convert y to binary
    return y.apply(lambda x: 0 if x==0 else 1)


# wrapper function for each of creation
def processing_pipeline(data,
            window_kwargs = {'window_length':30, 'step_length':1, 'fh':0},
            feature_func = make_summary_stats_input,
            add_first_diffs = True):

        # segment data into windows
        X, y = create_windows(data,
                                target_column='Status', 
                                window_kwargs = window_kwargs)

        # outlier and missing value filling
        X = [preprocess(x, add_first_diff=add_first_diffs) for x in X]

        # feature function
        X, y = feature_func(X, y)

        # make y binary 
        y = make_binary_target(y)

        return X, y


def train_traditional_model(
                X, y,
                scaler = StandardScaler(),
                estimator =  LogisticRegression(),
                ):
    # X, y should be training data

    # create pipeline using our chosen scaler and estimator
    model = Pipeline([('scaler', scaler), 
                    ('estimator', estimator)])

    # fit pipeline
    model.fit(X, y)

    return model



# this function trains a series classifier
def train_series_model(X, y, 
    model = KNeighborsTimeSeriesClassifier(n_neighbors=1)):

    # fit to training date
    model.fit(X, y)

    return model


def train_multiseries_model(X, y, 
        estimator=KNeighborsTimeSeriesClassifier(n_neighbors=1)):

    # get number of time series (columns) in X
    n_series = X.shape[1]

    # we make a list of estimators, one for each column
    # these format here is (name, estimator, column_number)
    estimators = [(f"knn_{i}", estimator, [i]) for i in range(n_series)]

    # column ensemble clasifier is a wrapper class to make this work
    model = ColumnEnsembleClassifier(estimators=estimators)

    # fit each estimator to its respective colun
    model.fit(X, y)

    return model



def test_model(X, y, model, threshold=0.5):
    # X, y should be testing data

    # create table to store results
    result = pd.DataFrame(index = range(len(y)),
                        columns = ['y_true', 'y_pred', 'residual'])

    # get probability estimate
    probs = model.predict_proba(X)
    probs = pd.DataFrame(probs, 
                columns = [f'class_{i}' for i in range(probs.shape[1])])
    result = result.join(probs)

    # store ground truth
    result['y_true'] = y

    # compute predictions for test
    result['y_pred'] = (result['class_1'] > threshold).astype(int)

    # add residuals
    result['residual'] = result.y_true - result.y_pred

    return result


def plot_model_result(result, crop=None):

    # if crop is not None, it should be a tuple e.g. (10, 100)
    if crop is not None:
        result = result.iloc[crop[0]:crop[1]]

    # we plot correct and incorrect answers separately
    correct = result[result['residual']==0]
    incorrect = result[result['residual']!=0]

    # plot correct, then for incorrect show both true and pred
    fig, ax = plt.subplots(figsize=(12,3))
    correct['y_true'].plot(ax=ax, marker='x', linewidth=0, color='k', label='correct')
    incorrect['y_true'].plot(ax=ax, marker='o', linewidth=0, color='b', label='incorrect (true)')
    incorrect['y_pred'].plot(ax=ax, marker='s', linewidth=0, color='r', label='incorrect (pred)')
    ax.legend()

    ax.set_ylabel('Target')
    ax.set_xlabel('Time interval')



def create_explainer_dashboard(model, X, y, save_path=None):

    # create explainer
    explainer = ClassifierExplainer(model, X, y)

    # create dashboard (shap disabled for time saving)
    dashboard = ExplainerDashboard(
            explainer,
            importances=False,
            model_summary=True,
            contributions=False,
            whatif=False,
            shap_dependence=False,
            shap_interaction=False,
            decision_trees=False)

    # if no save path we just run it            
    if save_path is None:
        dashboard.run()
    # else we save it
    else:
        dashboard.save_html(save_path)
        print(f'dashboard saved: {save_path}')

    # this will now host locally and can be saved as a HTML file