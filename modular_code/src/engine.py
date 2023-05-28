# importing required libraries
from ml_pipeline.dataset import (load_excel,
                                time_train_test_split)
from ml_pipeline.model import (processing_pipeline, 
                    train_traditional_model, 
                    train_multiseries_model,
                    test_model, 
                    create_explainer_dashboard)
from ml_pipeline.featureset import (make_series_input, 
                    make_summary_stats_input)
import pickle

# select type of model to train
ft_func = make_summary_stats_input
md_func = train_traditional_model

# load data
DATA_PATH = '../input/elevator_failure_prediction.xlsx'
df = load_excel(DATA_PATH, select_sheet='data')

# make train test split
train, test = time_train_test_split(df, proportion = 14/31)
train.to_csv('../input/train/data.csv')
test.to_csv('../input/test/data.csv')

# create X, y objects for training and testing
wkw = {'window_length':30, 'step_length':1, 'fh':5}
X_train, y_train = processing_pipeline(train[-3100:-2900],
            window_kwargs = wkw,
            feature_func = ft_func)

X_test, y_test = processing_pipeline(test[4200:4400],
            window_kwargs = wkw,
            feature_func = ft_func)

# train model and save it in output folder
model = md_func(X_train, y_train)
with open('../output/model.pickle', 'wb') as f:
    pickle.dump(model, f)

# save result 
test_model(X_test, y_test, model, threshold=0.95).to_csv('../output/results.csv')

# create explainer dashboard (only works for traditional models)
if md_func == train_traditional_model:
    create_explainer_dashboard(model, X_test, y_test, save_path='../output/dashboard.html')
