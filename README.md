# Autogluon for Kaggle competition

Reference: https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-kaggle.html

## pip install

```
!pip install autogluon.tabular[all]
#!pip install ray==2.0.0
```

## predictor fit and redict

```
from autogluon.tabular import TabularDataset, TabularPredictor

directory = '/kaggle/input/directory_name/'

train = TabularDataset(directory + 'train.csv')
test = TabularDataset(directory + 'test.csv')

label = 'target'
time_limit = 3600
eval_metric = 'rmse'

predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train, presets=['best_quality'], time_limit=time_limit)

submission = pd.read_csv(directory + 'sample_submission.csv')
submission[label] = predictor.predict(test)
submission.to_csv('submission.csv', index=False)
submission.head()
```

## eval_metric

* options for classification:
`[‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, ‘roc_auc’, ‘roc_auc_ovo_macro’, ‘average_precision’, ‘precision’, ‘precision_macro’, ‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’, ‘recall_micro’, ‘recall_weighted’, ‘log_loss’, ‘pac_score’]`

* Options for regression:
`[‘root_mean_squared_error’, ‘mean_squared_error’, ‘mean_absolute_error’, ‘median_absolute_error’, ‘r2’]`

* Options for quantile regression:
`[‘pinball_loss’]`
