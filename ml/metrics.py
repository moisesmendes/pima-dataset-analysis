__author__ = 'Moises Mendes'
__version__ = '0.1.0'
__all__ = [
    'classification_metrics',
    'multiple_run_metrics',
]

import typing as tp

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from preprocessing import ARR, SR

NDIGITS = 3


def classification_metrics(y_true: ARR, y_pred: ARR) -> tp.Tuple[float, float, float, float]:
    """Get four main classification metrics: accuracy, precision, recall, f1-score.
    
    :param y_true: Expected values.
    :type y_true: ``numpy.ndarry``
    :param y_pred: Predicted values.
    :type y_pred: ``numpy.ndarry``
    :return: Accuracy, precision, recall and f1-score values.
    :rtype: ``tuple`` of ``float``
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    return (accuracy.round(NDIGITS), precision.round(NDIGITS),
            recall.round(NDIGITS), fscore.round(NDIGITS))


def multiple_run_metrics(predictions: tp.List[tp.Tuple[ARR, ARR]]) -> tp.Dict[str, SR]:
    """Calculate mean and standard deviation of metrics for multiple runs of model.
    
    :param predictions: Pair of y_true and y_pred for each prediction.
    :type predictions: ``list`` of ``tuple`` of ``numpy.ndarry``
    :return: Mean and standard deviation of metrics.
    :rtype: ``dict`` of ``str`` to ``pandas.Series``
    """
    metrics = [classification_metrics(y_true=y_true, y_pred=y_pred) for y_true, y_pred in predictions]
    df_metrics = pd.DataFrame(data=metrics, columns=['accuracy', 'precision', 'recall', 'fscore'])
    return {
        'mean': df_metrics.mean(axis=0),
        'std': df_metrics.std(axis=0)
    }
