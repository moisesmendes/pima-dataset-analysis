__author__ = 'Moises Mendes'
__version__ = '0.2.0'
__all__ = [
    'randomize_data',
    'splip_x_y',
    'generate_train_test',
    'scale_data',
    'DF',
    'SR',
    'ARR'
]

import typing as tp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DF = pd.DataFrame
SR = pd.Series
ARR = np.ndarray


def randomize_data(df: DF) -> DF:
    """Randomly resample dataframe.

    :param df: Dataset with predictor and target variables.
    :type df: ``pandas.DataFrame``
    :return: Randomized dataset.
    :rtype: ``pandas.DataFrame``
    """
    return df.sample(frac=1).reset_index(drop=True)


def splip_x_y(df: DF, target_col: str) -> tp.Tuple[DF, SR]:
    """Split dataset into predict variables and target variable.
    
    :param df: Dataset with all columns
    :type df: ``pandas.DataFrame``
    :param target_col:
    :type target_col: ``str``
    :return: Tuple with separated variables [x, y].
    :rtype: ``tuple`` of [``pandas.Dataframe``, ``pandas.Series``]
    """
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    return x, y


def generate_train_test(x: DF, y: SR, test_size: float, stratify: bool) -> tp.List:
    """Split data into train and test with optional stratifying.
    
    :param x: Predictor variables.
    :type x: ``pandas.Dataframe``
    :param y: Target variable.
    :type y: ``pandas.Series``
    :param test_size: Proportion of data used for test.
    :type test_size: ``float``
    :param stratify: Flag to perform stratification (mantain classes proportions).
    :type stratify: ``bool``
    :return: List with splitted data [x_train, x_test, y_train, y_test].
    :rtype: ``list``
    """
    if stratify:
        train_test = train_test_split(x, y, test_size=test_size, stratify=y)
    else:
        train_test = train_test_split(x, y, test_size=test_size)
    
    return train_test


def scale_data(x_train: DF, x_test: DF, scaler: object = None) -> tp.Tuple[ARR, ARR]:
    """Scale train and test data using specified scaler (default StandardScaler).
    
    :param x_train: Train data.
    :type x_train: ``pandas.Dataframe``
    :param x_test: Test data.
    :type x_test: ``pandas.Dataframe``
    :param scaler: Scaler object with fit and transform methods. If None, uses StandardScaler from scikit-learn.
    :type scaler: ``object``
    :return: Scaled train and test data.
    :rtype: ``tuple`` of [``numpy.ndarray``, ``numpy.ndarray``]
    """
    if not scaler:
        scaler = StandardScaler()
    
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)
