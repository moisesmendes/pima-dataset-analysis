__author__ = 'Moises Mendes'
__version__ = '0.1.0'
__all__ = [
    'Classifier'
]

import typing as tp

from ml.preprocessing import randomize_data, splip_x_y, generate_train_test, scale_data
from ml.preprocessing import DF, SR, ARR


class Classifier:
    def __init__(self, model: object):
        """Creates a Classifier object.
        
        :param model: Classifier model that implements fit and predict methods.
        :type model: ``object``
        """
        self.model = model
    
    def single_run(self, df: DF, target_col: str, test_size: float, stratify: bool = True,
                   scaler: object = None) -> tp.Dict[str, tp.Tuple[ARR, ARR]]:
        """Run full pipeline of training phase for dataset using classifier.
        
        :param df: Dataset with all variables.
        :type df: ``pandas.DataFrame``
        :param target_col: Column indicating target variable.
        :type target_col: ``str``
        :param test_size: Ratio of samples reserved for test phase.
        :type test_size: ``float``
        :param stratify: Flag indicating if keeping data stratification (proportion of classes).
        :type stratify: ``bool``
        :param scaler: Scaler object. If None, uses scikit-learn StandardScaler.
        :type scaler: ``object``
        :return: True values and predicted values for train and test data.
        :rtype: ``dict`` of ``str`` to ``tupe`` of ``numpy.ndarry``
        """
        df = randomize_data(df=df)
        x, y = splip_x_y(df=df, target_col=target_col)
        x_train, x_test, y_train, y_test = generate_train_test(x=x, y=y, test_size=test_size, stratify=stratify)
        x_train_sc, x_test_sc = scale_data(x_train=x_train, x_test=x_test, scaler=scaler)
        self.model.fit(x_train_sc, y_train)
        
        return {
            'train': (y_train, self.model.predict(x_train_sc)),
            'test': (y_test, self.model.predict(x_test_sc))
        }
    
    def multiple_runs(self, n: int, **kwargs: tp.Any) -> tp.Dict[str, tp.List[tp.Tuple[ARR, ARR]]]:
        """Run classification process multiple times on dataset.
        
        :param n: Number of times to run.
        :type n: ``int``
        :param kwargs: Arguments passed to single_run method.
        :type kwargs: ``any``
        :return: True and predicted values for each run.
        :rtype: ``dict`` of ``str`` to ``list`` of ``tuple`` of ``numpy.array``
        """
        predictions = [self.single_run(kwargs) for i in range(n)]
        return {
            'train': [pred['train'] for pred in predictions],
            'test': [pred['test'] for pred in predictions]
        }
