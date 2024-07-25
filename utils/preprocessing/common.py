import numpy as np
import pandas as pd
from typing import Optional, Union
from collections.abc import Iterable
from sklearn.base import BaseEstimator, TransformerMixin



class QuantileImputer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 low_quantile: float, 
                 high_quantile: float, 
                 how='inf',
                 subset=None) -> None:
        """
        Заполнение экстремальных значений квантилями
        :param low_quantile: (float) нижний квантиль для заполнения
        :param high_quantile: (float) верхний квантиль для заполнения
        :param how: вид заполнения: ('inf' - заполнение inf значений, 'quantile' - заполнение значений выше квантилей)
        """

        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.lows, self.highs = None, None
        self.lagg_names = []
        self.subset = subset
        self.how = how
    
    def get_feature_names_out(self, *args, **params):
        return self.subset
        

    def fit(self, X: pd.DataFrame, y=None):
        if self.subset is None:
            self.subset = X.columns
        self.lows = X[self.subset].quantile(self.low_quantile)
        self.highs = X[self.subset].quantile(self.high_quantile)
        return self


    def transform(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if self.how == 'quantile':
            for column in self.subset:
                df.loc[df[column] < self.lows[column], column] = self.lows[column]
                df.loc[df[column] > self.highs[column], column] = self.highs[column]
        elif self.how == 'inf':
            for column in self.subset:
                df.loc[df[column] == -np.inf, column] = self.lows[column]
                df.loc[df[column] == np.inf, column] = self.highs[column]
        return df
    
class LaggBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, lags: Iterable, subset:Optional[Iterable]=None):
        self.lags = lags
        self.subset = subset

    def get_feature_names_out(self, *args, **params):
        return self.X.columns

    def fit(self, X, y=None):
        if self.subset is None:
            self.subset = X.columns
        return self

    def transform(self, X):
        self.lagg_names = []
        X = X.copy(deep=True)
        for lag in self.lags:
            lags = X[self.subset].shift(lag)
            lags.columns = [col + f"_T-{lag}" for col in self.subset]
            X = pd.concat([X, lags], axis=1)
            for name in [col + f"_T-{lag}" for col in self.subset]:
                self.lagg_names.append(name)
        self.X = X
        return X
