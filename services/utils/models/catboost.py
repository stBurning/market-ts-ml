import warnings
from datetime import datetime
from typing import Optional, Literal
import logging
import catboost
import numpy as np
import optuna
import pandas as pd
import shap
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection._split import _BaseKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CatBoostClassifierTuner:

    def __init__(self, 
                 random_state:int=42, 
                 cv: Optional[_BaseKFold] = None,
                 fixed_params: dict = None):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

        self.is_fitted = False  # Состояние модели
        self.random_state: int = random_state  # Инициализатор для случайных чисел
        self.model: Optional[catboost.CatBoostClassifier] = None
        self.cv = cv

        self.fixed_params = fixed_params.copy() if fixed_params is not None else dict()
        self.best_params = fixed_params.copy() if fixed_params is not None else dict()
        self.all_params = ('iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'rsm', 'subsample', 'bagging_temperature', 'od_wait')


    def __suggest(self, trial: optuna.trial.BaseTrial, 
                  param: Literal['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'rsm', 'subsample', 'bagging_temperature', 'od_wait']):
        match param:
            case "iterations":
                return trial.suggest_int("iterations", 10, 100)
            case "learning_rate":
                return trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            case "depth":
                return trial.suggest_int("depth", 2, 3)
            case "l2_leaf_reg":
                return trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True)
            case "random_strength":
                return trial.suggest_float("random_strength", 1e-8, 10.0, log=True)
            case "rsm":
                return trial.suggest_float("rsm", 0.2, 1.0, log=True)
            case "subsample":
                return trial.suggest_float("subsample", 0.05, 1.0)
            case "bagging_temperature":
                return trial.suggest_float("bagging_temperature", 0.0, 10.0)
            case "od_wait":
                return trial.suggest_int("od_wait", 5, 100)


    def tune(self, X_train, y_train, X_val=None, y_val=None, cat_features=None):


        def __objective(trial):
            # Копируем фиксированные гиперпараметры
            self.current_params = self.fixed_params.copy()
            # Сэмплируем гиперпараметры
            for param in self.all_params:
                if param not in self.current_params.keys():
                    self.current_params[param] = self.__suggest(trial, param)

            model = catboost.CatBoostClassifier(**self.current_params, cat_features=cat_features, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            y_predict = model.predict(X_val, prediction_type='Probability')[:, 1]
            return 2 * roc_auc_score(y_val, y_predict) - 1

        def __cv_objective(trial):

            # Копируем фиксированные гиперпараметры
            self.current_params = self.fixed_params.copy()
            # Сэмплируем гиперпараметры
            for param in self.all_params:
                if param not in self.current_params.keys():
                    self.current_params[param] = self.__suggest(trial, param)
           
            scores = []

            for train_index, val_index in self.cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                model = catboost.CatBoostClassifier(**self.current_params, cat_features=cat_features, verbose=False)

                model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold) )
                y_predict = model.predict(X_val_fold, prediction_type='Probability')[:, 1]
                scores.append(2 * roc_auc_score(y_val_fold, y_predict) - 1)
            return np.mean(scores)
        
        
        sampler = TPESampler(seed=42)
        storage_url = "sqlite:///notebooks//optuna-storage//opt.db"
        study_name = f"catboost-{datetime.now()}".replace(" ", "_").replace(".", "_")
        study = optuna.create_study(study_name=study_name,
                                    storage=storage_url,
                                    direction="maximize",
                                    sampler=sampler)
        
        if (X_val is None) or (y_val is None):
            study.optimize(__cv_objective, n_trials=100, show_progress_bar=True)
        else:
            study.optimize(__objective, n_trials=100, show_progress_bar=True)
        
        best_trial = study.best_trial
        self.logger.info(f"[score: {best_trial.value}] {best_trial.params}")
        self.best_params = best_trial.params
        self.model = catboost.CatBoostClassifier(**self.best_params, cat_features=cat_features, verbose=False)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val) if X_val is not None else None)
        self.is_fitted = True
        

    def explain(self, X: pd.DataFrame, max_display=10):
        """
        Shap-values explain
        :param X: данные для предсказания
        :param max_display: максимальной кол-во для вывода
        :return:
        """
        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values, max_display=max_display)

    def feature_importance(self):
        """
        Получение значимости факторов
        :return: (pd.Series) оценки feature_importance из catboost
        """
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_)

    # def save(self, path: str):
    #     self.model.save_model(path)

    # def load_from(self, path: str):
    #     pass
