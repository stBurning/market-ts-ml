import pickle
import warnings
from datetime import datetime
from typing import Optional, Union
import logging
from logging import getLogger, basicConfig, DEBUG
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import torch
import tqdm.notebook as tq
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Логирование
# logger = logging.getLogger(__name__)
# FORMAT = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
# basicConfig(level=DEBUG, format=FORMAT)


def init_device(verbose=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        if verbose:
            print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


class WindowDataset(Dataset):
    """
    Оконный датасет для рекуррентных моделей

    :param X: данные (предикторы)
    :param y: данные (целевая переменная)
    :param window_size: размер окна
    :param trailing_size: шаг скольжения
    :param device: устройство
    """

    def __init__(self, X, y=None, window_size=1, trailing_size=1, device=init_device()):
        self.device = device
        self.window_size = window_size  # Размер окна
        self.trailing_size = trailing_size  # Размер тени
        self.X, self.y = self.__window_split(X, y)

    def __window_split(self, X, y):
        raw_x = X.values
        raw_y = y.values if y is not None else np.array([np.nan] * X.shape[0])
        x, y = [], []
        for i in range(raw_x.shape[0] - self.window_size - self.trailing_size + 1):
            x += [raw_x[i:i + self.window_size]]
            y += [raw_y[i + self.window_size:i + self.window_size + self.trailing_size]]
        return (torch.from_numpy(np.array(x, dtype=float)).type(torch.float32).to(self.device),
                torch.from_numpy(np.array(y, dtype=float)).type(torch.float32).to(self.device))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


class BaseRNNModel(nn.Module):
    """
    Модуль RNN-модели
    :param input_dim: размерность входных данных
    :param hidden_dim: размерность скрытого слоя
    :param num_layers: кол-во рекуррентных блоков
    :param output_dim: размерность выходных данных
    :param dropout_rate: вероятность прореживания
    :param activation: функция активации
    :param model_type: тип модели (lstm, gru)
    :param device: устройство
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout_rate: float = 0,
                 activation=nn.Sigmoid(),
                 model_type: str = "lstm",
                 device=init_device()):

        super(BaseRNNModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation.to(device)
        self.model_type = model_type
        match self.model_type:
            case "lstm":
                self.rnn = nn.LSTM(input_dim,
                                   hidden_dim,
                                   num_layers,
                                   dropout=dropout_rate,
                                   batch_first=True,
                                   device=device)
            case "gru":
                self.rnn = nn.GRU(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  dropout=dropout_rate,
                                  batch_first=True,
                                  device=device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        if self.model_type == 'lstm':
            out, (_, _) = self.rnn(x, (h0.detach(), c0.detach()))
        else:
            out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out


class RNNTuner:
    """
        Класс для автоматической оптимизации гиперпараметров для RNN-моделей
        :param input_dim: размерность входных данных
        :param output_dim: размерность выходных данных
        :param hidden_dim: размерность скрытого состояния модели
        :param num_layers: кол-во рекуррентных слоев
        :param num_epochs: кол-во итераций обучения
        :param dropout_rate: вероятность прореживания
        :param learning_rate: скорость обучения
        :param lookback: размер окна
        :param batch_size: размер батча
        :param criterion: функция ошибки для обучения модели
        :param activation: функция активации модели
        :param device: устройство
        :param verbosity: уровень детальности
        :param model_name: имя модели
        :param model_type: вид модели (`lstm` или `gru`)
        :param metric: критерий оптимизации, зависит от `criterion`
        :param direction: направление для оптимизации (`maximize`, `minimize`)
        :param storage_url: путь для хранения результатов экспериментов
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 num_epochs: Optional[int] = None,
                 dropout_rate: Optional[float] = None,
                 learning_rate: Optional[float] = None,
                 lookback: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 criterion=torch.nn.BCELoss(reduction='mean'),
                 activation=nn.Sigmoid(),
                 device=init_device(),
                 verbosity: bool = True,
                 model_name: str = "AutoRNN",
                 model_type="lstm",
                 metric=roc_auc_score,
                 direction="maximize",
                 storage_url="sqlite:///optuna-storage//opt.db"):

        self.best_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_epochs": num_epochs,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "lookback": lookback,
            "batch_size": batch_size
        }
        self.current_params = {}

        self.criterion = criterion
        self.activation = activation
        self.device = device
        self.model_type = model_type
        self.model_name = model_name
        self.verbosity = verbosity
        self.metric = metric
        self.direction = direction
        self.storage_url = storage_url

    def __suggest(self, trial: optuna.trial.BaseTrial, param: str):
        match param:
            case "hidden_dim":
                return trial.suggest_categorical("hidden_dim", [32, 64, 128])
            case "num_layers":
                return trial.suggest_int("num_layers", 1, 4)
            case "num_epochs":
                return trial.suggest_int("num_epochs", 10, 120)
            case "dropout_rate":
                if self.current_params['num_layers'] > 1:
                    return trial.suggest_float("dropout_rate", 0.0, 0.5, log=False)
                elif self.current_params['num_layers'] is None:
                    warnings.warn('Define num_layers before suggesting dropout_rate.')
                    return trial.suggest_float("dropout_rate", 0.0, 0.5, log=False)
                else:
                    return 0
            case "learning_rate":
                return trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            case "batch_size":
                return trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            case "lookback":
                return trial.suggest_int("lookback", 4, 10)

    def tune(self, X_train, y_train, X_val, y_val,
             n_trials=10,
             show_progress_bar=True):

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(active_trial):
            # Копируем фиксированные гиперпараметры
            self.current_params = self.best_params.copy()
            # Сэмплируем гиперпараметры
            for param, value in self.current_params.items():
                if value is None:
                    self.current_params[param] = self.__suggest(active_trial, param)

            model = RNNModel(
                **self.current_params,
                activation=self.activation,
                criterion=self.criterion,
                model_name=self.model_name,
                model_type=self.model_type,
                device=self.device,
                verbosity=self.verbosity
            )

            model.fit(X_train, y_train)
            y_predict = np.concatenate(
                [np.ones((self.current_params["lookback"], self.current_params["output_dim"])) / 2,
                 model.predict(X_val).reshape(-1, 1)], axis=0)
            return self.metric(y_val, y_predict)

        sampler = TPESampler(seed=42)
        study_name = f"{self.model_name}-{datetime.now()}".replace(" ", "_").replace(".", "_")

        if self.verbosity:
            print(f'study name: {study_name}')

        study = optuna.create_study(study_name=study_name,
                                    direction=self.direction,
                                    storage=self.storage_url,
                                    sampler=sampler)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
        trial_params = study.best_trial.params
        if "dropout_rate" not in trial_params.keys():
            trial_params['dropout_rate'] = 0

        for param, value in self.best_params.items():
            if value is None:
                self.best_params[param] = trial_params[param]
        return study.best_value


class RNNModel:
    """
        Модель RNN (`lstm` или `gru`)
        :param input_dim: размерность входных данных
        :param output_dim: размерность выходных данных
        :param hidden_dim: размерность скрытого слоя
        :param num_layers: кол-во рекуррентных блоков
        :param num_epochs: кол-во итераций обучения
        :param dropout_rate: вероятность прореживания
        :param learning_rate: скорость обучения
        :param lookback: размер окна
        :param batch_size: размер батча
        :param criterion: функция ошибки для обучения
        :param activation: функция активации
        :param model_type: вид модели (`lstm` или `gru`)
        :param device: устройство
        :param verbosity: уровень детальности информации
        :param model_name: имя модели
        """

    def __init__(self, input_dim: int = None,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 num_epochs: int = 100,
                 dropout_rate: float = 0.0,
                 learning_rate: float = 1e-4,
                 lookback: int = 10,
                 batch_size: int = 64,
                 criterion=torch.nn.BCELoss(reduction='mean'),
                 activation=nn.Sigmoid(),
                 model_type="lstm",
                 device=init_device(),
                 verbosity: bool = True,
                 model_name: str = "RNN"):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.lookback = lookback
        self.batch_size = batch_size
        self.criterion = criterion
        self.model_type = model_type
        self.device = device
        self.verbosity = verbosity
        self.model_name = model_name
        self.feature_names = []
        self.val_loss_history = None
        self.train_loss_history = None
        self.model = None

    def get_params(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_epochs": self.num_epochs,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "lookback": self.lookback,
            "batch_size": self.batch_size,

        }

    def fit(self, train_X: pd.DataFrame,
            train_y: pd.Series,
            val_X: Optional[pd.DataFrame] = None,
            val_y: Optional[pd.Series] = None):
        """
        Функция для обучения модели
        :param train_X: данные для обучения (факторы)
        :param train_y: данные для обучения (целевая переменная)
        :param val_X: данные для валидации (факторы)
        :param val_y: данные для валидации (целевая переменная)
        """

        self.feature_names = train_X.columns

        validate = val_X is not None and val_y is not None

        train = WindowDataset(train_X, train_y, window_size=self.lookback, trailing_size=self.output_dim,
                              device=self.device)

        if validate:
            val = WindowDataset(val_X, val_y, window_size=self.lookback, trailing_size=self.output_dim,
                                device=self.device)
            val_loader = DataLoader(val, shuffle=False, batch_size=self.batch_size)
        else:
            val_loader = None

        model = BaseRNNModel(input_dim=self.input_dim,
                             hidden_dim=self.hidden_dim,
                             output_dim=self.output_dim,
                             num_layers=self.num_layers,
                             dropout_rate=self.dropout_rate,
                             model_type=self.model_type,
                             activation=self.activation,
                             device=self.device)

        train_loader = DataLoader(train, shuffle=True, batch_size=self.batch_size)

        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_loss, val_loss = [], []  # Средние ошибки за каждую итерацию
        pbar = tq.tqdm(total=int(self.num_epochs), position=0, leave=True, display=self.verbosity)
        for t in range(self.num_epochs):
            train_epoch_loss, val_epoch_loss = [], []  # Ошибки за итерацию
            for i, (x, y) in enumerate(train_loader):
                y_train_predict = model(x)
                optimiser.zero_grad()
                loss = self.criterion(y_train_predict, y)
                loss_value = loss.item()
                train_epoch_loss += [loss_value]
                loss.backward()
                optimiser.step()

            if validate:
                loss_value = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        y_val_predict = model(x)
                        loss = self.criterion(y_val_predict, y)
                        loss_value = loss.item()
                        val_epoch_loss += [loss_value]

            train_loss += [np.mean(train_epoch_loss)]

            val_loss += [np.mean(val_epoch_loss) if validate else np.nan]

            pbar.set_description(f"[Epoch {t}] Loss: {loss_value}", refresh=True)
            pbar.update()

        self.model = model
        self.model.eval()
        self.train_loss_history = train_loss
        self.val_loss_history = val_loss

    def reset(self):
        """
        Сброс состояния модели
        """
        self.model = BaseRNNModel(input_dim=self.input_dim,
                                  hidden_dim=self.hidden_dim,
                                  output_dim=self.output_dim,
                                  num_layers=self.num_layers,
                                  dropout_rate=self.dropout_rate,
                                  model_type=self.model_type,
                                  device=self.device)
        self.feature_names = []
        self.val_loss_history = None
        self.train_loss_history = None

    def predict(self, X: pd.DataFrame):
        """
        Получения предсказания модели
        :param X: данные для предсказания
        :return: y -- предсказание модели
        """
        dataset = WindowDataset(X=X[self.feature_names], y=None,
                                window_size=self.lookback,
                                trailing_size=self.output_dim,
                                device=self.device)
        predicts = self.model(dataset[:][0]).cpu().detach().numpy().reshape(1, -1)[0]
        return predicts

    def plot_history(self, title="<LSTM> Loss"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.train_loss_history))),
                                 y=self.train_loss_history, name="Train Loss",
                                 mode='lines'))
        fig.add_trace(go.Scatter(x=list(range(len(self.val_loss_history))),
                                 y=self.val_loss_history, name="Test Loss",
                                 mode='lines'))

        fig.update_layout(
            title=title,
            paper_bgcolor="rgba(255,255,255,255)",
            plot_bgcolor="rgba(255,255,255,255)"
        )
        fig.show()

    def backtest(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> pd.DataFrame:
        results = pd.DataFrame(index=X.index)
        predict = self.predict(X[self.feature_names])
        results[f"{self.model_name} Predict"] = np.concatenate([np.empty((self.lookback, self.output_dim)),
                                                                predict.reshape(-1, 1)], axis=0)
        if y is not None:
            results["Fact"] = y
        return results

    def load_from(self, path: str):
        """
        Загрузка модели
        :param path: путь до файла, в котором сохранена модель
        :return:
        """
        with open(path, 'rb') as f:
            param_dict = pickle.load(f)
            self.__dict__.update(param_dict)

    def save(self, path: str):
        """
        Сохранение модели
        :param path: путь до файла, в который будет сохранена модель
        :return:
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
