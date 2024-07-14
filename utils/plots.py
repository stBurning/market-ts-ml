import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = 'simple_white'
BG_COLOR = "rgba(255,255,255,255)"
GREEN = "rgba(8,153,129,255)"
RED = "rgba(242,54,69,255)"


def plot_candles(df, x=None, title:str="", yaxis_title:str="", show:bool=False, log:bool=False, secondary_y:bool=False, fig=None):
    """
    Отрисовка свечного графика
    :param df: исходные данные, содержащие OHLCV столбцы
    :param x: индекс (дата)
    :param title: название для графика
    :param yaxis_title: наименование оси y
    :param show: флаг для отложенной отрисовки
    :param log: логарифмическая шкала для оси y
    :param secondary_y: если True - добавляется отдельная ось y
    :param fig: существующий график
    :return:
    """
    x = df.index if x is None else df[x]

    if fig is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=BG_COLOR,
            decreasing_fillcolor=RED,
            name='Price',
        ), secondary_y=secondary_y
    )

    fig.update_layout(xaxis_rangeslider_visible=False,
                      paper_bgcolor=BG_COLOR,
                      plot_bgcolor=BG_COLOR)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        yaxis_title=yaxis_title,
    )

    if log:
        fig.update_yaxes(type="log")
    if show:
        fig.show()
    else:
        return fig


def plot_lines(df, x=None, y=None, title="", yaxis_title="", show=True, fig=None, log=False, c=None, name=None,
               secondary_y=False, heat=None, save_path=None, width=800, height=600):
    if fig is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=title,
            yaxis_title=yaxis_title,
            xaxis_rangeslider_visible=False,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR
        )

    if y is None:
        col_names = df.columns
    elif isinstance(y, list):
        col_names = y
    else:
        col_names = [y]

    if c is None:
        colors = None
    elif isinstance(c, list):
        colors = c
    else:
        colors = [c]

    for i, col_name in enumerate(col_names):
        fig.add_trace(
            go.Scatter(
                x=df.index if x is None else df[x],
                y=df[col_name],
                mode="lines" if heat is None else "markers+lines",
                line=dict(width=2) if colors is None else dict(color=colors[i], width=2),
                marker=None if heat is None else {'color': df[heat], 'colorscale': 'Inferno', 'size': 10},
                name=col_name if name is None else name,
                showlegend=True), secondary_y=secondary_y
        )
    if log:
        fig.update_yaxes(type="log")

    if save_path is not None:
        fig.write_image(save_path, width=width, height=height)
    if show:
        fig.show()
    else:
        return fig


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
