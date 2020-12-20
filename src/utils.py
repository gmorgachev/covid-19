import os
import statsmodels.api as sm
import pandas as pd
import plotly.graph_objects as go

from dataclasses import dataclass

from .transport_models import SIRBasedModel


COUNTRIES = ["All", "Russia", "United States", "United Kingdom", "Germany", "France", "India", "Brazil", "China"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@dataclass
class ArConfig:
    lags = (3, 7)
    seasonal = True
    period = 7
    forecasting_size = 14


@dataclass
class SarimaxConfig:
    order = (10, 1, 7)
    seasonal_order = (0, 1, 2, 10)
    forecasting_size = 14


def get_data(country, data_path):
    path = data_path.joinpath(f"{country}.csv")
    df = pd.read_csv(path, index_col=0).sort_index()
    df.index = pd.to_datetime(df.index)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return df


def experiment(df, model, end, forecasting_size, country):
    assert df.shape[0] >= end

    train = df.iloc[:end-forecasting_size, 0]
    test = df.iloc[end-forecasting_size-1:end, 0]

    res = model(train, country).fit()
    y_pred = res.forecast(steps=forecasting_size)

    fig = go.Figure()

    fig.add_scatter(x=train.index, y=train.values, line={"color": "blue"}, name="train")
    fig.add_scatter(x=test.index, y=test.values, line={"color": "green"}, name="ground true", mode="lines")
    fig.add_scatter(x=test.index, y=y_pred, line={"color": "red"}, name="predicted", mode="lines")
    fig.update(layout_showlegend=True)

    return fig


AR_CONFIG = ArConfig()
SARIMAX_CONFIG = SarimaxConfig()


MODELS = {
    "AR": lambda x, _: \
        sm.tsa.AutoReg(
            x,
            lags=AR_CONFIG.lags,
            seasonal=AR_CONFIG.seasonal,
            period=AR_CONFIG.period
        ),
    "SARIMAX": lambda x, _: \
        sm.tsa.SARIMAX(
            x,
            order=SARIMAX_CONFIG.order,
            seasonal_order=SARIMAX_CONFIG.seasonal_order,
        )
}

for name, cls in SIRBasedModel.models.items():
    MODELS[name] = lambda x, country, method_name=name: SIRBasedModel(method_name, country)(x)
