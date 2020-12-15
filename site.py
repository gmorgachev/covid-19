import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from dataclasses import dataclass
from pathlib import Path
from waitress import serve

COUNTRIES = ["All", "Russia", "United States", "United Kingdom", "Germany", "France", "India", "Brazil", "China"]
DATA_PATH = Path("data")
FIG = go.Figure()


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


AR_CONFIG = ArConfig()
SARIMAX_CONFIG = SarimaxConfig()


MODELS = {
    "AR": lambda x:\
        sm.tsa.AutoReg(
            x,
            lags=AR_CONFIG.lags,
            seasonal=AR_CONFIG.seasonal,
            period=AR_CONFIG.period
        ),
    "SARIMAX": lambda x:\
        sm.tsa.SARIMAX(
            x,
            order=SARIMAX_CONFIG.order,
            seasonal_order=SARIMAX_CONFIG.seasonal_order,
        )
}


def get_data(country):
    path = DATA_PATH.joinpath(f"{country}.csv")
    df = pd.read_csv(path, index_col=0).sort_index()
    df.index = pd.to_datetime(df.index)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return df


def experiment(df, model, end, forecasting_size):
    assert df.shape[0] >= end

    train = df.iloc[:end-forecasting_size, 0]
    test = df.iloc[end-forecasting_size-1:end, 0]

    res = model(train).fit()
    y_pred = res.forecast(steps=forecasting_size)

    fig = go.Figure()

    fig.add_scatter(x=train.index, y=train.values, line={"color": "blue"}, name="train")
    fig.add_scatter(x=test.index, y=test.values, line={"color": "green"}, name="ground true", mode="lines")
    fig.add_scatter(x=test.index, y=y_pred, line={"color": "red"}, name="predicted", mode="lines")
    fig.update(layout_showlegend=True)

    return fig


COUNTRY = "Russia"
DF = get_data(COUNTRY)
END_IDX = 21
FORECASTING_SIZE = 14


app = dash.Dash(external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([
    dcc.Markdown("""
    # COVID-19
    Демонстрация методов прогнозирования числа зараженных.  
    """, style={'font-family': 'Arial, Helvetica, sans-serif'}),
    dcc.Dropdown(
        id='countries-dropdown',
        options=[{'label': x, 'value': x} for x in COUNTRIES],
    ),
    dcc.Dropdown(
        id='models-dropdown',
        options=[{'label': x, 'value': x} for x in MODELS.keys()],
    ),
    dcc.Input(id="input_forecasting_size", placeholder="forecasting size", type="number"),
    dcc.Input(id="input_train_end", placeholder=f"train end (max = {DF.shape[0]})", type="number"),
    dcc.Graph(figure=FIG, id='graph'),
    dcc.Markdown("""
    Gleb Morgachev   
    Tamaz Gadaev
    """, style={"text-align": "right"})
])


@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('models-dropdown', 'value'),
     dash.dependencies.Input('input_forecasting_size', 'value'),
     dash.dependencies.Input('input_train_end', 'value')])
def update_output(country, model_name, forecasting_size, train_end):
    if country is None or forecasting_size is None or train_end is None or model_name is None:
        return FIG

    model = MODELS[model_name]
    DF = get_data(country)

    fig = experiment(DF, model, int(train_end), int(forecasting_size))
    fig.update_layout(title=f"{country} {model_name}")
    return fig


if __name__ == "__main__":
    serve(app.server, host='0.0.0.0')
