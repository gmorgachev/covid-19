import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from dataclasses import dataclass
from pathlib import Path

from pyro import poutine
from waitress import serve

from src.utils import experiment, get_data, MODELS, COUNTRIES

DATA_PATH = Path("data")
FIG = go.Figure()
COUNTRY = "Russia"
DF = get_data(COUNTRY, DATA_PATH)
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
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    dcc.Graph(figure=FIG, id='graph'),
    dcc.Markdown("""
    Gleb Morgachev   
    Tamaz Gadaev
    """, style={"text-align": "right"})
])


@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    dash.dependencies.Input('submit-button-state', 'n_clicks'),
    dash.dependencies.State('countries-dropdown', 'value'),
    dash.dependencies.State('models-dropdown', 'value'),
    dash.dependencies.State('input_forecasting_size', 'value'),
    dash.dependencies.State('input_train_end', 'value'))
def update_output(n_clicks, country, model_name, forecasting_size, train_end):
    try:
        if country is None or forecasting_size is None or train_end is None or model_name is None:
            FIG.update_layout(title="Choose all parameters")
            return FIG
        if train_end < forecasting_size:
            FIG.update_layout(title=f"train_end < forecasting_size")
            return FIG

        with poutine.block():
            model = MODELS[model_name]
            DF = get_data(country, DATA_PATH)

            fig = experiment(DF, model, int(train_end), int(forecasting_size), country)
            fig.update_layout(title=f"{country} {model_name}")

        return fig
    except Exception as e:
        print(e)
        FIG.update_layout(title="Some problem. Please, try one more time after several minutes."
                                "Maybe somebody else is playing with models :)")
        return FIG


if __name__ == "__main__":
    serve(app.server, host='0.0.0.0')
