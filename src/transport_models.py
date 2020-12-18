import inspect
import torch

from pyro.contrib.epidemiology.models import SimpleSIRModel, SimpleSEIRDModel, SimpleSEIRModel

torch.set_default_dtype(torch.float64)


def get_params(model_class, global_params):
    args = inspect.getfullargspec(model_class)[0][1:-1]
    kwargs = {}
    for arg in args:
        kwargs.update({arg: global_params[arg]})
    return kwargs


class SIRBasedModel:
    models = {cls.__name__: cls for cls in [SimpleSIRModel, SimpleSEIRModel, SimpleSEIRDModel]}
    population_dict = {
        country: company for country, company in zip(
            ['All', 'Russia', 'United States', 'United Kingdom', 'Germany', 'France', 'India', 'Brazil', 'China'],
            [7*10**9, 144500000, 328200000, 55980000, 83200000, 66990000, 1353000000, 209500000, 1393000000]
        )}
    global_params = {
        'recovery_time': 14.,
        'incubation_time': 5.,
        'mortality_rate': 0.04
    }

    def __init__(self, model_name, country):
        assert model_name in self.models.keys()

        params = self.global_params
        params["population"] = self.population_dict[country]
        self.kwargs = get_params(self.models[model_name], params)
        self.model_class = self.models[model_name]

    def __call__(self, train):
        self.train = torch.Tensor(train)
        self.model = self.model_class(data=self.train, **self.kwargs)
        return self

    def fit(self):
        self.model.fit_mcmc(num_samples=100)
        return self

    def forecast(self, steps):
        res = self.model.predict(forecast=steps)
        keys = [x for x in res.keys() if x.endswith("2I")]
        assert len(keys) == 1
        return res[keys[0]].numpy().mean(0)
