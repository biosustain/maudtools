import arviz
import pytest
from importlib_resources import files
from maud.data.example_outputs import linear
from maud.loading_maud_inputs import load_maud_input

from maudtools.plotting import plot_maud_prior, plot_maud_variable


@pytest.mark.parametrize(
    "stan_var",
    ["conc_enzyme_train", "conc_unbalanced_train", "dgf", "kcat", "ki", "km"],
)
def test_plot_posterior_and_prior(stan_var):
    mi = load_maud_input(files(linear).joinpath("user_input"))
    idata = arviz.from_json(files(linear).joinpath("idata.json"))
    plot_maud_variable(idata, getattr(mi.parameters, stan_var))


@pytest.mark.parametrize(
    "stan_var",
    ["conc_enzyme_train", "conc_unbalanced_train", "dgf", "kcat", "ki", "km"],
)
def test_plot_prior_alone(stan_var):
    mi = load_maud_input(files(linear).joinpath("user_input"))
    idata = arviz.from_json(files(linear).joinpath("idata.json"))
    plot_maud_prior(idata, getattr(mi.parameters, stan_var), concat=False)
