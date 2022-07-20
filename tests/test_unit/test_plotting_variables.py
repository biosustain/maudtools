import os

import arviz
import pytest
from maud.data_model.stan_variable_set import (
    ConcEnzyme,
    ConcPme,
    ConcUnbalanced,
    Dgf,
    Drain,
    Kcat,
    Ki,
    Km,
)
from maud.loading_maud_inputs import load_maud_input

from maudtools.plotting import plot_maud_prior, plot_maud_variable

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "input_G6PtoPEP"
)
OUTP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "methionine_output", "idata.nc"
)


@pytest.mark.parametrize(
    "stan_var", [ConcEnzyme, ConcUnbalanced, Dgf, Drain, Kcat, Ki, Km]
)
def test_plot_posterior_and_prior(stan_var):
    mi = load_maud_input(DATA_PATH)
    idata = arviz.from_netcdf(OUTP_PATH)
    plot_maud_variable(idata, mi.priors, stan_var)


@pytest.mark.parametrize(
    "stan_var", [ConcEnzyme, ConcUnbalanced, Dgf, Drain, Kcat, Ki, Km]
)
def test_plot_prior_alone(stan_var):
    mi = load_maud_input(DATA_PATH)
    idata = arviz.from_netcdf(OUTP_PATH)
    plot_maud_prior(idata, mi.priors, stan_var, concat=False)
