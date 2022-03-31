import os
from maud.io import load_maud_input
from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input_G6PtoPEP")


def test_fetch_dgf_priors_from_equilibrator():
    mi = load_maud_input(DATA_PATH, mode="sample")
    mu, cov = fetch_dgf_priors_from_equilibrator(mi)
    mu_expected = pd.read_csv(
        os.path.join(DATA_PATH, "dgf_prior_mean_equilibrator.csv"),
        index_col="metabolite"
    )
    cov_expected = pd.read_csv(
        os.path.join(DATA_PATH, f"dgf_prior_cov_equilibrator.csv"),
        index_col="metabolite"
    )
    assert isinstance(mu_expected, pd.DataFrame)
    assert isinstance(cov_expected, pd.DataFrame)
    mu_expected = mu_expected.squeeze("columns")
    cov_expected = cov_expected.rename_axis("metabolite", axis="columns")
    pd.testing.assert_series_equal(mu, mu_expected)
    pd.testing.assert_frame_equal(cov, cov_expected)

