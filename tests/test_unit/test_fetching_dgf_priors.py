import importlib_resources
import numpy as np
import toml
from maud.data.example_inputs import methionine
from maud.loading_maud_inputs import load_maud_input

from maudtools import data
from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator


def test_fetch_dgf_priors_from_equilibrator():
    expected_path = importlib_resources.files(data).joinpath(
        "expected_dgf_priors_methionine.toml"
    )
    expected = toml.load(expected_path)
    mi_path = importlib_resources.files(methionine)._paths[0]
    actual = fetch_dgf_priors_from_equilibrator(
        load_maud_input(mi_path), "298.15K", 7.5
    )
    for id_a, id_e, mu_a, mu_e, covs_a, covs_e in zip(
        actual["ids"],
        expected["ids"],
        actual["mean_vector"],
        expected["mean_vector"],
        actual["covariance_matrix"],
        expected["covariance_matrix"],
    ):
        assert id_a == id_e, f"Id {id_a} should be ({id_e})."
        assert np.isclose(
            mu_a, mu_e
        ), f"mean for {id_a} is {mu_a} but should be {mu_e}."
        for id_ak, cov_a, cov_e in zip(actual["ids"], covs_a, covs_e):
            assert np.isclose(
                cov_a, cov_e
            ), f"Covariance for {id_a} and {id_ak} is {cov_a} but should be {cov_e}."
