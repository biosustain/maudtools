"""Test the generating_inits module."""

import arviz as az
import numpy as np
import toml
from importlib_resources import files
from maud.data.example_inputs import methionine
from maud.loading_maud_inputs import load_maud_input

from maudtools import data
from maudtools.generating_inits import generate_inits


def test_generate_inits():
    """Check that the generate_inits function works as expected."""
    expected = toml.load(files(methionine).joinpath("inits.toml"))
    mi = load_maud_input(data_path=files(methionine)._paths[0])
    idata = az.from_json(files(data).joinpath("idata_methionine.json"))
    actual = generate_inits(idata, mi, 0, 0, 0)
    for param, init_atoms_actual in actual.items():
        for init_atom_actual in init_atoms_actual:
            try:
                init_atom_expected = next(
                    d
                    for d in expected[param]
                    if all(
                        d[k] == init_atom_actual[k]
                        for k in init_atom_actual.keys()
                        if k != "init"
                    )
                )
            except:
                raise ValueError(f"No {param} init matching {init_atom_actual}.")
            assert np.isclose(
                init_atom_actual["init"], init_atom_expected["init"]
            ), f"{param} init {init_atom_actual} should be {init_atom_expected}."
