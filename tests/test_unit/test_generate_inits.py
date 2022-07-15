import os

import arviz as az
import pandas as pd
from maud.loading_maud_inputs import load_maud_input

from maudtools.generating_inits import generate_inits

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "methionine_output"
)

ID_COLS = [
    "enzyme",
    "metabolite",
    "compartment",
    "parameter",
    "experiment",
    "reaction",
]


def test_generate_inits():
    """Check that the generate_inits function works as expected."""
    mi = load_maud_input(data_path=os.path.join(DATA_PATH, "user_input"))
    idata = az.from_netcdf(os.path.join(DATA_PATH, "idata.nc"))
    inits = generate_inits(idata, mi, 0, 0, 0).set_index(ID_COLS).sort_index()
    expected_inits = (
        pd.read_csv(os.path.join(DATA_PATH, "generated_inits.csv"), index_col=0)
        .set_index(ID_COLS)
        .sort_index()
    )
    pd.testing.assert_frame_equal(inits, expected_inits)
