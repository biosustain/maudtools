"""Test the generating_sbml module."""

import os

from maud.getting_idatas import get_idata
from maud.loading_maud_inputs import load_maud_input

from maudtools.generating_sbml import generate_sbml

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "methionine_output"
)


def test_generate_sbml():
    sample_dir = os.path.join(DATA_PATH, "samples")
    maud_input_dir = os.path.join(DATA_PATH, "user_input")
    csvs = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith(".csv")
    ]
    mi = load_maud_input(maud_input_dir)
    idata = get_idata(csvs, mi, "train")
    chain = draw = warmup = 0
    experiment = next(ex.id for ex in mi.measurements.experiments)
    sbml_doc, sbml_model = generate_sbml(
        idata, mi, experiment, chain, draw, warmup
    )
    expected_formula = (
        "pconcenzymeAHC1"
        " * pkcatAHC1"
        " * (1 - exp((pdgrsAHC1AHC + 2.4788191 * "
        "(-1 * log(sahcysc) + 1 * log(pconcunbalancedadnc) + 1 * log(shcysLc)))"
        " / 2.4788191))"
        " * sahcysc / pkmAHC1ahcysc"
        " * 1 / ((1 + sahcysc / pkmAHC1ahcysc)^1 + zero +"
        " ((1 + pconcunbalancedadnc / pkmAHC1adnc)^1 * (1 + shcysLc / pkmAHC1hcysLc)^1"
        " - 1))"
        " * 1"
        " * 1"
    )
    assert (
        sbml_model.getReaction("rAHC1AHC").getKineticLaw().getFormula()
        == expected_formula
    )
