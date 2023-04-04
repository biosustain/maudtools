"""Test the generating_sbml module."""

import arviz as az
from importlib_resources import files
from maud.data.example_inputs import methionine
from maud.loading_maud_inputs import load_maud_input

from maudtools import data
from maudtools.generating_sbml import generate_sbml


def test_generate_sbml():
    mi = load_maud_input(data_path=files(methionine)._paths[0])
    idata = az.from_json(files(data).joinpath("idata_methionine.json"))
    chain = draw = warmup = 0
    experiment = next(ex.id for ex in mi.experiments)
    sbml_doc, sbml_model = generate_sbml(
        idata, mi, experiment, chain, draw, warmup
    )
    expected_formula = (
        "prv_concenzymetrainAHC1"
        " * prv_kcatAHC1"
        " * (1 - exp((cst_dgrtrainAHC1AHC + 2.4788191 * "
        "(-1 * log(spc_ahcysc) + 1 * log(prv_concunbalancedtrainadnc) + 1 * log(spc_hcysLc)))"
        " / 2.4788191))"
        " * spc_ahcysc / prv_kmAHC1ahcysc"
        " * 1 / ((1 + spc_ahcysc / prv_kmAHC1ahcysc)^1 + cst_zero +"
        " ((1 + prv_concunbalancedtrainadnc / prv_kmAHC1adnc)^1 *"
        " (1 + spc_hcysLc / prv_kmAHC1hcysLc)^1"
        " - 1))"
        " * 1"
        " * 1"
    )
    assert (
        sbml_model.getReaction("rxn_AHC1AHC").getKineticLaw().getFormula()
        == expected_formula
    )
