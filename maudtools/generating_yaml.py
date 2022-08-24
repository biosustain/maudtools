"""Provides the function generate_yaml, which takes in an InferenceData,
MaudInput, chain, draw and experiment and returns a string that can be parsed by
yaml2sbml in order to create an SBML file.

"""

from typing import List

import arviz as az
import numpy as np
import xarray as xr
from jinja2 import Template
from maud.data_model.kinetic_model import Enzyme, EnzymeReaction
from maud.data_model.maud_input import MaudInput

from maudtools.generating_inits import generate_inits


def get_directional_component(
    er: EnzymeReaction,
    is_forward,
    mi: MaudInput,
    posterior: xr.Dataset,
    chain: int,
    draw: int,
    experiment: str,
):
    """Get an expression for prod_{+/-}((conc/km) ^ subunits)."""
    enzyme = next(e for e in mi.kinetic_model.enzymes if e.id == er.enzyme_id)
    rxn = next(r for r in mi.kinetic_model.reactions if r.id == er.reaction_id)
    sign = 1 if is_forward else -1
    mic_ids = [mic for mic, coef in rxn.stoichiometry if np.sign(coef) == sign]
    mics = [m for m in mi.kinetic_model.mics if m.id in mic_ids]
    ecd = dict(experiment=experiment, chain=chain, draw=draw)
    cd = dict(chain=chain, draw=draw)
    productands: List = []
    for mic in mics:
        km_id = f"{enzyme.id}_{mic.metabolite_id}_{mic.compartment_id}"
        conc = posterior["conc"].sel(mics=mic.id, **ecd).values[0]  # type: ignore
        km = posterior["km"].sel(kms=km_id, **cd).values[0]  # type: ignore
        productands += f"({conc}/{km})^{enzyme.subunits}"
    return "*".join(productands)


def get_Tr(
    er: EnzymeReaction,
    mi: MaudInput,
    posterior: xr.Dataset,
    chain: int,
    draw: int,
    experiment: str,
    reversible: bool,
) -> str:
    """Get the Tr component of modular rate law."""
    template_reversible = "{enz} * {kcat} * ({trf} - {trr} * {hal})"
    template_irreversible = "{enz} * {kcat} * {trf}"
    ecd = dict(experiment=experiment, chain=chain, draw=draw)
    chdr = dict(chain=chain, draw=draw)
    enz = posterior["conc_enzyme"].sel(enzyme=er.enzyme_id, **ecd).values[0]
    kcat = posterior["kcat"].sel(enzyme=er.enzyme_id, **chdr).values[0]
    trf = get_directional_component(er, True, mi, posterior, **ecd)
    if not reversible:
        return template_irreversible.format(enz, kcat, trf)
    else:
        keq = idata.posterior["keq"].sel(edge=er.id, **ecd).values[0]
        trr = get_directional_component(er, False, mi, posterior, **ecd)
        hal = get_haldane_expression()
        return template_reversible.format(enz, kcat, trf, trr, hal)


def generate_yaml(
    idata,
    mi: MaudInput,
    chain: int,
    draw: int,
    warmup: bool,
    experiment: Optional[str],
) -> str:
    """Get a string of yaml from a maud output idata"""
    parameter_values = generate_inits(idata, mi, chain, draw, int(warmup))
