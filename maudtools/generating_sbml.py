"""Provides the function generate_sbml.

This function takes in an InferenceData, MaudInput, chain, draw and experiment
and returns an SBML file in string format.

Prefixes for sbml variables are as follows

_rxn_: reaction
_spc_: species
_cst_: constant
_prv_: parameter value
_prl_: parameter prior location
_prs_: parameter prior scale
_prd_: parameter distribution: 0 if normal, 1 if lognormal
_mtv_: measurement value
_mts_: measurement error scale

"""
import warnings
from typing import List, Tuple

import arviz as az
import libsbml as sbml  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import fields
from maud.data_model.hardcoding import ID_SEPARATOR  # type: ignore
from maud.data_model.prior import IndPrior1d, IndPrior2d
from maud.data_model.maud_parameter import MaudParameter
from maud.data_model.kinetic_model import EnzymeReaction  # type:ignore
from maud.data_model.kinetic_model import (
    KineticModel,
    ModificationType,
    Reaction,
    ReactionMechanism,
)
from maud.data_model.maud_input import MaudInput  # type: ignore

PREFIXES = {
    "reaction": "_rxn_",
    "species": "_spc_",
    "constant": "_cst_",
    "param_value": "_prv_",
    "param_location": "_prl_",
    "param_scale": "_prs_",
    "param_dist": "_prd_",
    "measurement_value": "_mtv_",
    "measurement_error_scale": "_mts_",
}
SBML_VARS = [
    "drain_train",
    "kcat",
    "km",
    "ki",
    "dissociation_constant",
    "transfer_constant",
    "conc_unbalanced_train",
    "conc_enzyme_train",
]


def squash(long: str) -> str:
    return long.replace("_", "").replace("-", "")


def generate_sbml(
    idata: az.InferenceData,
    mi: MaudInput,
    experiment_id: str,
    chain: int,
    draw: int,
    warmup: int,
) -> Tuple[sbml.SBMLDocument, sbml.Model, pd.DataFrame]:
    """Run the main function of this module."""
    posterior = idata.posterior if not warmup else idata.warmup_posterior  # type: ignore
    draw = posterior.sel(chain=chain, draw=draw, experiments=experiment_id)
    experiment_ix, experiment = next(
        (i, exp)
        for i, exp in enumerate(mi.experiments)
        if exp.id == experiment_id
    )
    doc, model = initialise_model(mi)
    assert isinstance(draw, xr.Dataset)

    zero = model.createParameter()
    zero.setId(PREFIXES["constant"] + "zero")
    zero.setName(PREFIXES["constant"] + "zero")
    zero.setValue(0.0)
    zero.setConstant(True)
    for mp in (getattr(mi.parameters, f.name) for f in fields(mi.parameters)):
        if mp.name in draw.data_vars:
            add_parameter_to_model(model, mp, draw, experiment_ix)
    for dgr_id, dgr_val in draw["dgr_train"].to_series().iteritems():
        add_parameter_atom_to_model(model, "dgr_train", dgr_id, dgr_val)
    add_species_to_model(model, mi, experiment_ix)
    add_reactions_to_model(model, mi, experiment.temperature)
    add_measurements_to_model(model, mi, experiment_ix)
    doc.setConsistencyChecks(sbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    if doc.checkConsistency():
        for error_num in range(doc.getErrorLog().getNumErrors()):
            if not doc.getErrorLog().getError(error_num).isWarning():
                warnings.warn(
                    doc.getErrorLog().getError(error_num).getMessage(),
                    RuntimeWarning,
                )
    return doc, model


def initialise_model(mi: MaudInput) -> Tuple[sbml.SBMLDocument, sbml.Model]:
    """Generate sbml document and model and add basic info from maud input."""
    doc = sbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId(mi.config.name)
    model.setName(mi.config.name)
    for maud_cpt in mi.kinetic_model.compartments:
        cpt: sbml.Compartment = model.createCompartment()
        cpt.setId(maud_cpt.id)
        cpt.setConstant(True)
        cpt.setSize(maud_cpt.volume)
        cpt.setVolume(maud_cpt.volume)
        cpt.setSpatialDimensions(3)
        cpt.setUnits("litre")
    return doc, model


def add_measurements_to_model(
    model: sbml.Model, mi: MaudInput, experiment_ix: int
):
    for m in mi.experiments[experiment_ix].measurements:
        if m.target_type == "mic":
            target_id = m.metabolite + m.compartment
        elif m.target_type == "flux":
            target_id = m.reaction
        elif m.target_type == "enzyme":
            target_id = m.enzyme_id
        val_id = (
            PREFIXES["measurement_value"]
            + m.target_type.value
            + m.experiment
            + target_id
        )
        val = model.createParameter()
        if val.setId(val_id) != sbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(
                f"Unable to generate parameter with id {val_id}."
            )
        val.setId(val_id)
        val.setConstant(True)
        val.setValue(m.value)
        sd = model.createParameter()
        sd_id = (
            PREFIXES["measurement_error_scale"]
            + m.target_type.value
            + m.experiment
            + target_id
        )
        sd.setId(sd_id)
        sd.setConstant(True)
        sd.setValue(m.error_scale)
        if sd.setId(sd_id) != sbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(f"Unable to generate parameter with id {sd_id}.")


def add_parameter_atom_to_model(
    model, param_name, coord, pv, loc=None, scale=None, dist=None
):
    """Add a parameter atom to an sbml model."""
    pv_param = model.createParameter()
    param_atom_id = squash(param_name + coord)
    is_const = param_name not in SBML_VARS
    pvid = (
        PREFIXES["constant"] + param_atom_id
        if is_const
        else PREFIXES["param_value"] + param_atom_id
    )
    if pv_param.setId(pvid) != sbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"Unable to generate parameter with id {pvid}.")
    pv_param.setId(pvid)
    pv_param.setName(pvid)
    pv_param.setValue(pv)
    if is_const:
        pv_param.setConstant(True)
    else:
        pv_param.setConstant(False)
        for s, val in zip(["location", "scale", "dist"], [loc, scale, dist]):
            sbml_param = model.createParameter()
            sbml_param_id = PREFIXES[f"param_{s}"] + param_atom_id
            sbml_param.setId(sbml_param_id)
            sbml_param.setName(sbml_param_id)
            sbml_param.setValue(val)
            sbml_param.setConstant(True)


def add_parameter_to_model(
    model: sbml.Model, mp: MaudParameter, draw: xr.Dataset, experiment_ix: int
):
    """Add a parameter to an sbml model."""
    is_const = mp.name not in SBML_VARS
    dist = None if is_const else (1 if mp.non_negative else 0)
    if len(draw[mp.name].dims) == 0:
        prv = float(draw[mp.name].values)
        loc = None if is_const else mp.prior.location[0]
        scale = None if is_const else mp.prior.scale[0]
        add_parameter_atom_to_model(model, mp.name, "", prv, loc, scale, dist)
    if len(draw[mp.name].dims) == 1:
        for i, (coord, pv) in enumerate(draw[mp.name].to_series().iteritems()):
            if is_const:
                continue
            elif isinstance(mp.prior, IndPrior1d):
                loc = mp.prior.location[i]
                scale = mp.prior.scale[i]
            elif isinstance(mp.prior, IndPrior2d):
                loc = mp.prior.location[experiment_ix][i]
                scale = mp.prior.scale[experiment_ix][i]
            else:
                raise ValueError(
                    f"{mp.prior} is not an IndPrior1d or IndPrior2d."
                )
            add_parameter_atom_to_model(
                model, mp.name, coord, pv, loc, scale, dist
            )


def add_species_to_model(model: sbml.Model, mi: MaudInput, experiment_ix: int):
    """Add species to a model with initial concs from an experiment."""
    conc_init = mi.stan_input_train["conc_init"]
    balanced_mics = [m for m in mi.kinetic_model.mics if m.balanced]
    for i, mic in enumerate(balanced_mics):
        spid = PREFIXES["species"] + squash(mic.id)
        sp = model.createSpecies()
        sp.setCompartment(mic.compartment_id)
        sp.setId(spid)
        sp.setName(spid)
        sp.setInitialConcentration(conc_init[experiment_ix][i])


def add_reactions_to_model(
    model: sbml.Model, mi: MaudInput, temperature: float
):
    """Add reactions to an sbml model."""
    for edge in mi.kinetic_model.edges:
        maud_rxn_id = (
            edge.reaction_id if isinstance(edge, EnzymeReaction) else edge.id
        )
        maud_rxn = next(
            r for r in mi.kinetic_model.reactions if r.id == maud_rxn_id
        )
        sbml_rxn_id = PREFIXES["reaction"] + squash(edge.id)
        sbml_rxn = model.createReaction()
        sbml_rxn.setId(sbml_rxn_id)
        for mic_id, stoic in maud_rxn.stoichiometry.items():
            mic = next(m for m in mi.kinetic_model.mics if m.id == mic_id)
            spid = PREFIXES["species"] + squash(mic.id)
            if mic.balanced and stoic < 0:
                spr = sbml_rxn.createReactant()
                spr.setSpecies(spid)
            elif mic.balanced and stoic > 0:
                spr = sbml_rxn.createProduct()
                spr.setSpecies(spid)
        # handle modifiers
        if isinstance(edge, EnzymeReaction):
            for mic in filter(lambda m: m.balanced, mi.kinetic_model.mics):
                spid = PREFIXES["species"] + squash(mic.id)
                if mi.kinetic_model.competitive_inhibitions is not None:
                    for ci in mi.kinetic_model.competitive_inhibitions:
                        if (
                            ci.mic_id == mic.id
                            and ci.enzyme_id == edge.enzyme_id
                        ):
                            mfr = sbml_rxn.createModifier()
                            mfr.setSpecies(spid)
                if mi.kinetic_model.allosteries is not None:
                    for al in mi.kinetic_model.allosteries:
                        if (
                            al.mic_id == mic.id
                            and al.enzyme_id == edge.enzyme_id
                        ):
                            mfr = sbml_rxn.createModifier()
                            mfr.setSpecies(spid)
        kl = sbml_rxn.createKineticLaw()
        flux_expr = get_edge_flux(mi, edge.id, temperature)
        math_ast = sbml.parseL3Formula(flux_expr)
        if math_ast is None:
            raise RuntimeError(
                f"Unable to generate flux expression for reaction {sbml_rxn_id}"
            )
        kl.setMath(math_ast)


def get_edge_flux(mi: MaudInput, edge_id: str, temperature: float) -> str:
    """Get the flux for an edge."""
    edge = next(e for e in mi.kinetic_model.edges if e.id == edge_id)
    if isinstance(edge, Reaction):
        return get_drain_flux(
            mi.kinetic_model,
            edge,
            mi.config.drain_small_conc_corrector,
        )
    elif isinstance(edge, EnzymeReaction):
        return get_enzyme_reaction_flux(mi, edge, temperature)
    else:
        raise ValueError("Input must be either a Reaction or an EnzymeReactoin")


def get_drain_flux(km: KineticModel, drain: Reaction, corrector: float) -> str:
    """Get an expression for the flux for a drain edge."""
    pid = PREFIXES["param_value"] + "draintrain" + squash(drain.id)
    sub_ids = list(k for k, v in drain.stoichiometry.items() if v < 0)
    if len(sub_ids) == 0:
        return f"({pid})"
    sub_conc_exprs = get_conc_expressions(km, sub_ids)
    corrector_cpts = [
        f"({conc}/({conc}+{str(corrector)}))" for conc in sub_conc_exprs
    ]
    corrector_term = "*".join(corrector_cpts)
    return f"({pid}*({corrector_term}))"


def get_enzyme_reaction_flux(
    mi: MaudInput,
    edge: EnzymeReaction,
    temperature: float,
) -> str:
    """Get an expression for the flux for an EnzymeReaction edge."""
    flux_components = [
        get_enzyme_concentration(edge.enzyme_id),
        get_kcat(edge.enzyme_id),
        get_reversibility(mi.kinetic_model, edge, temperature),
        get_saturation(edge.id, mi.kinetic_model),
        get_allostery(mi.kinetic_model, edge),
        get_phosphorylation(mi.kinetic_model, edge),
    ]
    return f"({'*'.join(flux_components)})"


def get_enzyme_concentration(enzyme_id: str) -> str:
    """Get the concentration for an enzyme."""
    return PREFIXES["param_value"] + "concenzymetrain" + squash(enzyme_id)


def get_kcat(enzyme_id: str) -> str:
    """Get an expression for kcat."""
    return PREFIXES["param_value"] + "kcat" + squash(enzyme_id)


def get_reversibility(
    km: KineticModel,
    edge: EnzymeReaction,
    temperature: float,
) -> str:
    """Get the reversibility for a reaction in an experiment."""
    RT = str(0.008314 * temperature)
    reaction = next(r for r in km.reactions if r.id == edge.reaction_id)
    if reaction.mechanism == ReactionMechanism.irreversible_michaelis_menten:
        return "1"
    mic_ids = list(reaction.stoichiometry.keys())
    stoics = list(reaction.stoichiometry.values())
    dgr_expr = PREFIXES["constant"] + "dgrtrain" + squash(edge.id)
    conc_exprs = get_conc_expressions(km, mic_ids)
    reaction_quotient_cpts = [
        # this is because matlab cannot parse 'ln'
        f"({stoic}*ln({conc_expr}))"
        for stoic, conc_expr in zip(stoics, conc_exprs)
    ]
    reaction_quotient_expression = f"({'+'.join(reaction_quotient_cpts)})"
    return f"(1-exp(({dgr_expr}+{RT}*{reaction_quotient_expression})/{RT}))"


def get_allostery(km: KineticModel, er: EnzymeReaction) -> str:
    """Get the allostery component for an enzyme reaction in an experiment."""
    allosteries = [a for a in km.allosteries if a.enzyme_id == er.enzyme_id]
    if len(allosteries) == 0:
        return "(1)"
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    fer = get_free_enzyme_ratio(er.id, km)
    tc = PREFIXES["param_value"] + "transferconstant" + squash(er.enzyme_id)
    Qnum_cpts = ["1"]
    Qdenom_cpts = ["1"]
    for allostery in allosteries:
        conc = get_conc_expressions(km, [allostery.mic_id])[0]
        dc = (
            PREFIXES["param_value"]
            + "dissociationconstant"
            + squash(allostery.id)
        )
        if allostery.modification_type == ModificationType.activation:
            Qdenom_cpts += [f"{conc}/{dc}"]
        else:
            Qnum_cpts += [f"{conc}/{dc}"]
        Qnum = f"({'+'.join(Qnum_cpts)})"
        Qdenom = f"({'+'.join(Qdenom_cpts)})"
    return f"(1/(1+{tc}*({fer}*{Qnum}/{Qdenom})^{str(enzyme.subunits)}))"


def get_conc_expressions(km: KineticModel, mic_ids: List[str]) -> List[str]:
    """Get expressions for the concentrations of some mic ids.

    This job needs its own function because some mic concentrations are state
    variables and others are parameters. Specifically, the state variables are
    the balanced mics.

    """
    balanced_mic_ids = [m.id for m in km.mics if m.balanced]
    return [
        PREFIXES["species"] + squash(mic_id)
        if mic_id in balanced_mic_ids
        else PREFIXES["param_value"] + "concunbalancedtrain" + squash(mic_id)
        for mic_id in mic_ids
    ]


def get_free_enzyme_ratio(er_id: str, km: KineticModel) -> str:
    """Get the free enzyme ratio for an EnzymeReaction."""
    er = next(e for e in km.edges if e.id == er_id)
    S = km.stoichiometric_matrix
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    reaction = next(r for r in km.reactions if r.id == er.reaction_id)
    sub_ids = list(k for k, v in reaction.stoichiometry.items() if v < 0)
    sub_km_ids = [ID_SEPARATOR.join([enzyme.id, s]) for s in sub_ids]
    sub_conc_exprs = get_conc_expressions(km, sub_ids)
    sub_km_exprs = [
        PREFIXES["param_value"] + "km" + squash(km_id) for km_id in sub_km_ids
    ]
    denom_sub_cpt = "*".join(
        f"(1+{conc_expr}/{km_expr})^{np.abs(S.loc[sub_id, er_id])}"
        for conc_expr, km_expr, sub_id in zip(
            sub_conc_exprs, sub_km_exprs, sub_ids
        )
    )
    if km.competitive_inhibitions is not None and any(
        ci.er_id == er_id for ci in km.competitive_inhibitions
    ):
        cis = [ci for ci in km.competitive_inhibitions if ci.er_id == er_id]
        ci_mic_ids = [ci.mic_id for ci in cis]
        ci_conc_exprs = get_conc_expressions(km, ci_mic_ids)
        ki_exprs = [
            PREFIXES["param_value"] + "ki" + squash(ci.id) for ci in cis
        ]
        denom_ci_cpt = "+".join(
            f"({conc_expr}/{ki_expr})"
            for conc_expr, ki_expr in zip(ci_conc_exprs, ki_exprs)
        )
    else:
        denom_ci_cpt = PREFIXES["constant"] + "zero"
    if reaction.mechanism == ReactionMechanism.reversible_michaelis_menten:
        prod_ids = list(k for k, v in reaction.stoichiometry.items() if v > 0)
        prod_km_ids = [ID_SEPARATOR.join([enzyme.id, p]) for p in prod_ids]
        prod_km_exprs = [
            PREFIXES["param_value"] + "km" + squash(km_id)
            for km_id in prod_km_ids
        ]
        prod_conc_exprs = get_conc_expressions(km, prod_ids)
        denom_prod_cpt = (
            "*".join(
                f"(1+{conc_expr}/{km_expr})^{np.abs(S.loc[prod_id, er_id])}"
                for conc_expr, km_expr, prod_id in zip(
                    prod_conc_exprs, prod_km_exprs, prod_ids
                )
            )
            + "-1"
        )
    else:
        denom_prod_cpt = "constzero"
    return f"1/(({denom_sub_cpt})+({denom_ci_cpt})+({denom_prod_cpt}))"


def get_saturation(er_id: str, km: KineticModel) -> str:
    """Get the saturation component for an enzyme."""
    er = next(e for e in km.edges if e.id == er_id)
    assert isinstance(er, EnzymeReaction), f"{er_id} is not an EnzymeReaction"
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    reaction = next(r for r in km.reactions if r.id == er.reaction_id)
    free_enzyme_ratio = get_free_enzyme_ratio(er_id, km)
    sub_ids = list(k for k, v in reaction.stoichiometry.items() if v < 0)
    sub_km_ids = [ID_SEPARATOR.join([enzyme.id, s]) for s in sub_ids]
    sub_conc_exprs = get_conc_expressions(km, sub_ids)
    sub_km_exprs = [
        PREFIXES["param_value"] + "km" + squash(sub_km_id)
        for sub_km_id in sub_km_ids
    ]
    sub_concs_over_kms = [
        f"({conc_expr}/{km_expr})"
        for conc_expr, km_expr in zip(sub_conc_exprs, sub_km_exprs)
    ]
    prod_of_sub_concs_over_kms = f"({'*'.join(sub_concs_over_kms)})"
    return f"({prod_of_sub_concs_over_kms}*{free_enzyme_ratio})"


def get_phosphorylation(km: KineticModel, er: EnzymeReaction) -> str:
    """Get the phosphorylation component for an enzyme in an experiment."""
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    if km.phosphorylations is None:
        return "(1)"
    phosphorylations = [
        p for p in km.phosphorylations if p.modified_enzyme_id == enzyme.id
    ]
    beta_cpts = []
    alpha_cpts = []
    for phosphorylation in phosphorylations:
        kcat_pme = (
            PREFIXES["param_value"]
            + "kcatpme"
            + squash(phosphorylation.modifying_enzyme_id)
        )
        conc_pme = (
            PREFIXES["param_value"]
            + "concpme"
            + squash(phosphorylation.modifying_enzyme_id)
        )
        kcat_times_conc = f"{kcat_pme}*{conc_pme}"
        if phosphorylation.modification_type == ModificationType.Inhibition:
            alpha_cpts += [kcat_times_conc]
        else:
            beta_cpts += [kcat_times_conc]
        alpha = "+".join(alpha_cpts)
        beta = "+".join(beta_cpts)
    if len(beta_cpts) == 0:
        return "(1)"
    return f"({beta}/({alpha}+{beta}))^{str(enzyme.subunits)}"
