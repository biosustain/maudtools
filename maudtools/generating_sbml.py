"""Provides the function generate_sbml.

This function takes in an InferenceData, MaudInput, chain, draw and experiment
and returns an SBML file in string format.

"""
import warnings
from typing import List, Tuple

import arviz as az
import libsbml as sbml  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr
from maud.data_model.hardcoding import ID_SEPARATOR  # type: ignore
from maud.data_model.kinetic_model import EnzymeReaction  # type:ignore
from maud.data_model.kinetic_model import (KineticModel, ModificationType,
                                           Reaction, ReactionMechanism)
from maud.data_model.maud_input import MaudInput  # type: ignore

SBML_VARS = [
    "dgrs",
    "drain",
    "kcat",
    "km",
    "ki",
    "dissociation_constant",
    "transfer_constant",
    "conc_unbalanced",
    "conc_enzyme",
]


def generate_sbml(
    idata: az.InferenceData,
    mi: MaudInput,
    experiment_id: str,
    chain: int,
    draw: int,
    warmup: int,
) -> Tuple[sbml.SBMLDocument, sbml.Model]:
    """Run the main function of this module."""
    posterior = idata.posterior if not warmup else idata.warmup_posterior  # type: ignore
    draw = posterior.sel(chain=chain, draw=draw, experiments=experiment_id)
    assert isinstance(draw, xr.Dataset)
    param_df = get_parameter_df(draw)
    experiment_ix, experiment = next(
        (i, exp)
        for i, exp in enumerate(mi.measurements.experiments)
        if exp.id == experiment_id
    )
    doc, model = initialise_model(mi)
    add_parameters_to_model(model, param_df)
    add_species_to_model(model, mi, experiment_ix)
    add_reactions_to_model(model, mi, param_df, experiment.temperature)

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
        cpt = model.createCompartment()
        cpt.setId(maud_cpt.id)
        cpt.setConstant(True)
        cpt.setSize(maud_cpt.volume)
    return doc, model


def add_parameters_to_model(model: sbml.Model, param_df: pd.DataFrame):
    """Add parameters to an sbml model from a dataframe."""
    zero = model.createParameter()
    zero.setId("zero")
    zero.setName("zero")
    zero.setConstant(True)
    zero.setValue(0)
    for _, row in param_df.iterrows():
        param = model.createParameter()
        pid, pval = row["pid"], row["value"]
        if param.setId(pid) != sbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(f"Unable to generate parameter with id {pid}.")
        param.setName(pid)
        param.setConstant(True)
        param.setValue(pval)


def add_species_to_model(model: sbml.Model, mi: MaudInput, experiment_ix: int):
    """Add species to a model with initial concs from an experiment."""
    conc_init = mi.stan_input_train.conc_init.value
    balanced_mic_ix, balanced_mics = zip(
        *[(i, m) for i, m in enumerate(mi.kinetic_model.mics) if m.balanced]
    )
    for i, mic in zip(balanced_mic_ix, balanced_mics):
        spid = "s" + mic.id.replace("_", "")
        sp = model.createSpecies()
        sp.setCompartment(mic.compartment_id)
        sp.setId(spid)
        sp.setName(spid)
        sp.setInitialAmount(conc_init[experiment_ix][i])


def add_reactions_to_model(
    model: sbml.Model, mi: MaudInput, param_df: pd.DataFrame, temperature: float
):
    """Add reactions to an sbml model."""
    for edge in mi.kinetic_model.edges:
        maud_rxn_id = (
            edge.reaction_id if isinstance(edge, EnzymeReaction) else edge.id
        )
        maud_rxn = next(
            r for r in mi.kinetic_model.reactions if r.id == maud_rxn_id
        )
        sbml_rxn_id = "r" + edge.id.replace("_", "")
        sbml_rxn = model.createReaction()
        sbml_rxn.setId(sbml_rxn_id)
        for mic_id, stoic in maud_rxn.stoichiometry.items():
            mic = next(m for m in mi.kinetic_model.mics if m.id == mic_id)
            spid = "s" + mic.id.replace("_", "")
            if mic.balanced and stoic < 0:
                spr = sbml_rxn.createReactant()
                spr.setSpecies(spid)
            elif mic.balanced and stoic > 0:
                spr = sbml_rxn.createProduct()
                spr.setSpecies(spid)
        # handle modifiers
        for mic in filter(lambda m: m.balanced, mi.kinetic_model.mics):
            spid = "s" + mic.id.replace("_", "")
            for ci in mi.kinetic_model.competitive_inhibitions:
                if ci.mic_id == mic.id:
                    mfr = sbml_rxn.createModifier()
                    mfr.setSpecies(spid)
            for al in mi.kinetic_model.allosteries:
                if al.mic_id == mic.id:
                    mfr = sbml_rxn.createModifier()
                    mfr.setSpecies(spid)
        kl = sbml_rxn.createKineticLaw()
        flux_expr = get_edge_flux(param_df, mi, edge.id, temperature)
        math_ast = sbml.parseL3Formula(flux_expr)
        if math_ast is None:
            raise RuntimeError(
                f"Unable to generate flux expression for reaction {sbml_rxn_id}"
            )
        kl.setMath(math_ast)


def get_parameter_df(draw: xr.Dataset) -> pd.DataFrame:
    """Get a dataframe of parameter ids and values."""
    return (
        pd.concat(
            {
                v: draw[v].to_series().rename("value")  # type: ignore
                for v in SBML_VARS
                if v in draw.data_vars
            }
        )
        .reset_index()
        .rename(columns={"level_0": "parameter", "level_1": "id"})
        .assign(
            pid=lambda df: "p"
            + df["parameter"]
            .str.replace("-", "")
            .str.replace("_", "")
            .str.cat(df["id"].str.replace("-", "").str.replace("_", ""))
        )
    )


def get_edge_flux(
    parameter_df: pd.DataFrame, mi: MaudInput, edge_id: str, temperature: float
) -> str:
    """Get the flux for an edge."""
    edge = next(e for e in mi.kinetic_model.edges if e.id == edge_id)
    if isinstance(edge, Reaction):
        return get_drain_flux(
            parameter_df,
            mi.kinetic_model,
            edge,
            mi.config.drain_small_conc_corrector,
        )
    elif isinstance(edge, EnzymeReaction):
        return get_enzyme_reaction_flux(parameter_df, mi, edge, temperature)
    else:
        raise ValueError("Input must be either a Reaction or an EnzymeReactoin")


def get_drain_flux(
    param_df: pd.DataFrame, km: KineticModel, drain: Reaction, corrector: float
) -> str:
    """Get an expression for the flux for a drain edge."""
    drain_val = lookup_param(param_df, "drain", [drain.id])["pid"].iloc[0]
    sub_ids = list(k for k, v in drain.stoichiometry.items() if v < 0)
    if len(sub_ids) == 0:
        return f"({drain_val})"
    sub_conc_exprs = get_conc_expressions(param_df, km, sub_ids)
    corrector_cpts = [
        f"({conc}/({conc}+{str(corrector)}))" for conc in sub_conc_exprs
    ]
    corrector_term = "*".join(corrector_cpts)
    return f"({drain_val}*({corrector_term}))"


def lookup_param(
    param_df: pd.DataFrame, param: str, ids: List[str]
) -> pd.DataFrame:
    """Get parameter entries for a parameter and ids."""
    return param_df.loc[  # type: ignore
        lambda df: (df["parameter"] == param) & (df["id"].isin(ids))
    ]


def get_enzyme_reaction_flux(
    parameter_df: pd.DataFrame,
    mi: MaudInput,
    edge: EnzymeReaction,
    temperature: float,
) -> str:
    """Get an expression for the flux for an EnzymeReaction edge."""
    flux_components = [
        get_enzyme_concentration(parameter_df, edge.enzyme_id),
        get_kcat(parameter_df, edge.enzyme_id),
        get_reversibility(parameter_df, mi.kinetic_model, edge, temperature),
        get_saturation(parameter_df, edge.id, mi.kinetic_model),
        get_allostery(parameter_df, mi.kinetic_model, edge),
        get_phosphorylation(parameter_df, mi.kinetic_model, edge),
    ]
    return f"({'*'.join(flux_components)})"


def get_enzyme_concentration(param_df: pd.DataFrame, enzyme_id: str) -> str:
    """Get the concentration for an enzyme."""
    return lookup_param(param_df, "conc_enzyme", [enzyme_id])["pid"].iloc[0]


def get_kcat(param_df: pd.DataFrame, enzyme_id: str) -> str:
    """Get an expression for kcat."""
    return lookup_param(param_df, "kcat", [enzyme_id])["pid"].iloc[0]


def get_reversibility(
    parameter_df: pd.DataFrame,
    km: KineticModel,
    edge: EnzymeReaction,
    temperature: float,
) -> str:
    """Get the reversibility for a reaction in an experiment."""
    RT = str(0.008314 * temperature)
    reaction = next(r for r in km.reactions if r.id == edge.reaction_id)
    if reaction.mechanism == ReactionMechanism.IRREVERSIBLE_MICHAELIS_MENTEN:
        return "1"
    mic_ids = list(reaction.stoichiometry.keys())
    stoics = list(reaction.stoichiometry.values())
    dgr_expr = lookup_param(parameter_df, "dgrs", [edge.id])["pid"].iloc[0]
    conc_exprs = get_conc_expressions(parameter_df, km, mic_ids)
    reaction_quotient_cpts = [
        f"({stoic}*ln({conc_expr}))"
        for stoic, conc_expr in zip(stoics, conc_exprs)
    ]
    reaction_quotient_expression = f"({'+'.join(reaction_quotient_cpts)})"
    return f"(1-exp(({dgr_expr}+{RT}*{reaction_quotient_expression})/{RT}))"


def get_allostery(
    param_df: pd.DataFrame, km: KineticModel, er: EnzymeReaction
) -> str:
    """Get the allostery component for an enzyme reaction in an experiment."""
    allosteries = [a for a in km.allosteries if a.enzyme_id == er.enzyme_id]
    if len(allosteries) == 0:
        return "(1)"
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    fer = get_free_enzyme_ratio(param_df, er.id, km)
    tc = lookup_param(param_df, "transfer_constant", [er.enzyme_id])[
        "pid"
    ].iloc[0]
    Qnum_cpts = ["1"]
    Qdenom_cpts = ["1"]
    for allostery in allosteries:
        conc = get_conc_expressions(param_df, km, [allostery.mic_id])[0]
        dc = lookup_param(param_df, "dissociation_constant", [allostery.id])[
            "pid"
        ].iloc[0]
        if allostery.modification_type == ModificationType.ACTIVATION:
            Qdenom_cpts += [f"{conc}/{dc}"]
        else:
            Qnum_cpts += [f"{conc}/{dc}"]
        Qnum = f"({'+'.join(Qnum_cpts)})"
        Qdenom = f"({'+'.join(Qdenom_cpts)})"
    return f"(1/(1+{tc}*({fer}*{Qnum}/{Qdenom})^{str(enzyme.subunits)}))"


def get_conc_expressions(
    param_df: pd.DataFrame, km: KineticModel, mic_ids: List[str]
) -> List[str]:
    """Get expressions for the concentrations of some mic ids.

    This job needs its own function because some mic concentrations are state
    variables and others are parameters. Specifically, the state variables are
    the balanced mics.

    """
    balanced_mic_ids = [m.id for m in km.mics if m.balanced]
    return [
        "s" + mic_id.replace("-", "").replace("_", "")
        if mic_id in balanced_mic_ids
        else "pconcunbalanced" + mic_id.replace("-", "").replace("_", "")
        for mic_id in mic_ids
    ]


def get_free_enzyme_ratio(
    param_df: pd.DataFrame, er_id: str, km: KineticModel
) -> str:
    """Get the free enzyme ratio for an EnzymeReaction."""
    er = next(e for e in km.edges if e.id == er_id)
    S = km.stoichiometric_matrix
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    reaction = next(r for r in km.reactions if r.id == er.reaction_id)
    sub_ids = list(k for k, v in reaction.stoichiometry.items() if v < 0)
    sub_km_ids = [ID_SEPARATOR.join([enzyme.id, s]) for s in sub_ids]
    sub_conc_exprs = get_conc_expressions(param_df, km, sub_ids)
    sub_km_exprs = lookup_param(param_df, "km", sub_km_ids)["pid"].values
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
        ci_conc_exprs = get_conc_expressions(param_df, km, ci_mic_ids)
        ki_exprs = lookup_param(param_df, "ki", [ci.id for ci in cis])[
            "pid"
        ].values
        denom_ci_cpt = "+".join(
            f"({conc_expr}/{ki_expr})"
            for conc_expr, ki_expr in zip(ci_conc_exprs, ki_exprs)
        )
    else:
        denom_ci_cpt = "zero"
    if reaction.mechanism == ReactionMechanism.REVERSIBLE_MICHAELIS_MENTEN:
        prod_ids = list(k for k, v in reaction.stoichiometry.items() if v > 0)
        prod_km_ids = [ID_SEPARATOR.join([enzyme.id, p]) for p in prod_ids]
        prod_km_exprs = lookup_param(param_df, "km", prod_km_ids)["pid"].values
        prod_conc_exprs = get_conc_expressions(param_df, km, prod_ids)
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
        denom_prod_cpt = "zero"
    return f"1/(({denom_sub_cpt})+({denom_ci_cpt})+({denom_prod_cpt}))"


def get_saturation(param_df: pd.DataFrame, er_id: str, km: KineticModel) -> str:
    """Get the saturation component for an enzyme."""
    er = next(e for e in km.edges if e.id == er_id)
    assert isinstance(er, EnzymeReaction), f"{er_id} is not an EnzymeReaction"
    enzyme = next(e for e in km.enzymes if e.id == er.enzyme_id)
    reaction = next(r for r in km.reactions if r.id == er.reaction_id)
    free_enzyme_ratio = get_free_enzyme_ratio(param_df, er_id, km)
    sub_ids = list(k for k, v in reaction.stoichiometry.items() if v < 0)
    sub_km_ids = [ID_SEPARATOR.join([enzyme.id, s]) for s in sub_ids]
    sub_conc_exprs = get_conc_expressions(param_df, km, sub_ids)
    sub_km_exprs = lookup_param(param_df, "km", sub_km_ids)["pid"].values
    sub_concs_over_kms = [
        f"({conc_expr}/{km_expr})"
        for conc_expr, km_expr in zip(sub_conc_exprs, sub_km_exprs)
    ]
    prod_of_sub_concs_over_kms = f"({'*'.join(sub_concs_over_kms)})"
    return f"({prod_of_sub_concs_over_kms}*{free_enzyme_ratio})"


def get_phosphorylation(
    param_df: pd.DataFrame, km: KineticModel, er: EnzymeReaction
) -> str:
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
        kcat_pme = lookup_param(
            param_df, "kcat_pme", [phosphorylation.modifying_enzyme_id]
        )["pid"].iloc[0]
        conc_pme = lookup_param(
            param_df, "conc_pme", phosphorylation.modifying_enzyme_id
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
