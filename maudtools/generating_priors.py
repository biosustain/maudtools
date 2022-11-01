"""Functions for generating a prior file format for kinetic parameters""" 

import pandas as pd
from maud.data_model.kinetic_model import KineticModel
from maud.data_model.stan_variable_set import Km,\
                                              Kcat,\
                                              Ki,\
                                              DissociationConstant,\
                                              TransferConstant,\
                                              KcatPme
from maud.parsing_kinetic_models import parse_kinetic_model
from maud.getting_stan_variables import get_km_coords,\
                                        get_dc_coords,\
                                        get_ci_coords,\
                                        get_kcat_coords
import toml


def get_kinetic_variable_set(
    km: KineticModel
):

    km_ids, km_enzs, km_mets, km_cpts = get_km_coords(km)
    dc_ids, dc_enzs, dc_mets, dc_cpts = get_dc_coords(km)
    ci_ids, ci_enzs, ci_rxns, ci_mets, ci_cpts = get_ci_coords(km)
    kcat_ids, kcat_enzs, kcat_rxns = get_kcat_coords(km)
    allosteric_enzyme_ids = (
        [e.id for e in km.allosteric_enzymes]
        if km.allosteric_enzymes is not None
        else []
    )
    phos_modifying_enzymes = (
        [p.modifying_enzyme_id for p in km.phosphorylations]
        if km.phosphorylations is not None
        else []
    )
    prior_dict = pd.DataFrame()
    km = Km(ids=[km_ids], split_ids=[km_enzs, km_mets, km_cpts])
    km_dict = {
        km.id_components[0][i].value: km.split_ids[i]
        for i in range(len(km.id_components[0]))
        }
    km_dict = pd.DataFrame.from_dict(km_dict)
    km_dict.insert(0, "parameter", "km")
    prior_dict = pd.concat([prior_dict, km_dict])
    kcat = Kcat(ids=[kcat_ids], split_ids=[kcat_enzs, kcat_rxns])
    kcat_dict = {
        kcat.id_components[0][i].value: kcat.split_ids[i]
        for i in range(len(kcat.id_components[0]))
        }
    kcat_dict = pd.DataFrame.from_dict(kcat_dict)
    kcat_dict.insert(0, "parameter", "kcat")
    prior_dict = pd.concat([prior_dict, kcat_dict])
    ki = Ki(ids=[ci_ids], split_ids=[ci_enzs, ci_rxns, ci_mets, ci_cpts])
    ki_dict = {
        ki.id_components[0][i].value: ki.split_ids[i]
        for i in range(len(ki.id_components[0]))
        }
    ki_dict = pd.DataFrame.from_dict(ki_dict)
    ki_dict.insert(0, "parameter", "ki")
    prior_dict = pd.concat([prior_dict, ki_dict])
    dissociation_constant = DissociationConstant(
        ids=[dc_ids], split_ids=[dc_enzs, dc_mets, dc_cpts]
    )
    dissociation_constant_dict = {
        dissociation_constant.id_components[0][i].value:
        dissociation_constant.split_ids[i]
        for i in range(len(dissociation_constant.id_components[0]))
        }
    dissociation_constant_dict = pd.DataFrame.from_dict(dissociation_constant_dict)
    dissociation_constant_dict.insert(0, "parameter", "dissociation_constant")
    prior_dict = pd.concat([prior_dict, dissociation_constant_dict])
    transfer_constant = TransferConstant(ids=[allosteric_enzyme_ids])
    transfer_constant_dict = {
        transfer_constant.id_components[0][i].value:
        transfer_constant.ids[i]
        for i in range(len(transfer_constant.id_components[0]))
        }
    transfer_constant_dict = pd.DataFrame.from_dict(transfer_constant_dict)
    transfer_constant_dict.insert(0, "parameter", "transfer_constant")
    prior_dict = pd.concat([prior_dict, transfer_constant_dict])
    kcat_pme = KcatPme(ids=[phos_modifying_enzymes])
    kcat_pme_dict = {
        kcat_pme.id_components[0][i].value: kcat_pme.ids[i]
        for i in range(len(kcat_pme.id_components[0]))
        }
    kcat_pme_dict = pd.DataFrame.from_dict(kcat_pme_dict)
    kcat_pme_dict.insert(0, "parameter", "kcat_pme")
    prior_dict = pd.concat([prior_dict, kcat_pme_dict])

    import pdb; pdb.set_trace()

    return


def generate_priors(
    km: KineticModel
) -> pd.DataFrame:
    """Generate template for kinetic parameter input

    :param km: a KineticModel object.

    """
    km = toml.load(km)
    km = parse_kinetic_model(km)
    get_kinetic_variable_set(km)

    return


def main():
    generate_priors("../tests/data/methionine_output/user_input/methionine_cycle.toml")

    return


if __name__ == "__main__":
    main()
