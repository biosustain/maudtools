"""Functions for generating a prior file format for kinetic parameters""" 

import pandas as pd
from maud.data_model.stan_variable_set import StanVariableSet


def get_kinetic_variable_set_dataframe(
    svs: StanVariableSet
):
    prior_dict_set = {}
    for par_id, sv in svs.__dict__.items():
        if (par_id != "__initialised__") and (sv is not None):
            prior_dict_set[par_id] = pd.DataFrame.from_dict({
                sv.id_components[0][i].value:
                sv.split_ids[i]
                for i in range(len(sv.id_components[0]))
                })
            prior_dict_set[par_id].insert(0, "parameter", par_id)
    return prior_dict_set


def generate_prior_template(
    svs: StanVariableSet
) -> pd.DataFrame:
    """Generate template for kinetic parameter input

    :param km: a KineticModel object.:

    """
    value_columns = [
            "location",
            "scale",
            "pct1",
            "pct99",
            "metadata",
        ]

    prior_dict_set = get_kinetic_variable_set_dataframe(svs)
    prior_template = pd.concat(prior_dict_set.values())
    for col in value_columns:
             prior_template[col] = pd.Series(dtype="object")
    return prior_template

