"""Functions for generating initial values from draws."""
import json
from dataclasses import asdict, fields

import arviz as az
import numpy as np
import pandas as pd
import toml
from maud.data_model.maud_init import InitAtomInput, InitInput
from maud.data_model.maud_input import MaudInput
from maud.data_model.maud_parameter import MaudParameter


def generate_inits(
    idata: az.InferenceData, mi: MaudInput, chain: int, draw: int, warmup: int
) -> dict:
    """Generate template for init definitions.

    :param idata: an arviz InferenceData created by Maud

    :param mi: a MaudInput object. This must come from the same run as the
    input idata

    :param chain: the sampling chain of the stan sampler you want to
    export

    :param draw: the sampling draw of the sampling chain you want to
    export from the start of the sampling or warmup phase

    :param warmup: indicator variable of if it is for the warmup
    or sampling phase

    """
    posterior = idata.warmup_posterior if warmup == 1 else idata.posterior
    idata_draw = posterior.sel(chain=chain, draw=draw)
    param_names = [
        f for f in fields(mi.parameters) if f.name in idata_draw.data_vars
    ]
    params = [getattr(mi.parameters, f.name) for f in param_names]
    init_input = {}
    for param in params:
        vals = getattr(idata_draw, param.name)
        if len(param.ids) == 1:
            idc_strings = [idc.value for idc in param.id_components[0]]
            param_init_input_dict = [
                {"init": val} | dict(zip(idc_strings, ids))
                for val, ids in zip(vals.values, zip(*param.split_ids[0]))
            ]
        else:
            idc_strings = [idc.value for idc in param.id_components[1]]
            param_init_input_dict = []
            for exp_id in param.ids[0]:
                exp_vals = vals.sel(experiments=exp_id)
                param_init_input_dict_exp = [
                    {"init": val, "experiment": exp_id}
                    | dict(zip(idc_strings, ids))
                    for val, ids in zip(
                        exp_vals.values.flatten(), zip(*param.split_ids[1])
                    )
                ]
                param_init_input_dict += param_init_input_dict_exp
        init_input[param.name] = param_init_input_dict
    return init_input


    #     for id_component in id_components
    #     id_components_repeated = itertools.repeat
    #     id_c = id_components_flat
    #     if len(param.shape_names) == 1:
    #         import pdb; pdb.set_trace()
    #         vals = getattr(idata_draw, param.name).values

    #         # param_df = pd.DataFrame(getattr(idata_draw, param))
    #     if len(param.shape_names) == 1:
    #         ix_names = [idc.value for idc in param.id_components[0]]
    #         col_ix_names = [idc.value for idc in sv.id_components[1]]
    #         param_df.index = pd.Index(training_experiments, name="experiment")
    #         param_df.columns = (
    #             pd.MultiIndex.from_arrays(sv.split_ids[1], names=col_ix_names)
    #             if sv.split_ids is not None
    #             else pd.Index(sv.ids[1], name=col_ix_names[0])
    #         )
    #         param_df = (
    #             param_df.stack(level=param_df.columns.names)
    #             .rename("value")
    #             .reset_index()
    #         )
    #     else:

    #         ixs = sv.split_ids if sv.split_ids is not None else [sv.ids[0]]
    #         ix_df = pd.DataFrame(dict(zip(ix_names, ixs)))
    #         if "experiment" in ix_names:
    #             ix_df = ix_df.loc[
    #                 lambda df: df["experiment"].isin(training_experiments)
    #             ]
    #         param_df.index = pd.MultiIndex.from_frame(ix_df)
    #         param_df = param_df[0].rename("value").reset_index()
    #     param_df["parameter"] = param
    #     out = pd.concat([out, param_df], ignore_index=True)
    # return out
