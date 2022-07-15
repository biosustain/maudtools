"""Functions for generating initial values from draws."""

import os

import arviz as az
import pandas as pd
from maud.data_model.maud_input import MaudInput


def generate_inits(
    idata: az.InferenceData, mi: MaudInput, chain: int, draw: int, warmup: int
) -> pd.DataFrame:
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
    idata_draw = idata.posterior.sel(chain=chain, draw=draw)
    training_experiments = [
        e.id for e in mi.measurements.experiments if e.is_train
    ]
    out = pd.DataFrame()
    params = set(idata.posterior.variables.keys()).intersection(
        set(dir(mi.stan_variable_set))
    )
    for param in params:
        sv = getattr(mi.stan_variable_set, param)
        param_df = pd.DataFrame(getattr(idata_draw, param))
        if len(sv.shape_names) > 1:
            col_ix_names = [idc.value for idc in sv.id_components[1]]
            param_df.index = pd.Index(training_experiments, name="experiment")
            param_df.columns = (
                pd.MultiIndex.from_arrays(sv.split_ids[1], names=col_ix_names)
                if sv.split_ids is not None
                else pd.Index(sv.ids[1], name=col_ix_names[0])
            )
            param_df = param_df.stack().rename("value").reset_index()
        else:
            ix_names = [idc.value for idc in sv.id_components[0]]
            ixs = sv.split_ids if sv.split_ids is not None else [sv.ids[0]]
            ix_df = pd.DataFrame(dict(zip(ix_names, ixs)))
            if "experiment" in ix_names:
                ix_df = ix_df.loc[
                    lambda df: df["experiment"].isin(training_experiments)
                ]
            param_df.index = pd.MultiIndex.from_frame(ix_df)
            param_df = param_df[0].rename("value").reset_index()
        param_df["parameter"] = param
        out = pd.concat([out, param_df], ignore_index=True)
    return out
