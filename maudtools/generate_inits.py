import os
from typing import Dict, List, Optional

import pandas as pd
import arviz as az

from maud.loading_maud_inputs import load_maud_input
from maud.analysis import load_infd


def get_2d_coords(coords_1, coords_2):
    """Return unpacked coordinates for 2-D indexing."""
    set_of_coords = []
    for c1 in coords_1:
        for c2 in coords_2:
            set_of_coords.append((c1, c2))
    return list(zip(*set_of_coords)) if len(set_of_coords) > 0 else ([], [])


def generate_inits(data_path, chain, draw, warmup):
    """Generate template for init definitions.

    :params data_path: a path to a maud output folder with both samples
    and user_input folders
    :params chain: the sampling chain of the stan sampler you want to
    export
    :params draw: the sampling draw of the sampling chain you want to
    export from the start of the sampling or warmup phase
    :params warmup: indicator variable of if it is for the warmup
    or sampling phase
    """

    output_name = "generated_inits.csv"
    output_path = os.path.join(data_path, output_name)
    print("Creating init")

    idata = az.InferenceData.from_netcdf(os.path.join(data_path, "idata.nc"))
    idata_parameters = {k: v for k, v in idata.posterior.variables.items()}.keys()
    idata_draw = idata.posterior.sel(chain=chain, draw=draw)
    mi = load_maud_input(os.path.join(data_path, "user_input"))
    experiments = [e.id for e in mi.measurements.experiments if e.is_train]

    par_dataframe = pd.DataFrame(columns = ["parameter", "metabolite", "compartment", "enzyme", "reaction", "experiment", "value"])
    for par in idata_parameters:
        if par in dir(mi.stan_variable_set):
            if len(getattr(mi.stan_variable_set, par).ids) == 1:
                tmp_df = pd.DataFrame()
                for p in getattr(mi.stan_variable_set, par).ids[0]: 
                    tmp_df=tmp_df.append(
                        pd.DataFrame(
                            {k.value: v for k, v in zip(getattr(mi.stan_variable_set, par).id_components[0], 
                                                        p.split("_"))}, 
                            index=[0]))
                    if "experiment" in tmp_df.columns:
                        tmp_df = tmp_df.loc[tmp_df["experiment"].isin(experiments)]
                tmp_df["value"] = getattr(idata_draw, par).values
                tmp_df["parameter"] = par
                par_dataframe = pd.concat([par_dataframe, tmp_df])

            else:
                for exp in getattr(mi.stan_variable_set, par).ids[0]:
                    if exp in experiments:
                        tmp_df = pd.DataFrame()
                        for p1 in getattr(mi.stan_variable_set, par).ids[1]: 
                            tmp_df=tmp_df.append(
                                pd.DataFrame(
                                    {pc1.value: p1id for pc1, p1id in zip(getattr(mi.stan_variable_set, par).id_components[1], 
                                                                p1.split("_"))}, 
                                    index=[0]))
                        tmp_df["experiment"] = exp
                        tmp_df["value"] = getattr(idata_draw, par).values[experiments.index(exp)]
                        tmp_df["parameter"] = par
                        par_dataframe = pd.concat([par_dataframe, tmp_df])

    print(f"Saving inits to: {output_path}")
    par_dataframe.to_csv(output_path)
    return "Successfully generated prior template"
