import os

from maud.getting_idatas import get_idata
from maud.user_templates import get_inits_from_draw

INIT_FILE_COLUMNS = [
    "parameter_name",
    "experiment_id",
    "metabolite_id",
    "mic_id",
    "enzyme_id",
    "phos_enz_id",
    "drain_id",
    "value",
]

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

    csvs = [
        os.path.join(data_path, "samples", f)
        for f in os.listdir(os.path.join(data_path, "samples"))
        if f.endswith(".csv")
    ]
    mi = load_maud_input(os.path.join(data_path, "user_input"), mode="sample")
    idata = get_idata(csvs, mi)
    output_name = "generated_inits.csv"
    output_path = os.path.join(data_path, output_name)
    print("Creating init")
    init_dataframe = get_inits_from_draw(idata, mi, chain, draw, warmup)
    print(f"Saving inits to: {output_path}")
    init_dataframe.to_csv(output_path)
    return "Successfully generated prior template"


def get_inits_from_draw(infd, mi, chain, draw, warmup):
    """Extact parameters from an infd object."""

    scs = mi.stan_coords
    list_of_input_inits = get_parameter_coords(scs)
    init_dataframe = pd.DataFrame(columns=INIT_FILE_COLUMNS)
    if warmup == 1:
        infd_parameters = list(infd.warmup_posterior.variables.keys())
    else:
        infd_parameters = list(infd.posterior.variables.keys())
    for par in list_of_input_inits:
        if par.id in infd_parameters:
            if warmup == 1:
                value_dataframe = (
                    infd.warmup_posterior[par.id][chain][draw]
                    .to_dataframe()
                    .reset_index()
                )

            else:
                value_dataframe = (
                    infd.posterior[par.id][chain][draw].to_dataframe().reset_index()
                )
            par_dataframe = pd.DataFrame.from_dict(par.coords)
            if par.linking_list:
                par_dataframe["linking_list"] = list(par.linking_list.values())[0]
                par_dataframe = par_dataframe.merge(
                    value_dataframe,
                    left_on="linking_list",
                    right_on=par.infd_coord_list,
                )
            else:
                par_dataframe = par_dataframe.merge(
                    value_dataframe,
                    left_on=list(par.coords.keys()),
                    right_on=par.infd_coord_list,
                )
            par_dataframe["parameter_name"] = par.id
            par_dataframe["value"] = par_dataframe[par.id]
            init_column_list = ["parameter_name"] + list(par.coords.keys()) + ["value"]
            init_dataframe = init_dataframe.append(par_dataframe[init_column_list])
    return init_dataframe


def get_parameter_coords(scs):
    """Define parameter coordinates for stan and infd objects."""
    return [
        Input_Coords(
            id="km",
            coords={"enzyme_id": scs.km_enzs, "mic_id": scs.km_mics},
            infd_coord_list=["kms"],
            linking_list={"kms": join_list_of_strings(scs.km_enzs, scs.km_mics)},
        ),
        Input_Coords(
            id="drain",
            coords={
                "drain_id": get_2d_coords(scs.drains, scs.experiments)[0],
                "experiment_id": get_2d_coords(scs.drains, scs.experiments)[1],
            },
            infd_coord_list=["drains", "experiments"],
        ),
        Input_Coords(
            id="ki",
            coords={"enzyme_id": scs.ci_enzs, "mic_id": scs.ci_mics},
            infd_coord_list=["kis"],
            linking_list={"kis": join_list_of_strings(scs.ci_enzs, scs.ci_mics)},
        ),
        Input_Coords(
            id="diss_t",
            coords={"enzyme_id": scs.ai_enzs, "mic_id": scs.ai_mics},
            infd_coord_list=["diss_ts"],
            linking_list={"diss_ts": join_list_of_strings(scs.ai_enzs, scs.ai_mics)},
        ),
        Input_Coords(
            id="diss_r",
            coords={"enzyme_id": scs.aa_enzs, "mic_id": scs.aa_mics},
            infd_coord_list=["diss_rs"],
            linking_list={"diss_rs": join_list_of_strings(scs.aa_enzs, scs.aa_mics)},
        ),
        Input_Coords(
            id="transfer_constant",
            coords={"enzyme_id": scs.allosteric_enzymes},
            infd_coord_list=["allosteric_enzymes"],
        ),
        Input_Coords(
            id="kcat",
            coords={"enzyme_id": scs.enzymes},
            infd_coord_list=["enzymes"],
        ),
        Input_Coords(
            id="kcat_phos",
            coords={"phos_enz_id": scs.phos_enzs},
            infd_coord_list=["phos_enzs"],
        ),
        Input_Coords(
            id="conc_unbalanced",
            coords={
                "mic_id": get_2d_coords(scs.unbalanced_mics, scs.experiments)[0],
                "experiment_id": get_2d_coords(scs.unbalanced_mics, scs.experiments)[1],
            },
            infd_coord_list=["unbalanced_mics", "experiments"],
        ),
        Input_Coords(
            id="conc_enzyme",
            coords={
                "enzyme_id": get_2d_coords(scs.enzymes, scs.experiments)[0],
                "experiment_id": get_2d_coords(scs.enzymes, scs.experiments)[1],
            },
            infd_coord_list=["enzymes", "experiments"],
        ),
        Input_Coords(
            id="conc_phos",
            coords={
                "phos_enz_id": get_2d_coords(scs.phos_enzs, scs.experiments)[0],
                "experiment_id": get_2d_coords(scs.phos_enzs, scs.experiments)[1],
            },
            infd_coord_list=["phos_enzs", "experiments"],
        ),
        Input_Coords(
            id="dgf",
            coords={"metabolite_id": scs.metabolites},
            infd_coord_list=["metabolites"],
        ),
        Input_Coords(id="keq", coords={"edges": scs.edges}, infd_coord_list=["edges"]),
    ]
