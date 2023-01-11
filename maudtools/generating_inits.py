"""Functions for generating initial values from draws."""
from dataclasses import fields

import arviz as az
from maud.data_model.maud_input import MaudInput


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
        init_input[param.name.replace("_train", "")] = param_init_input_dict
    return init_input
