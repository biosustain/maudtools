from typing import Tuple, Type, Union

import arviz as az
import numpy as np
import pandas as pd
from maud.data_model.maud_parameter import MaudParameter
from maud.data_model.prior import PriorMVN
from plotnine import aes, facet_wrap, geom_pointrange, geom_violin, ggplot


def concat_experiments(infd, x_var: str = "reactions"):
    with_exp = {
        exp.item(): pd.DataFrame(
            np.concatenate(infd[:, :, i, :]), columns=infd[x_var]
        )
        for i, exp in enumerate(infd.experiments)
    }
    list(map(lambda x: x[1].insert(0, "experiment", x[0]), with_exp.items()))
    return pd.concat(with_exp.values())


def get_aes_keys(
    infd: az.InferenceData, param: MaudParameter
) -> Tuple[bool, bool, str, str]:
    xa = infd.posterior[param.name]
    posterior_keys = list(xa.xindexes.keys())
    with_experiment = "experiments" in posterior_keys
    x_var = param.name
    key = (set(posterior_keys) - {"experiments", "chain", "draw"}).pop()
    return with_experiment, param.non_negative, x_var, key


def plot_maud_posterior(
    infd: az.InferenceData,
    param: Type[MaudParameter],
) -> ggplot:
    """Plot a `MaudParameter` posterior and priors distributions as violins.

    Example
    -------
    Plot the ΔG formation energy.

    >>> import arviz
    >>> from maud.loading_maud_inputs import load_maud_input
    >>> from maud.data_model.param_set import Dgf
    >>> from maudtools.plotting import plot_maud_posterior
    >>> infd = arviz.from_netcdf(SAMPLES_DIR / ".." / "idata.nc")
    >>> mi = load_maud_input(DATA_DIR)
    >>> plot_maud_posterior(infd, mi.priors, Dgf)
    """
    with_experiment, _, x_var, key = get_aes_keys(infd, param)
    xa = infd.posterior[x_var]

    if with_experiment:
        df_posterior = concat_experiments(xa, key)
        df_posterior = df_posterior.melt(
            id_vars=["experiment"], var_name=x_var, value_name=key
        )
    else:
        df_posterior = pd.DataFrame(np.concatenate(xa), columns=xa[key])
        df_posterior = df_posterior.melt(var_name=x_var, value_name=key)

    if with_experiment:
        plot = ggplot(df_posterior, aes(x_var, key, fill="experiment"))
    else:
        plot = ggplot(df_posterior, aes(x_var, key))
    plot += geom_violin(color="#AAAAAA", scale="width")
    if with_experiment:
        plot += facet_wrap("~experiment")
    return plot


def plot_maud_prior(
    infd: az.InferenceData,
    param: MaudParameter,
    concat: bool = True,
) -> Union[ggplot, geom_pointrange]:
    """Plot a `MaudParameter` priors distributions as point or 95% quantiles.

    Example
    -------
    Plot the ΔG formation energy prior (without posteriors).

    >>> import arviz
    >>> from maud.loading_maud_inputs import load_maud_input
    >>> from maud.data_model.param_set import Dgf
    >>> from maudtools.plotting import plot_maud_prior
    >>> infd = arviz.from_netcdf(SAMPLES_DIR / ".." / "idata.nc")
    >>> mi = load_maud_input(DATA_DIR)
    >>> plot_maud_prior(infd, mi.priors, Dgf, concat=False)

    If `concat=True` (default), the plot can be appended to another plot, like
    the posterior plot.

    >>> from maudtools.plotting import plot_maud_posterior
    >>> plot_maud_posterior(infd, Dgf) + plot_maud_prior(infd, mi.priors, Dgf)
    """
    with_experiment, non_negative, x_var, key = get_aes_keys(infd, param)

    if with_experiment:
        index = pd.Index(param.ids[0], name="experiment")
        columns = pd.Index(param.ids[1], name=x_var)
        loc_df = (
            pd.DataFrame(param.prior.location, index=index, columns=columns)
            .reset_index()
            .melt(id_vars=["experiment"], var_name=x_var, value_name="location")
        )
        scale_df = (
            pd.DataFrame(
                0 if isinstance(param.prior, PriorMVN) else param.prior.scale,
                index=index,
                columns=columns,
            )
            .reset_index()
            .melt(id_vars=["experiment"], var_name=x_var, value_name="scale")
        )
        df_priors = pd.merge(loc_df, scale_df, on=["experiment", x_var])
    else:
        index = pd.Index(param.ids[0], name=x_var)
        df_priors = pd.DataFrame(
            {
                "location": param.prior.location,
                "scale": param.prior.scale
                if not isinstance(param.prior, PriorMVN)
                else 0,
            },
            index=index,
        ).reset_index()
    # prepare scatter intervals
    if non_negative:
        df_priors.location = np.exp(df_priors.location)
        df_priors.scale = np.exp(df_priors.scale)
        df_priors["y_max"] = df_priors.location / df_priors.scale**2
        df_priors["y_min"] = df_priors.location * df_priors.scale**2
    else:
        df_priors["y_max"] = df_priors.location + 2 * df_priors.scale
        df_priors["y_min"] = df_priors.location - 2 * df_priors.scale
    if concat:
        return geom_pointrange(
            df_priors,
            aes(x_var, "location", ymin="y_min", ymax="y_max"),
            fill="white",
        )
    if with_experiment:
        plot = ggplot(df_priors, aes(x_var, key, fill="experiment"))
    else:
        plot = ggplot(df_priors, aes(x_var, key))
    plot += geom_pointrange(
        aes(x_var, "location", ymin="y_min", ymax="y_max"),
        fill="white",
    )
    if with_experiment:
        plot += facet_wrap("~experiment")
    return plot


def plot_maud_variable(infd: az.InferenceData, param: MaudParameter):
    return plot_maud_posterior(infd, param) + plot_maud_prior(infd, param, True)
