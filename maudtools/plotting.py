from typing import Type

import arviz as az
import numpy as np
import pandas as pd
from maud.data_model.prior_set import PriorSet
from maud.data_model.stan_variable_set import StanVariable
from plotnine import aes, facet_wrap, geom_pointrange, geom_violin, ggplot


def concat_experiments(infd, x_var: str = "reactions"):
    with_exp = {
        exp.item(): pd.DataFrame(np.concatenate(infd[:, :, i, :]), columns=infd[x_var])
        for i, exp in enumerate(infd.experiments)
    }
    list(map(lambda x: x[1].insert(0, "experiment", x[0]), with_exp.items()))
    return pd.concat(with_exp.values())


def plot_maud_posterior(
    infd: az.InferenceData,
    priors: PriorSet,
    stan_variable: Type[StanVariable],
):
    """Plot a `StanVariable` posterior and priors distributions as violins.

    Example
    -------
    Plot the ΔG formation energy.

    >>> import arviz
    >>> from maud.loading_maud_inputs import load_maud_input
    >>> from maud.data_model.stan_variable_set import Dgf
    >>> from maudtools.plotting import plot_maud_posterior
    >>> infd = arviz.from_netcdf(SAMPLES_DIR / ".." / "idata.nc")
    >>> mi = load_maud_input(DATA_DIR)
    >>> plot_maud_posterior(infd, mi.priors, Dgf)
    """
    try:
        var = stan_variable("")
    except TypeError:
        var = stan_variable("", "_")
    xa = infd.posterior[var.name]
    posterior_keys = list(xa.xindexes.keys())
    with_experiment = "experiments" in posterior_keys
    x_var = var.name
    key = (set(posterior_keys) - {"experiments", "chain", "draw"}).pop()

    # tidy posteriors
    if with_experiment:
        df_posterior = concat_experiments(xa, key)
        df_posterior = df_posterior.melt(
            id_vars=["experiment"], var_name=x_var, value_name=key
        )
    else:
        df_posterior = pd.DataFrame(np.concatenate(xa), columns=xa[key])
        df_posterior = df_posterior.melt(var_name=x_var, value_name=key)

    # tidy priors
    var_priors = priors.__getattribute__(x_var)
    if with_experiment:
        var_priors.location["experiment"] = var_priors.location.index
        df_priors_loc = var_priors.location.melt(
            id_vars=["experiment"], var_name=x_var, value_name="location"
        )
        if "scale" in var_priors.__dict__:
            var_priors.scale["experiment"] = var_priors.scale.index
            df_priors_scale = var_priors.scale.melt(
                id_vars=["experiment"], var_name=x_var, value_name="scale"
            )
            df_priors = pd.merge(
                df_priors_loc, df_priors_scale, on=["experiment", x_var]
            )
    else:
        df_priors = pd.DataFrame(var_priors.location)
        df_priors.columns = ["location"]
        df_priors[x_var] = df_priors.index
        if "scale" in var_priors.__dict__:
            if isinstance(var_priors.scale, pd.Series):
                df_priors_scale = pd.DataFrame(var_priors.scale)
                df_priors_scale.columns = ["scale"]
                df_priors_scale[x_var] = df_priors.index
                df_priors = pd.merge(df_priors, df_priors_scale, on=[x_var])
    if "scale" not in df_priors.columns:
        # geom_poinrange will be transformed into a geom_point
        df_priors["scale"] = 0

    # prepare scatter intervals
    if var.non_negative:
        df_priors.location = np.exp(df_priors.location)
        df_priors.scale = np.exp(df_priors.scale)
        df_priors["y_max"] = df_priors.location / df_priors.scale ** 2
        df_priors["y_min"] = df_priors.location * df_priors.scale ** 2
    else:
        df_priors["y_max"] = df_priors.location + 2 * df_priors.scale
        df_priors["y_min"] = df_priors.location - 2 * df_priors.scale
    # df_posterior = filter_df(df_posterior)

    if with_experiment:
        plot = ggplot(df_posterior, aes(x_var, key, fill="experiment"))
    else:
        plot = ggplot(df_posterior, aes(x_var, key))
    plot = (
        plot
        + geom_violin(color="#AAAAAA", scale="width")
        + geom_pointrange(
            df_priors,
            aes(x_var, "location", ymin="y_min", ymax="y_max"),
            fill="white",
        )
    )
    if with_experiment:
        plot += facet_wrap("~experiment")
    return plot
