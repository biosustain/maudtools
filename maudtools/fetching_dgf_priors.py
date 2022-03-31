"""Provides the function fetch_dgf_priors_from_equilibrator."""

from typing import Tuple

import numpy as np
import pandas as pd
from equilibrator_api import ComponentContribution
from equilibrator_cache.models.compound import Compound

from maud.data_model import MaudInput


def fetch_dgf_priors_from_equilibrator(mi: MaudInput) -> Tuple[pd.Series, pd.DataFrame]:
    """Given a Maud input, get a multivariate prior from equilibrator.

    Returns a pandas Series of prior means and a pandas DataFrame of
    covariances. Both are indexed by metabolite ids.

    :param mi: A MaudInput object

    """
    cc = ComponentContribution()
    mu = []
    sigmas_fin = []
    sigmas_inf = []
    met_ix = pd.Index(mi.stan_coords.metabolites, name="metabolite")
    met_order = [m.id for m in mi.kinetic_model.metabolites]
    for m in mi.kinetic_model.metabolites:
        external_id = m.id if m.inchi_key is None else m.inchi_key
        c = cc.get_compound(external_id)
        if isinstance(c, Compound):
            mu_c, sigma_fin_c, sigma_inf_c = cc.standard_dg_formation(c)
            mu_c += c.transform(
                cc.p_h, cc.ionic_strength, cc.temperature, cc.p_mg
            ).m_as("kJ/mol")
            mu.append(mu_c)
            sigmas_fin.append(sigma_fin_c)
            sigmas_inf.append(sigma_inf_c)
        else:
            raise ValueError(
                f"cannot find compound for metabolite {m.id}"
                f" with external id {external_id}."
                "\nConsider setting the field metabolite_inchi_key"
                " if you haven't already."
            )
    sigmas_fin = np.array(sigmas_fin)
    sigmas_inf = np.array(sigmas_inf)
    cov = sigmas_fin @ sigmas_fin.T + 1e6 * sigmas_inf @ sigmas_inf.T
    cov = (
        pd.DataFrame(cov, index=met_order, columns=met_order)
        .loc[met_ix, met_ix]
        .round(10)
    )
    mu = pd.Series(mu, index=met_order, name="prior_mean_dgf").loc[met_ix].round(10)
    return mu, cov

