"""Console script for maudtools."""
import os
from typing import Optional

import click
from maud.loading_maud_inputs import load_maud_input
from maud.getting_idatas import get_idata

from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator
from maudtools.get


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Use the maudtools command line interface."""
    pass


@cli.command()
@click.argument(
    "maud_input_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "-p",
    "--print_results",
    is_flag=True,
    help="Print the mean and covariance matrix?",
    default=False,
)
@click.option(
    "-t",
    "--temperature",
    default="310.15K",
    type=str,
    help="A temperature quantity with units",
)
@click.option("-ph", "--pH", default=7.0, type=float, help="pH")
def fetch_dgf_priors(
    maud_input_dir: str, temperature: str, ph: float, print_results: bool
):
    """Write csv files with dgf prior means and covariances from equilibrator.

    Make sure that the metabolites-in-compartments in the kinetic model file
    have values in the field "metabolite_external_id" that equilibrator can
    recognise - otherwise this script will raise an error.

    """
    file_mean = os.path.join(maud_input_dir, "dgf_prior_mean_equilibrator.csv")
    file_cov = os.path.join(maud_input_dir, "dgf_prior_cov_equilibrator.csv")
    mi = load_maud_input(maud_input_dir)
    mu, cov = fetch_dgf_priors_from_equilibrator(mi, temperature, ph)
    if print_results:
        click.echo("Prior mean vector:")
        click.echo(mu)
        click.echo("Prior covariance:")
        click.echo(cov)
    mu.to_csv(file_mean)
    cov.to_csv(file_cov)
    click.echo(f"Wrote files {file_mean} and {file_cov}.")


@cli.command()
@click.argument(
    "maud_output_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--chain",
    default=0,
    type=int,
    help="MCMC chain number",
)
@click.option(
    "--draw",
    default=0,
    type=int,
    help="MCMC draw number",
)
@click.option("--warmup", is_flag=True, help="If draw is in warmup phase or not")
@click.option(
    "--experiment",
    default=None,
    help="Id of an experiment",
)
def generate_yaml(
    maud_output_dir: str, chain: int, draw: int, warmup: bool, experiment: Optional[str]
):
    sample_dir = os.path.join(maud_output_dir, "samples")
    maud_input_dir = os.path.join(maud_output_dir, "user_input")
    csvs = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith(".csv")
    ]
    mi = load_maud_input(maud_input_dir)
    idata = get_idata(csvs, mi, "train")
    if experiment is None:
        experiment = next(experiment.id for experiment in mi.experiments)
    parameter_values = get_inits_gg
    
