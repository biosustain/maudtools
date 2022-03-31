"""Console script for maudtools."""
import os
import click
from maud.io import load_maud_input

from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator

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
    "-p", "--print_results",
    is_flag=True,
    help="Print the mean and covariance matrix?",
    default=False
)
def fetch_dgf_priors(maud_input_dir: str, print_results: bool):
    """Write csv files with dgf prior means and covariances from equilibrator.

    Make sure that the metabolites-in-compartments in the kinetic model file
    have values in the field "metabolite_external_id" that equilibrator can
    recognise - otherwise this script will raise an error.

    """
    file_mean = os.path.join(maud_input_dir, "dgf_prior_mean_equilibrator.csv")
    file_cov = os.path.join(maud_input_dir, "dgf_prior_cov_equilibrator.csv")
    mi = load_maud_input(maud_input_dir, "sample")
    mu, cov = fetch_dgf_priors_from_equilibrator(mi)
    if print_results:
        click.echo("Prior mean vector:")
        click.echo(mu)
        click.echo("Prior covariance:")
        click.echo(cov)
    mu.to_csv(file_mean)
    cov.to_csv(file_cov)
    click.echo(f"Wrote files {file_mean} and {file_cov}.")

