"""Console script for maudtools."""
import os

import arviz as az
import click
import toml
from maud.loading_maud_inputs import load_maud_input
from maud.parsing_kinetic_models import parse_kinetic_model
from maud.parsing_measurements import parse_measurement_set
from maud.parsing_configs import parse_config
from maud.getting_stan_variables import get_stan_variable_set
from maud.utils import load_df

from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator
from maudtools.generating_inits import generate_inits
from maudtools.generating_priors import generate_prior_template


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


@cli.command("generate-inits")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option("--chain", default=0, help="Sampling chain using python indexing")
@click.option(
    "--draw",
    default=0,
    help="Sampling draw using python indexing from start of phase",
)
@click.option(
    "--warmup", default=0, help="0 if in sampling, 1 if in warmup phase"
)
def generate_inits_command(data_path, chain, draw, warmup):
    """Run the generate_inits function as a click command."""
    output_name = "generated_inits.csv"
    output_path = os.path.join(data_path, output_name)
    idata = az.InferenceData.from_netcdf(os.path.join(data_path, "idata.nc"))
    mi = load_maud_input(os.path.join(data_path, "user_input"))
    click.echo("Creating inits table")
    inits = generate_inits(idata, mi, chain, draw, warmup)
    click.echo(f"Saving inits table to: {output_path}")
    inits.to_csv(output_path)
    click.echo("Successfully generated inits csv")


@cli.command("generate-prior-template")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def generate_prior_template_command(data_path):
    """Run the generate_prior_template function as a click command."""
    output_name = "prior_template.csv"
    output_path = os.path.join(data_path, output_name)
    # get config
    config_path = os.path.join(data_path, "config.toml")
    config = parse_config(toml.load(config_path))
    kinetic_model_path = os.path.join(data_path, config.kinetic_model_file)
    bio_config_path = os.path.join(data_path, config.experimental_setup_file)
    raw_bio_config = toml.load(bio_config_path)
    measurements_path = os.path.join(data_path, config.measurements_file)
    raw_bio_config = toml.load(bio_config_path)
    measurements_path = os.path.join(data_path, config.measurements_file)
    raw_measurements = load_df(measurements_path)
    km_toml = toml.load(kinetic_model_path)
    km = parse_kinetic_model(km_toml)
    measurement_set = parse_measurement_set(raw_measurements, raw_bio_config)
    stan_variable_set = get_stan_variable_set(km, measurement_set)
    click.echo("Creating prior template")
    prior_template = generate_prior_template(stan_variable_set)
    click.echo(f"Saving prior template to: {output_path}")
    prior_template.to_csv(output_path, index=False)
    click.echo("Successfully generated prior template")
