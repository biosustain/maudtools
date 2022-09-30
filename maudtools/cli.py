"""Console script for maudtools."""
import os
from typing import Optional

import arviz as az
import click
from maud.getting_idatas import get_idata  # type: ignore
from maud.loading_maud_inputs import load_maud_input  # type: ignore

from maudtools.fetching_dgf_priors import fetch_dgf_priors_from_equilibrator
from maudtools.generating_inits import generate_inits
from maudtools.generating_sbml import generate_sbml
import libsbml as sbml  # type: ignore


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


@cli.command("generate-sbml")
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
@click.option(
    "--warmup", is_flag=True, help="If draw is in warmup phase or not"
)
@click.option(
    "--experiment",
    default=None,
    help="Id of an experiment",
)
def generate_sbml_command(
    maud_output_dir: str,
    chain: int,
    draw: int,
    warmup: bool,
    experiment: Optional[str],
):
    """Get inputs for the generate_sbml function and run it."""
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
        experiment = next(
            experiment.id for experiment in mi.measurements.experiments
        )
    file_name = f"ch{chain}-dr{draw}-wu{warmup}-ex{experiment}.xml"
    file_out = os.path.join(maud_output_dir, file_name)
    sbml_doc, sbml_model = generate_sbml(idata, mi, experiment, chain, draw, warmup)
    with open(file_out, "w") as f:
        f.write(sbml.writeSBMLToString(sbml_doc))

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
    idata_file = os.path.join(data_path, "idata.nc")
    if not os.path.exists(idata_file):
        idata_file = os.path.join(data_path, f"idata-chain{chain+1}.nc")
    assert os.path.exists(idata_file), f"Directory {data_path} contains no idata file."
    idata = az.InferenceData.from_netcdf(idata_file)
    mi = load_maud_input(os.path.join(data_path, "user_input"))
    click.echo("Creating inits table")
    inits = generate_inits(idata, mi, chain, draw, warmup)
    click.echo(f"Saving inits table to: {output_path}")
    inits.to_csv(output_path)
    click.echo("Successfully generated inits csv")


@cli.command("rescue-idata")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option("--chain", default=1, help="Chain to use")
def rescue_idata(data_path, chain):
    """Generate an idata from a single chain after running Maud."""
    output_file = os.path.join(data_path, f"idata-chain{chain}.nc")
    input_dir = os.path.join(data_path, "user_input")
    csv_dir = os.path.join(data_path, "samples")
    end_pattern = f"{chain}.csv"
    csv_filename = next(f for f in os.listdir(csv_dir) if f.endswith(end_pattern))
    csv_file = os.path.join(csv_dir, csv_filename)
    assert os.path.exists(csv_file), f"No file in directory {csv_dir} ends with {end_pattern}"
    mi = load_maud_input(data_path=input_dir)
    idata = get_idata([csv_file], mi, "train")
    idata.to_netcdf(output_file)
