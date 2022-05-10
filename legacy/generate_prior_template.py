
def generate_prior_template(data_path):
    """Generate template for prior definitions.

    :params data_path: a path to a maud input folder with a kinetic model
    and optionally experimental input file.
    """
    config = parse_config(toml.load(os.path.join(data_path, "config.toml")))
    kinetic_model_path = os.path.join(data_path, config.kinetic_model_file)
    kinetic_model = parse_toml_kinetic_model(toml.load(kinetic_model_path))
    measurements_path = os.path.join(data_path, config.measurements_file)
    biological_config_path = os.path.join(data_path, config.biological_config_file)
    all_experiments = get_all_experiment_object(toml.load(biological_config_path))
    raw_measurements = pd.read_csv(measurements_path)
    output_name = "prior_template.csv"
    output_path = os.path.join(data_path, output_name)
    print("Creating template")
    prior_dataframe = get_prior_template(
        km=kinetic_model,
        raw_measurements=raw_measurements,
        experiments=all_experiments,
        mode="sample",
    )
    print(f"Saving template to: {output_path}")
    prior_dataframe.to_csv(output_path)
    return "Successfully generated prior template"

