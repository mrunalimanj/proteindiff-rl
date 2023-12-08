sampling_utils.py


def build_datasets(
    model_dir: Path, load_actual: bool = True
) -> Tuple[NoisedAnglesDataset, NoisedAnglesDataset, NoisedAnglesDataset]:
    """
    Build datasets given args again. If load_actual is given, the load the actual datasets
    containing actual values; otherwise, load a empty shell that provides the same API for
    faster generation.
    """
    with open(model_dir / "training_args.json") as source:
        training_args = json.load(source)
    # Build args based on training args
    if load_actual:
        dset_args = dict(
            timesteps=training_args["timesteps"],
            variance_schedule=training_args["variance_schedule"],
            max_seq_len=training_args["max_seq_len"],
            min_seq_len=training_args["min_seq_len"],
            var_scale=training_args["variance_scale"],
            syn_noiser=training_args["syn_noiser"],
            exhaustive_t=training_args["exhaustive_validation_t"],
            single_angle_debug=training_args["single_angle_debug"],
            single_time_debug=training_args["single_timestep_debug"],
            toy=training_args["subset"],
            angles_definitions=training_args["angles_definitions"],
            train_only=False,
        )

        train_dset, valid_dset, test_dset = get_train_valid_test_sets(**dset_args)
        logging.info(
            f"Training dset contains features: {train_dset.feature_names} - angular {train_dset.feature_is_angular}"
        )
        return train_dset, valid_dset, test_dset
    else:
        print('load_actual is None?')
        mean_file = model_dir / "training_mean_offset.npy"
        placeholder_dset = AnglesEmptyDataset(
            feature_set_key=training_args["angles_definitions"],
            pad=training_args["max_seq_len"],
            mean_offset=None if not mean_file.exists() else np.load(mean_file),
        )
        noised_dsets = [
            NoisedAnglesDataset(
                dset=placeholder_dset,
                dset_key="coords"
                if training_args["angles_definitions"] == "cart-coords"
                else "angles",
                timesteps=training_args["timesteps"],
                exhaustive_t=False,
                beta_schedule=training_args["variance_schedule"],
                nonangular_variance=1.0,
                angular_variance=training_args["variance_scale"],
            )
            for _ in range(3)
        ]
        return noised_dsets
