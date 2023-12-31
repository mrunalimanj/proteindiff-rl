## copied over from bin/sample and bin/train, for dummy dataset construction

import json
from pathlib import Path
from typing import *
import os
import json
import logging
import functools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import glob
import argparse
import subprocess
import tempfile
from typing import *
import numpy as np
from tqdm.auto import tqdm
import re
import math

from tqdm.auto import tqdm
from torch import nn

from foldingdiff import datasets as dsets
from foldingdiff import beta_schedules, utils, tmalign

from huggingface_hub import snapshot_download

# Import data loading code from main training script
from foldingdiff import datasets
from foldingdiff import beta_schedules
from foldingdiff.angles_and_coords import create_new_chain_nerf

from foldingdiff.datasets import AnglesEmptyDataset, NoisedAnglesDataset


import os
import logging
import argparse
import subprocess
import shutil
import multiprocessing as mp
from typing import *

import torch
import numpy as np
from biotite.sequence import ProteinSequence, AlphabetError


import argparse
import functools
import os
import logging
from pathlib import Path
import multiprocessing as mp
import json
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

from biotite import structure as struc
from biotite.structure.io.pdb import PDBFile


from foldingdiff import tmalign
from foldingdiff.angles_and_coords import get_pdb_length


assert torch.cuda.is_available(), "Requires CUDA to train"
# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Define some typing literals
ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords"
]

PROTEINMPNN_SCRIPT = os.path.expanduser("~/project/ProteinMPNN/protein_mpnn_run.py") # TODO: update to be new one. 
assert os.path.isfile(PROTEINMPNN_SCRIPT), f"Expected {PROTEINMPNN_SCRIPT} to exist"


# :)
SEED = int(
    float.fromhex("54616977616e20697320616e20696e646570656e64656e7420636f756e747279")
    % 10000
)

FT_NAME_MAP = {
    "phi": r"$\phi$",
    "psi": r"$\psi$",
    "omega": r"$\omega$",
    "tau": r"$\theta_1$",
    "CA:C:1N": r"$\theta_2$",
    "C:1N:1CA": r"$\theta_3$",
}



class TrajectoryDataset(IterableDataset):
    """Basic experience source dataset.

    Takes a generate_batch function that returns an iterator. When given a generate_batch function, will produce trajectories on the fly. 
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator

        
    

def get_train_valid_test_sets(
    dataset_key: str = "cath",
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 512,
    min_seq_len: int = 0,
    seq_trim_strategy: datasets.TRIM_STRATEGIES = "leftalign",
    timesteps: int = 250,
    variance_schedule: beta_schedules.SCHEDULES = "linear",
    var_scale: float = np.pi,
    toy: Union[int, bool] = False,
    exhaustive_t: bool = False,
    syn_noiser: str = "",
    single_angle_debug: int = -1,  # Noise and return a single angle. -1 to disable, 1-3 for omega/theta/phi
    single_time_debug: bool = False,  # Noise and return a single time
    train_only: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the dataset objects to use for train/valid/test

    Note, these need to be wrapped in data loaders later
    """
    assert (
        single_angle_debug != 0
    ), f"Invalid value for single_angle_debug: {single_angle_debug}"

    clean_dset_class = {
        "canonical": datasets.CathCanonicalAnglesDataset,
        "canonical-full-angles": datasets.CathCanonicalAnglesOnlyDataset,
        "canonical-minimal-angles": datasets.CathCanonicalMinimalAnglesDataset,
        "cart-coords": datasets.CathCanonicalCoordsDataset,
    }[angles_definitions]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    splits = ["train"] if train_only else ["train", "validation", "test"]
    logging.info(f"Creating data splits: {splits}")
    clean_dsets = [
        clean_dset_class(
            pdbs=dataset_key,
            split=s,
            pad=max_seq_len,
            min_length=min_seq_len,
            trim_strategy=seq_trim_strategy,
            zero_center=False if angles_definitions == "cart-coords" else True,
            toy=toy,
        )
        for s in splits
    ]
    assert len(clean_dsets) == len(splits)
    # Set the training set mean to the validation set mean
    if len(clean_dsets) > 1 and clean_dsets[0].means is not None:
        logging.info(f"Updating valid/test mean offset to {clean_dsets[0].means}")
        for i in range(1, len(clean_dsets)):
            clean_dsets[i].means = clean_dsets[0].means

    if syn_noiser != "":
        if syn_noiser == "halfhalf":
            logging.warning("Using synthetic half-half noiser")
            dset_noiser_class = datasets.SynNoisedByPositionDataset
        else:
            raise ValueError(f"Unknown synthetic noiser {syn_noiser}")
    else:
        if single_angle_debug > 0:
            logging.warning("Using single angle noise!")
            dset_noiser_class = functools.partial(
                datasets.SingleNoisedAngleDataset, ft_idx=single_angle_debug
            )
        elif single_time_debug:
            logging.warning("Using single angle and single time noise!")
            dset_noiser_class = datasets.SingleNoisedAngleAndTimeDataset
        else:
            dset_noiser_class = datasets.NoisedAnglesDataset

    logging.info(f"Using {dset_noiser_class} for noise")
    noised_dsets = [
        dset_noiser_class(
            dset=ds,
            dset_key="coords" if angles_definitions == "cart-coords" else "angles",
            timesteps=timesteps,
            exhaustive_t=(i != 0) and exhaustive_t,
            beta_schedule=variance_schedule,
            nonangular_variance=1.0,
            angular_variance=var_scale,
        )
        for i, ds in enumerate(clean_dsets)
    ]
    for dsname, ds in zip(splits, noised_dsets):
        logging.info(f"{dsname}: {ds}")

    # Pad with None values
    if len(noised_dsets) < 3:
        noised_dsets = noised_dsets + [None] * int(3 - len(noised_dsets))
    assert len(noised_dsets) == 3

    return tuple(noised_dsets)



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
    
# <------------------------- Sampling methods ------------------------------------->

# Needs to sample according probabilities as well. 
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attention_mask=attn_mask)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        x_0 = model_mean
        # posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        # mean along all but batch dimension # TODO: what is the batch dimension here?
        # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim))
        return model_mean, torch.zeros_like(model_mean)
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # could just get probability from here lol.  
        x_t_minus_1 = model_mean + torch.sqrt(posterior_variance_t) * noise
        # print out my prob!
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((x_t_minus_1.detach() - model_mean) ** 2) / (2 * (posterior_variance_t))
            - torch.log(torch.sqrt(posterior_variance_t))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        # mean along all but batch dimension # TODO: what is the batch dimension here?
        # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        # print(f"log probs all neg at time step {t_index}")
        return x_t_minus_1, log_prob

def get_probs_from_samples(
    model: nn.Module,
    x: torch.Tensor,
    x_prev: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    # new model mean 
    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attention_mask=attn_mask)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        x_0 = model_mean,
        # posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        # mean along all but batch dimension # TODO: what is the batch dimension here?
        # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim))
        return torch.zeros_like(model_mean)
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        # print out my prob!
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((x_prev.detach() - model_mean) ** 2) / (2 * (posterior_variance_t))
            - torch.log(torch.sqrt(posterior_variance_t))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        # mean along all but batch dimension # TODO: what is the batch dimension here?
        # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        # print(f"log probs all neg at time step {t_index}")
        return log_prob


def p_probs_loop(
    model: nn.Module,
    lengths: Sequence[int],
    sample: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    
    probs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        img, prob_t = get_probs_from_samples(
            model=model,
            x=sample[:, :, i, :],
            x_prev=sample[:, :, i - 1, :],
            t=torch.full((sample[:, :, i, :],), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )
        
        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
            # consider wrapping probability? 
            # wrapped pdf is not tractable... oops. 
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        probs.append(prob_t.cpu())
    return torch.stack(probs)


def p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []
    probs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        img, prob_t = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )
        
        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
            # consider wrapping probability? 
            # wrapped pdf is not tractable... oops. 
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
        probs.append(prob_t.cpu())
    return torch.stack(imgs), torch.stack(probs)


def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 512,
    feature_key: str = "angles",
    disable_pbar: bool = False,
    trim_to_length: bool = True,  # Trim padding regions to reduce memory
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        if not sweep_min < sweep_max:
            raise ValueError(
                f"Minimum length {sweep_min} must be less than maximum {sweep_max}"
            )
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    prob_retval = [] 
    for this_lengths in lengths_chunkified:
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        # This is sampling noise, so here is where we can get out the probability of that original noise! 
        noise = train_dset.sample_noise(
            torch.zeros((batch, train_dset.pad, model.n_inputs), dtype=torch.float32)
        )

        # Trim things that are beyond the length of what we are generating
        if trim_to_length:
            noise = noise[:, : max(this_lengths), :]
            
        std_log_prob = - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))) - 0.5 * noise ** 2
        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled, probs = p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular[feature_key],
            disable_pbar=disable_pbar,
        )
        # add x_T probs to front, drop the 0 probs from last step 
        probs = torch.cat([std_log_prob[None, :, :, :], probs])[:1000, :, :, :]
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
        probs_trimmed_sampled = [
            probs[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        prob_retval.extend(probs_trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval, prob_retval


# <------------------------- reward computation functions; modified from bin/ scripts ------------------------------------->


class RewardStructure():
    def __init__(self, config):
        self.config = config
        for subpath in ["gen_pdb_outdir", "mpnn_outdir", "omegafold_outdir", "sctm_score_dir"]:
            self.config[subpath] = os.path.join(self.config["new_results_dir"], subpath)
            os.makedirs(self.config[subpath], exist_ok=True)

    # <------------------------- R_0: Generate PDBs from angle sequences ------------------------------------->

    def write_preds_pdb_folder(
        self, 
        final_sampled: Sequence[pd.DataFrame],
        device, 
        feature_names = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"],
        basename_prefix: str = "generated_",
        threads: int = mp.cpu_count(),
    ) -> List[str]:
        """
        Write the predictions as pdb files in the given folder along with information regarding the
        tm_score for each prediction. Returns the list of files written.
        """
        gen_pdb_outdir = self.config["gen_pdb_outdir"]
        os.makedirs(gen_pdb_outdir, exist_ok=True)
        logging.info(
            f"Writing sampled angles as PDB files to {gen_pdb_outdir} using {threads} threads"
        )
        # Create the pairs of arguments
        sampled_dfs = [
        pd.DataFrame(s, columns=feature_names)
        for s in final_sampled]
        seqlen = final_sampled[0].shape[0]
        arg_tuples = [
            (os.path.join(gen_pdb_outdir, f"{basename_prefix}{i}_len_{seqlen}_dev_{device}.pdb"), samp)
            for i, samp in enumerate(sampled_dfs)
        ]
        # Write in parallel
        with mp.Pool(threads) as pool:
            files_written = pool.starmap(create_new_chain_nerf, arg_tuples)

        return files_written

    # <------------------------- R_1: Sequence Design w/ ProteinMPNN ------------------------------------->


    def write_fasta_mpnn(self, fname: str, seq: str, seqname: str = "sampled") -> str:
        """Write a fasta file"""
        assert fname.endswith(".fasta")
        with open(fname, "w") as f:
            f.write(f">{seqname}\n")
            for chunk in [seq[i : i + 80] for i in range(0, len(seq), 80)]:
                f.write(chunk + "\n")
        return fname


    def read_fasta_mpnn(self, fname: str) -> Dict[str, str]:
        """
        Read the given fasta file and return a dictionary of its sequences
        """
        seq_dict = {}
        curr_key, curr_seq = "", ""
        with open(fname, "r") as source:
            for line in source:
                if line.startswith(">"):
                    if curr_key:
                        assert curr_seq
                        seq_dict[curr_key] = curr_seq
                    curr_key = line.strip().strip(">")
                    curr_seq = ""
                else:
                    curr_seq += line.strip()

            assert curr_key and curr_seq
            seq_dict[curr_key] = curr_seq
        return seq_dict


    def update_fname(self, fname: str, i: int, device,  new_dir: str = "",) -> str:
        """
        Update the pdb filename to include a numeric index and a .fasta extension.
        If new_dir is given then we move the output filename to that directory.
        """
        assert os.path.isfile(fname)
        parent, child = os.path.split(fname)
        assert child
        child_base, _child_ext = os.path.splitext(child)
        assert child_base
        if new_dir:
            assert os.path.isdir(new_dir), f"Expected {new_dir} to be a directory"
            parent = new_dir
        return os.path.join(parent, f"{child_base}_dev_{device}_proteinmpnn_residues_{i}.fasta")


    def generate_residues_proteinmpnn(
        self, 
        pdb_fname: str, n_sequences: int = 8, temperature: float = 0.1 # TODO: are these defaults the same as the argparser? # Number of sequences will have to be averaged over in reward computation. 
    ) -> List[str]:
        """
        Generates residues for the given pdb_filename using ProteinMPNN

        Trippe et al uses a temperature of 0.1 to sample 8 amino acid sequences per structure
        """
        bname = os.path.basename(pdb_fname).replace(".pdb", ".fa")
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir) # TODO: they are using a subprocess call. I think that's ok because PROTEINMPNN uses CPUs anyway.
            cmd = f'python {PROTEINMPNN_SCRIPT} --pdb_path_chains A --out_folder {tempdir} --num_seq_per_target {n_sequences} --seed 1234 --batch_size {n_sequences} --pdb_path {pdb_fname} --sampling_temp "{temperature}" --ca_only'
            retval = subprocess.call(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            assert retval == 0, f"Command {cmd} failed with return value {retval}"
            outfile = tempdir / "seqs" / bname
            assert os.path.isfile(outfile)

            # Read the fasta file, return the sequences that were generated
            seqs = self.read_fasta_mpnn(outfile)
            seqs = {k: v for k, v in seqs.items() if k.startswith("T=")}
        assert len(seqs) == n_sequences
        return list(seqs.values())


    def pdbs_to_seqs(self, pdb_fnames: List[str], device) -> List[str]:
        """
        For each pdb file in the given list, generate sequences using ProteinMPNN and write them to a fasta file.
        Returns a list of the fasta files written.
        """
        # print(f"Running ProteinMPNN on {len(pdb_fnames)} PDB files")
        proteinmpnn_seqs_written = [] # TODO: wait this should be double indexed. oops. 
        for pdb_fname in tqdm(pdb_fnames):
            seqs = self.generate_residues_proteinmpnn(
                pdb_fname, n_sequences=self.config["mpnn_replicates"], temperature=0.8
            )
            # file_names_pdb_i = []
            for i, seq in enumerate(seqs):
                out_fname = self.update_fname(pdb_fname, i, device=device, new_dir=self.config["mpnn_outdir"])
                self.write_fasta_mpnn(
                    out_fname, seq, seqname=os.path.splitext(os.path.basename(out_fname))[0] # TODO: make sure this is right. 
                )
                # file_names_pdb_i.append(out_fname)
                proteinmpnn_seqs_written.append(out_fname)

        # print(proteinmpnn_seqs_written)

        return proteinmpnn_seqs_written


    # <------------------------- R_2: Structure Design w/ OmegaFold ------------------------------------->

    """
    Short script to parallelize omegafold across GPUs to speed up runtime.
    https://github.com/HeliXonProtein/OmegaFold
    """

    def read_fasta_omega(self, fname: str, check_valid: bool = True) -> Dict[str, str]:
        """Read the sequences in the fasta to a dict"""

        def add_seq_if_valid(d: Dict[str, str], k: str, v: str) -> None:
            """Add v to d[k] if v is a valid sequence"""
            if not check_valid:
                d[k] = v
                return
            try:
                _ = ProteinSequence(v)
                d[k] = v
            except AlphabetError:
                logging.warning(f"Illegal character in entry {k}: {v} | skipping")

        retval = {}
        curr_k, curr_v = "", ""
        with open(fname) as source:
            for line in source:
                if line.startswith(">"):
                    if curr_k:  # Record and reset
                        assert curr_v
                        assert curr_k not in retval, f"Duplicated fasta entry: {curr_k}"
                        add_seq_if_valid(retval, curr_k, curr_v)
                    curr_k = line.strip().strip(">")
                    curr_v = ""
                else:
                    curr_v += line.strip()
        # Write the last sequence
        assert curr_k
        assert curr_v
        add_seq_if_valid(retval, curr_k, curr_v)
        return retval


    def write_fasta_omega(self, sequences: Dict[str, str], out_fname: str):
        """Write the sequeces to the given fasta file"""
        with open(out_fname, "w") as sink:
            for k, v in sequences.items():
                sink.write(f">{k}\n")
                for segment in [v[i : i + 80] for i in range(0, len(v), 80)]:
                    sink.write(segment + "\n")


    def run_omegafold(self, input_fasta: str, outdir: str, gpu: int, weights: str = ""):
        """
        Runs omegafold on the given fasta file
        """
        logging.info(
            f"Running omegafold on {input_fasta} > {outdir} with gpu {gpu} with weights {weights}"
        )
        assert shutil.which("omegafold")
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} omegafold {input_fasta} {outdir} --device cuda:0"
        if weights:
            assert os.path.isfile(weights)
            cmd += f" --weights_file {weights}"

        bname = os.path.splitext(os.path.basename(input_fasta))[0]
        with open(
            os.path.join(outdir, f"omegafold_{bname}_gpu_reassign_{gpu}.stdout"), "wb"
        ) as sink:
            output = subprocess.call(cmd, shell=True, stdout=sink)


    def seqs_to_structures(self, list_of_fasta_files: List[str], device):
        gpus = self.config["omegafold_gpus"]
        outdir = self.config["omegafold_outdir"]
        # outdir=os.path.abspath(os.path.join(os.getcwd(), self.config["omegafold_outdir"])),
        # TODO: could use output directory, figure out logic for saving + storing
        input_sequences = {}
        for fname in list_of_fasta_files:
            fname_seqs = self.read_fasta_omega(fname)
            assert fname_seqs.keys().isdisjoint(input_sequences.keys())
            input_sequences.update(fname_seqs)
        n = len(input_sequences)
        logging.info(f"Parsed {n} sequences")

        # Divide up the inputs, shuffling their indices su that the load is spread
        # across GPUs; otherwise, if we just give them in order, the first GPU will
        # finish first since it has shorter sequences.
        idx = np.arange(n)
        rng = np.random.default_rng(seed=1234)
        rng.shuffle(idx)
        idx_split = np.array_split(idx, len(gpus))
        all_keys = list(input_sequences.keys())
        all_keys_split = [[all_keys[i] for i in part] for part in idx_split]
        # Write the tempfiles and call omegafold

        processes = []
        for i, key_chunk in enumerate(all_keys_split):
            fasta_filename = os.path.join(outdir, f"{i}_dev_{device}_omegafold_input.fasta")
            # assert not os.path.exists(fasta_filename)
            logging.info(f"Writing {len(key_chunk)} sequences to {fasta_filename}")
            self.write_fasta_omega({k: input_sequences[k] for k in key_chunk}, fasta_filename)
            proc = mp.Process(
                target=self.run_omegafold,
                args=(fasta_filename, outdir, gpus[i], ""),
            )
            processes.append(proc)
            proc.start()
        for p in processes:
            p.join()


    # <------------------------- R_3: Alignment scoring between new / old structures ------------------------------------->

    """
    Script for calculating self consistency TM scores
    """

    def get_sctm_score(self, orig_pdb: Path, folded_dirname: Path) -> Tuple[float, str]:
        """get the self-consistency tm score"""
        bname = os.path.splitext(os.path.basename(orig_pdb))[0] + "_*_residues_*.pdb"
        folded_pdbs = glob.glob(os.path.join(folded_dirname, bname))
        assert len(folded_pdbs) <= 10  # We have never run more than 10 per before
        if len(folded_pdbs) > 8:
            folded_pdbs = folded_pdbs[:8]
        assert len(folded_pdbs) <= 8
        if len(folded_pdbs) < 8:
            logging.warning(
                f"Fewer than 8 (n={len(folded_pdbs)}) structures corresponding to {orig_pdb}"
            )
        if not folded_pdbs:
            return np.nan, ""
        return tmalign.max_tm_across_refs(orig_pdb, folded_pdbs, parallel=False)

    def get_all_sctm_scores(self, orig_pdb: Path, folded_dirname: Path) -> Tuple[float, str]:
        """get the self-consistency tm score"""
        bname = os.path.splitext(os.path.basename(orig_pdb))[0] + "_*_residues_*.pdb"
        folded_pdbs = glob.glob(os.path.join(folded_dirname, bname))
        assert len(folded_pdbs) <= 10  # We have never run more than 10 per before
        if len(folded_pdbs) > 8:
            folded_pdbs = folded_pdbs[:8]
        assert len(folded_pdbs) <= 8
        if len(folded_pdbs) < 8:
            logging.warning(
                f"Fewer than 8 (n={len(folded_pdbs)}) structures corresponding to {orig_pdb}"
            )
        if not folded_pdbs:
            return np.nan, ""
        return tmalign.all_tm_across_refs(orig_pdb, folded_pdbs, parallel=False)


    def score_structures(self, folded, predicted, step):
        """
        folded is the structures from the inverse folding test
        predicted is the generated backbone structures, from the sampler"""
        # assert os.path.isdir(folded)
        assert os.path.isdir(predicted), f"Directory not found: {predicted}"
        # TODO: make sure this is right
        # perhaps need to wait until all files are done computing. 
        orig_predicted_backbones = glob.glob(os.path.join(predicted, "*.pdb"))
        logging.info(
            f"Computing scTM scores across {len(orig_predicted_backbones)} generated structures"
        )
        orig_predicted_backbone_lens = {
            os.path.splitext(os.path.basename(f))[0]: get_pdb_length(f)
            for f in orig_predicted_backbones
        }
        orig_predicted_backbone_names = [
            os.path.splitext(os.path.basename(f))[0] for f in orig_predicted_backbones
        ]

        # Match up the files
        pfunc = functools.partial(self.get_all_sctm_scores, folded_dirname=Path(folded))
        pool = mp.Pool(mp.cpu_count())
        sctm_scores_raw_and_ref = list(
            pool.map(pfunc, orig_predicted_backbones, chunksize=5)
        )
        pool.close()
        pool.join()
        print(sctm_scores_raw_and_ref)
        sctm_non_nan_idx = [
            i for i, val in enumerate(sctm_scores_raw_and_ref) if ~(np.isnan(val)).all()
        ]
        full_sctm_scores_mapping = {
            orig_predicted_backbone_names[i]: sctm_scores_raw_and_ref[i]
            for i in sctm_non_nan_idx
        }
        # sctm_scores_reference = {
        #    orig_predicted_backbone_names[i]: sctm_scores_raw_and_ref[i][1]
        #    for i in sctm_non_nan_idx
        # }
        sctm_file = os.path.join(self.config["sctm_score_dir"], f"{self.config['sctm_score_file']}_step_{step}.csv")
        scores_all_df = pd.DataFrame.from_dict(full_sctm_scores_mapping, orient = 'index').reset_index()
        scores_all_df.to_csv(sctm_file, index = False)
        
        
        sctm_scores_mapping = {backbone: np.nanmax(scores) for backbone, scores in full_sctm_scores_mapping.items()}
        sctm_scores = np.array(list(sctm_scores_mapping.values()))

        passing_num = np.sum(sctm_scores >= 0.5)
        logging.info(
            f"{len(sctm_scores)} entries with scores, {passing_num} passing 0.5 cutoff"
        )

        # Write the output
        logging.info(
            f"scTM score mean/median: {np.mean(sctm_scores), np.median(sctm_scores)}"
        )
        
        # Need to return one score for each structure!
        grouped_sctm_scores = {}
        dev_values = set([int(re.findall(r'dev_(\d+)', file_name)[0]) for file_name in sctm_scores_mapping.keys()])
        for key, val in sctm_scores_mapping.items():
            dev_num = int(re.findall(r'dev_(\d+)', key)[0])
            if dev_num not in grouped_sctm_scores:
                grouped_sctm_scores[dev_num] = []
            grouped_sctm_scores[dev_num].append(val)
    
        flattened_sctms = np.array(list(grouped_sctm_scores.values())).mean(axis=0)

        return flattened_sctms

    # <------------------------- R_total: End-to-end computation ------------------------------------->

    def compute_scTM_scores(self, final_sampled, feature_names, device):
        self.config["abs_step"] += 1
        # clear temp before proceeding
        for subpath in ["gen_pdb_outdir", "mpnn_outdir", "omegafold_outdir"]:
            files = glob.glob(os.path.join(self.config[subpath], "*"))
            for f in files:
                os.remove(f)

        pdbs_written = self.write_preds_pdb_folder(final_sampled, feature_names = feature_names, device=device)
        mpnns_written = self.pdbs_to_seqs(pdbs_written, device=device)
        # above is debugged 
        self.seqs_to_structures(mpnns_written,device=device)
        print("rewards successfully computed for this epoch?")
        orig_pdb_folder = self.config["gen_pdb_outdir"]
        new_pdbs_folder = self.config["omegafold_outdir"]
        rewards = self.score_structures(predicted = orig_pdb_folder, folded = new_pdbs_folder,
                                        step = self.config["abs_step"])
        assert len(final_sampled) == len(rewards)
        return rewards 