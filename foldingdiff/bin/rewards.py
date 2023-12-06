from sample import write_preds_pdb_folder


def compute_scTM_scores(final_sampled, args):
    pdbs_written = write_preds_pdb_folder(final_sampled, args)

    # pass through ProteinMPNN call
    # get sequence, fold structure w/ OmegaFold
    # compute TMalign score between old backbone and new backbone
    # return 
    return rewards 