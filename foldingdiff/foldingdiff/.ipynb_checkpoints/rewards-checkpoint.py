### def rewards.py:


def compute_reward_for_backbone(backbone, args):
    # pass through ProteinMPNN call
    # get sequence, fold structure w/ OmegaFold
    # compute TMalign score between old backbone and new backbone
    # return 