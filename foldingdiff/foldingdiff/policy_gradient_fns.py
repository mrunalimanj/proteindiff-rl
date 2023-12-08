import torch 

def reinforce(batch):
    """REINFORCE algorithm"""
    samples, rewards, log_probs = batch
    policy_loss = []
    # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
    overall_log_probs = torch.stack([log_prob.nansum(dim=tuple(range(log_prob.ndim - 1))) for log_prob in log_probs]).requires_grad_()
    loss = -overall_log_probs * rewards[:, None]
    return loss.transpose(0, 1) # TODO: hopefully this remains 2-dimensional?


def vanilla_pg(batch):
    entropy_beta = 0.01 # per https://arxiv.org/pdf/1704.06440.pdf except not really because Copilot might be hallucinating
    samples, rewards, log_probs = batch
    log_probs = batch["log_probs"]
    rewards = batch["rewards"]
    policy_loss = []
    for log_prob, reward in zip(log_probs, rewards):
        overall_log_prob = log_prob.sum() # Each element is num_timesteps x num_residues x 6
        # TODO: across a certain dimension, right?
        # Not sure about the separate angles... 
        policy_loss.append(-overall_log_prob * reward) 
    # entropy loss
    prob = torch.exp(log_prob)
    entropy = -(prob * log_prob).sum().mean()
    entropy_loss = -entropy_beta * entropy

    # total loss
    loss = policy_loss + entropy_loss

    return loss
