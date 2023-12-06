import torch 

def reinforce(batch):
    """REINFORCE algorithm"""
    log_probs = batch["log_probs"] # Each element is (num_timesteps, num_residues, 6)
    rewards = batch["rewards"]
    policy_loss = []
    for log_prob, reward in zip(log_probs, rewards):
        overall_log_prob = log_prob.sum() # sum over all!! 
        # TODO: across a certain dimension, right?
        # Not sure about the separate angles... 
        policy_loss.append(-overall_log_prob * reward)
    # TODO: what is torch.cat's purpose? 
    return policy_loss


def vanilla_pg(batch):
    entropy_beta = 0.01 # per https://arxiv.org/pdf/1704.06440.pdf except not really because Copilot might be hallucinating
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
