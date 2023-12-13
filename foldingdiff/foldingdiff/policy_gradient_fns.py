import torch 

def reinforce(batch):
    """REINFORCE algorithm"""
    if len(batch) != 3:
        batch_list = batch
        policy_loss = []
        for batch in batch_list:
            samples, rewards, log_probs = batch
            policy_loss = []
            # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
            # print("log_probs", log_probs) # these log_probabilities aren't gettign computed correctly :(
            overall_log_probs = torch.stack([log_prob.nansum(dim=tuple(range(log_prob.ndim - 1))) for log_prob in log_probs]).requires_grad_()
            loss = -overall_log_probs * rewards[:, None]
            policy_loss.append(loss.transpose(0, 1))
            
        return policy_loss # TODO: hopefully this remains 2-dimensional?
    else: 
        samples, rewards, log_probs = batch
        policy_loss = []
        # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
        overall_log_probs = torch.stack([log_prob.nansum(dim=tuple(range(log_prob.ndim - 1))) for log_prob in log_probs]).requires_grad_()
        loss = -overall_log_probs * rewards[:, None]
        return loss.transpose(0, 1)

def reinforce_pos(batch):
    """REINFORCE algorithm"""
    if len(batch) != 3:
        batch_list = batch
        policy_loss = []
        for batch in batch_list:
            samples, rewards, log_probs = batch
            policy_loss = []
            # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
            # print("log_probs", log_probs) # these log_probabilities aren't gettign computed correctly :(
            # flatten any positive values to 0
        
            # log_probs = torch.where(log_probs > 0, torch.zeros_like(log_probs), log_probs)
            overall_log_probs = torch.stack([torch.where(log_prob > 0, torch.zeros_like(log_prob), log_prob).nansum(dim=tuple(range(log_prob.ndim - 1))) for log_prob in log_probs]).requires_grad_()
            loss = -overall_log_probs * rewards[:, None]
            policy_loss.append(loss.transpose(0, 1))
            
        return policy_loss # TODO: hopefully this remains 2-dimensional?
    else: 
        samples, rewards, log_probs = batch
        policy_loss = []
        
        overall_log_probs = torch.stack([torch.where(log_prob > 0, torch.zeros_like(log_prob), log_prob).nansum(dim=tuple(range(log_prob.ndim - 1))) for log_prob in log_probs]).requires_grad_()
        loss = -overall_log_probs * rewards[:, None]
        return loss.transpose(0, 1)


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


def ppo(batch):
    """Proximal Policy Optimization algorithm"""
    if len(batch) != 3:
        batch_list = batch
        policy_loss = []
        for batch in batch_list:
            samples, advs, new_log_probs, importances = batch
            # Implement proximal policy optimization

            # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
            # print("log_probs", log_probs) # these log_probabilities aren't gettign computed correctly :(
            surrogate1 = importances * advs
            surrogate2 = torch.clamp(importances, 1.0 - 0.2, 1.0 + 0.2) * advs
            policy_loss.append(-torch.min(surrogate1, surrogate2).mean().transpose(0, 1))
            
        return policy_loss # TODO: hopefully this remains 2-dimensional?
    else: 
        samples, advs, new_log_probs, importances = batch
            # Implement proximal policy optimization

            # print("back log probs", log_probs[0][-1]) # this is the one to diagnose... 
            # print("log_probs", log_probs) # these log_probabilities aren't gettign computed correctly :(
            surrogate1 = importances * advs
            surrogate2 = torch.clamp(importances, 1.0 - 0.2, 1.0 + 0.2) * advs
            loss = -torch.min(surrogate1, surrogate2).mean()
        return loss.transpose(0, 1)