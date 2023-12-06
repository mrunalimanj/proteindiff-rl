import torch 

def reinforce(self, rewards, log_probs):
    """REINFORCE algorithm"""
    policy_loss = []
    for log_prob, reward in zip(log_probs, rewards):
        overall_log_prob = log_prob.sum() # TODO: across a certain dimension, right?
        policy_loss.append(-overall_log_prob * reward)
    # TODO: what is torch.cat's purpose? 
    policy_loss = torch.cat(policy_loss).sum()
    # Do I need these specifically computed here?
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
    return policy_loss