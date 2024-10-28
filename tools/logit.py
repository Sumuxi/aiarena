import numpy as np
import torch


def logits_from_probabilities1(probabilities):
    probabilities = np.array(probabilities)
    logits = np.log(probabilities)
    max_logit = np.max(logits)
    return logits - max_logit

# 示例
probabilities = [0.6, 0.3, 0.1]
logits = logits_from_probabilities1(probabilities)
print(logits)
print(torch.softmax(torch.tensor(logits), dim=0))

def logits_from_probabilities(probabilities):
    probabilities = torch.tensor(probabilities)
    log_probs = torch.log(torch.clamp(probabilities, min=1e-8))
    max_logit = torch.max(log_probs)
    logits = log_probs - max_logit
    return logits

logits = logits_from_probabilities(probabilities)
print(logits)
print(torch.softmax(torch.tensor(logits), dim=0))

logits = [torch.log(torch.clamp(prob, min=1e-8)) for prob in torch.tensor(probabilities)]
print(logits)
print(torch.softmax(torch.tensor(logits), dim=0))




