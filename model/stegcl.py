import torch
import torch.nn.functional as F

def select_positive_samples(mask_positive):
    positive_samples = []
    for i in range(mask_positive.size(0)):
        pos_indices = mask_positive[i].nonzero(as_tuple=False).view(-1)
        positive_sample = pos_indices[torch.randint(0, len(pos_indices), (1,))]
        positive_samples.append(positive_sample)
    return torch.cat(positive_samples)

def stegcl_loss(features, labels, tau=0.07):
    features = F.normalize(features, p=2, dim=1)
    similarity = torch.matmul(features, features.T) / tau

    n = similarity.size(0)

    labels = labels.view(-1, 1)
    mask_positive = torch.eq(labels, labels.T).float()
    mask_negative = 1 - mask_positive

    positive_samples_indices = select_positive_samples(mask_positive)

    loss_individual = -torch.log(
        torch.exp(similarity[range(n), positive_samples_indices]) /
        (torch.exp(similarity[range(n), positive_samples_indices]) + torch.sum(torch.exp(similarity) * mask_negative, dim=1))
    )

    loss = loss_individual.mean()
    return loss