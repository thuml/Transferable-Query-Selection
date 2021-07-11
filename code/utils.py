import torch
import torch.nn as nn
import numpy as np


def single_entropy(fc2_s):
    fc2_s = nn.Softmax(-1)(fc2_s)
    entropy = torch.sum(- fc2_s * torch.log(fc2_s + 1e-10), dim=1)
    entropy_norm = np.log(fc2_s.size(1))
    entropy = entropy / entropy_norm
    return entropy


def margin(out):
    out = nn.Softmax(-1)(out)
    top2 = torch.topk(out, 2).values
    # print(top2)
    return 1 - (top2[:, 0] - top2[:, 1])


def get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, domain_temperature=1.0, class_temperature=10.0):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    entropy = torch.sum(- fc2_s * torch.log(fc2_s + 1e-10), dim=1)
    entropy2 = torch.sum(- fc2_s2 * torch.log(fc2_s2 + 1e-10), dim=1)
    entropy3 = torch.sum(- fc2_s3 * torch.log(fc2_s3 + 1e-10), dim=1)
    entropy4 = torch.sum(- fc2_s4 * torch.log(fc2_s4 + 1e-10), dim=1)
    entropy5 = torch.sum(- fc2_s5 * torch.log(fc2_s5 + 1e-10), dim=1)
    entropy_norm = np.log(fc2_s.size(1))

    weight = (entropy + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return weight


def get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency


def get_predict_prob(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    predict_prob = torch.mean(c, 1)
    predict_prob = nn.Softmax(-1)(predict_prob)
    return predict_prob


def get_target_weight(entropy, consistency, threshold):
    sorce = (entropy + consistency) / 2
    weight = [0.0 for i in range(len(sorce))]
    for i in range(len(sorce)):
        if sorce[i] < (threshold / 2):
            weight[i] = 1.0
    return torch.tensor(weight)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()


def nega_normalize_weight(x):
    x = 1 - x
    return x.detach()
