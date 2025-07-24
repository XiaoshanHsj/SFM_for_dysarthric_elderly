"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math, time


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
#    target = target.unsqueeze(1).expand_as(sigma)
#    #print(target.shape)
#    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
#    return torch.prod(ret, 2)
    target = target.unsqueeze(1).expand_as(sigma)
    result = (target - mu) * torch.reciprocal(sigma)
    result = - 0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * ONEOVERSQRT2PI


#def mdn_loss(pi, sigma, mu, target):
#    """Calculates the error, given the MoG parameters and the target
#
#    The loss is the negative log likelihood of the data given the MoG
#    parameters.
#    """
##    prob = pi * gaussian_probability(sigma, mu, target)
##    nll = -torch.log(torch.sum(prob, dim=1))
##    return torch.mean(nll)
#    epsilon = 1e-6
#    result = gaussian_probability(sigma, mu, target) * pi
#    result = torch.sum(result, dim = 1)
#    result = - torch.log(epsilon + result)
#    return torch.mean(result)

def mdn_loss(pi, sigma, mu, target):
    epsilon = 1e-6
    sigma = sigma.squeeze()
    mu = mu.squeeze()
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    loss = torch.exp(m.log_prob(target))
    loss = torch.sum(loss * pi, dim = 1)
    loss = -torch.log(epsilon + loss)
    mean_loss = torch.mean(loss)
    return mean_loss

def mdn_loss_iter(pi, sigma, mu, target):
    sigma = sigma.squeeze()
    mu = mu.squeeze()
    if target.size(0) % 1024 != 0:
        total_iter = target.size(0) // 1024 + 1
    else:
        total_iter = target.size(0) // 1024
    epsilon = 1e-6
    for iter in range(total_iter):
        iter_pi = pi[iter * 1024 : (iter + 1) * 1024, :].cuda()
        iter_sigma = sigma[iter * 1024 : (iter + 1) * 1024, :].cuda()
        iter_mu = mu[iter * 1024 : (iter + 1) * 1024, :].cuda()
        iter_target = target[iter * 1024 : (iter + 1) * 1024, :].cuda()
        iter_m = torch.distributions.Normal(loc = iter_mu, scale = iter_sigma)
        iter_loss = torch.exp(iter_m.log_prob(iter_target))
        iter_loss = torch.sum(iter_loss * iter_pi, dim = 1)
        iter_loss = -torch.log(epsilon + iter_loss)
        if iter == 0:
            totol_loss = iter_loss
        else:
            totol_loss = torch.cat((totol_loss, iter_loss))
        print(totol_loss.size())
    mean_loss = torch.mean(totol_loss)
    return mean_loss

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
