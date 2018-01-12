import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import visdom

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones
from pyro.optim import Adam


class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def get_data(batch_size=128, use_cuda=False):
	root = './data'
	download = True
	trans = transforms.ToTensor()
	train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
	test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
	kwargs = {'num_workers': 1, 'pin_memory': use_cuda}

	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, **kwargs)

	return train_loader, test_loader


def plot_vae_samples(vae, visdom_session):
	vis = visdom_session
	images = []
	for rr in range(100):
		sample_i, sample_mu_i = vae.model_sample()
		img = sample_mu_i[0].view(1, 28, 28).cpu().data.numpy()
		images.append(img)
	vis.images(images, 10, 2)

class Decoder(nn.Module):
	"""Variational Decoder"""

	def __init__(self, z_dim, hidden_dim):
		super(Decoder, self).__init__()
		self.fc1 = nn.Linear(z_dim, hidden_dim)
		self.fc21 = nn.Linear(hidden_dim, 784)
		# setup the non-linearity
		self.softplus = nn.Softplus()
		self.sigmoid = nn.Sigmoid()
		self.fudge = 1e-7

	def forward(self, z):
		# define the forward computation on the latent z
		# first compute the hidden units
		hidden = self.softplus(self.fc1(z))
		# return the parameter for the output Bernoulli
		# each is of size batch_size x 784
		# fixing numerical instabilities of sigmoid with a fudge
		mu_img = (self.sigmoid(self.fc21(hidden)) + self.fudge) * (1 - 2 * self.fudge)
		return mu_img


class Encoder(nn.Module):
	"""Variational Encoder"""

	def __init__(self, z_dim, hidden_dim):
		super(Encoder, self).__init__()

		# setup the three linear transformations used
		self.fc1  = nn.Linear(784, hidden_dim)
		self.fc21 = nn.Linear(hidden_dim, z_dim)
		self.fc22 = nn.Linear(hidden_dim, z_dim)

		# setup the non-linearity
		self.softplus = nn.Softplus()
		self.relu = nn.ReLU()

	def forward(self, x):
		# define the forward computation on the image x
		# first shape the mini-batch to have pixels in the rightmost dimension
		x = x.view(-1, 784)
		hidden = self.softplus(self.fc1(x))
		# then return a mean vector and a (positive) square root covariance
		# each of size batch_size x z_dim
		z_mu = self.fc21(hidden)
		z_sigma = torch.exp(self.fc22(hidden))
		return z_mu, z_sigma


class VAE(object):
	"""docstring for VAE"""

	def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
		super(VAE, self).__init__()
		self.encoder = Encoder(z_dim, hidden_dim)
		self.decoder = Decoder(z_dim, hidden_dim)
		self.z_dim = z_dim

	def model(self, x):
		# register PyTorch module `decoder` with Pyro
		pyro.module("decoder", self.decoder)

		# Setup hyperparameters for prior p(z)
		# the type_as ensures we get CUDA tensors if x in on gpu
		z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
		z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
		# sample from prior
		# (value will be sampled by guide when computing the ELBO)
		z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

		# decode the latent code z
		mu_img = self.decoder(z)
		# score against actual images
		pyro.observe("obs", dist.bernoulli, x.view(-1, 784), mu_img)

	def guide(self, x):
		# register PyTorch model 'encoder' w/ pyro
		pyro.module("encoder", self.encoder)
		# Use the encoder to get the parameters use to define q(z|x)
		z_mu, z_sigma = self.encoder(x)
		# Sample the latent code z
		pyro.sample("latent", dist.normal, z_mu, z_sigma)

	def reconstruct_img(self, x):
		# encode image x
		z_mu, z_sigma = self.encoder(x)
		# sample in latent space
		z = dist.normal(z_mu, z_sigma)
		# decode the image (note we don't sample in image space)
		mu_img = self.decoder(z)
		return mu_img

	def model_sample(self, batch_size=1):
		# Sample the handwriting style from the constant prior dist
		prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
		prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
		zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)
		mu = self.decoder.forward(zs)
		xs = pyro.sample("sample", dist.bernoulli, mu)
		return xs, mu


def main(args):
	train_loader, test_loader = get_data()

	vae = VAE(use_cuda=False)
	optimizer = Adam({"lr": 0.0001})
	svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

	# setup visdom for visualization
	if args.visdom_flag:
		vis = visdom.Visdom()

	train_elbo = []
	test_elbo = []

	# training loop
	for epoch in range(args.num_epochs):
		# initialize loss accumulator
		epoch_loss = 0.
		# do a training epoch over each mini-batch x returned
		# by the data loader
		for _, (x, _) in enumerate(train_loader):
			# wrap the mini-batch in a PyTorch Variable
			x = Variable(x)
			# do ELBO gradient and accumulate loss
			epoch_loss += svi.step(x)

		# report training diagnostics
		normalizer_train = len(train_loader.dataset)
		total_epoch_loss_train = epoch_loss / normalizer_train
		train_elbo.append(total_epoch_loss_train)
		print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

		if epoch % args.test_frequency == 0:
			# initialize loss accumulator
			test_loss = 0.
			# compute the loss over the entire test set
			for i, (x, _) in enumerate(test_loader):
				# wrap the mini-batch in a PyTorch Variable
				x = Variable(x)
				# compute ELBO estimate and accumulate loss
				test_loss += svi.evaluate_loss(x)
				# visualize how well we're reconstructing them
				if i == 0:
					if args.visdom_flag:
						plot_vae_samples(vae, vis)
			# report test diagnostics
			normalizer_test = len(test_loader.dataset)
			total_epoch_loss_test = test_loss / normalizer_test
			test_elbo.append(total_epoch_loss_test)
			print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))


if __name__ == '__main__':
	# Settings
	args = dotdict()
	args.num_epochs = 100
	args.test_frequency = 10
	args.visdom_flag = True
	main(args)
