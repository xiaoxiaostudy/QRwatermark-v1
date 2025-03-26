import torch
import torch.nn as nn
from network.blocks import *
import numpy as np
from network.blocks.fcanet import *

class Fcanet_Decoder(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length,channels=64):
		super(Fcanet_Decoder, self).__init__()

		self.m_length = int(H/8 * W/8) 
		self.m_size = int(self.m_length ** 0.5)

		stride_blocks = int(np.log2(H // int(np.sqrt(self.m_length))))

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			FcaNet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = FcaNet_decoder(channels, channels, blocks=1)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = nn.Linear(H*W, message_length)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		x = self.message_layer(x)
		return x
