import numpy as np
import torch
from network.blocks import *
from network.blocks.fcanet import *
	
class FcaEncoder_MP_Diffusion(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(FcaEncoder_MP_Diffusion, self).__init__()
		self.H = H
		self.W = W

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = FcaNet(channels, channels, blocks=blocks)

		self.m_length =int(H/8 * W/8) 
		self.m_size = int(self.m_length ** 0.5)

		self.message_duplicate_layer = nn.Linear(message_length, self.m_length)
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=3)
		self.message_pre_layer_2 = FcaNet(channels, channels, blocks=1)
		self.message_first_layer = FcaNet(channels, channels, blocks=blocks)


		self.after_concat_layer = ConvBNRelu(2 * channels, channels)
		self.fcalayer = FcaNet(channels, channels, blocks=blocks)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		#Image Processor
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		#print("intermediate1 shape:",intermediate1.shape)

		# Message Processor
		message_duplicate = self.message_duplicate_layer(message)
		message_image = message_duplicate.view(-1, 1, self.m_size, self.m_size)
		message_pre_0 = self.message_pre_layer_0(message_image)
		message_pre_1 = self.message_pre_layer_1(message_pre_0)
		message_pre_2 = self.message_pre_layer_2(message_pre_1)
		intermediate2 = self.message_first_layer(message_pre_2)
		#print("intermediate2 shape:",intermediate2.shape)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)
		intermediate4 = self.fcalayer(intermediate3)
	
		
		# skip connection
		concat2 = torch.cat([intermediate4, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)

		return output
