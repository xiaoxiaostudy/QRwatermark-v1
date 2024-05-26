import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from .Encoder_MP import FcaEncoder_MP_Diffusion
from .Decoder import Fcanet_Decoder
from .Discriminator import NLayerDiscriminator
from diffusion.ldm.models.VQModel import VQModel
from transformers import AutoImageProcessor, AutoModel
from loss.loss_provider import LossProvider
class Network:

	def __init__(self, H, W, message_length, device, batch_size, lr,encoder_weight,encoder_percepweight,discriminator_weight,decoder_weight):
		# device
		self.device = device

		#encoder+vqvae+decoder
		self.encoder = FcaEncoder_MP_Diffusion(H, W, message_length)
		self.decoder = Fcanet_Decoder(H, W, message_length)

		ddconfig={"double_z": False,"z_channels": 3,"resolution": 256,"in_channels": 3,"out_ch": 3,"ch": 128,"ch_mult": [ 1,2,4 ], "num_res_blocks": 2,"attn_resolutions": [ ],"dropout": 0.0}
		vqvae_aug = VQModel(ddconfig, n_embed=8192, embed_dim=3, ckpt_path='/home/wangyue2/xiaoxiao/biggerwatermarkvqvae/regeneration/ckpt/vq_f4_model.ckpt')
		self.vqvae_aug = vqvae_aug.to(device)
		for p in self.vqvae_aug.parameters():
			p.requires_grad = False

		self.vqmodel = VQmodel(self.encoder,self.decoder, self.vqvae_aug)
		self.vqmodel.to(device)

		#discriminator
		self.discriminator = NLayerDiscriminator(input_nc=3).to(device)

		# mark "cover" as 1, "encoded" as 0
		self.label_cover = torch.ones((batch_size, 1, 14, 14), dtype=torch.float, device=device)
		self.label_encoded = torch.zeros((batch_size, 1, 14, 14), dtype=torch.float, device=device)

		# optimizer
		print(lr)
		self.opt_vqmodel = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.vqmodel.parameters()), lr=lr)
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr,weight_decay=1e-5)

		# loss function
		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
		self.criterion_MSE = nn.MSELoss().to(device)
		provider = LossProvider()
		loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
		self.loss_percep = loss_percep.to(device)

		# weight of encoder-decoder loss
		self.discriminator_weight = discriminator_weight
		self.encoder_percepweight = encoder_percepweight
		self.encoder_weight = encoder_weight
		self.decoder_weight = decoder_weight


	def train(self, images: torch.Tensor, messages: torch.Tensor):
		self.discriminator.train()
		self.vqmodel.train()

		with torch.enable_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.vqmodel(images, messages)

			'''
			train discriminator
			'''
			self.opt_discriminator.zero_grad()


			# RAW: target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# GAN: target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()

			'''
			train encoder and decoder
			'''
			self.opt_vqmodel.zero_grad()

			# GAN : target label for encoded image should be "cover"(1)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
			
			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)
			loss_i=lambda encoded_images, images: self.loss_percep((1+encoded_images)/2.0, (1+images)/2.0)/ encoded_images.shape[0]
			g_perceploss_on_encoder = loss_i(encoded_images,images)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss =self.discriminator_weight*g_loss_on_discriminator + self.encoder_percepweight * g_perceploss_on_encoder +\
				self.encoder_weight * g_loss_on_encoder + self.decoder_weight * g_loss_on_decoder
			g_loss.backward()
			self.opt_vqmodel.step()

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			#"g_dinoloss_on_discriminator":g_dinoloss_on_discriminator,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_perceploss_on_encoder": g_perceploss_on_encoder,
			"g_loss_on_encoder":g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result

	def validation(self, images: torch.Tensor, messages: torch.Tensor):
		self.discriminator.eval()
		self.vqmodel.eval()

		with torch.no_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.vqmodel(images, messages)

			'''
			validate discriminator
			'''
			
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			
			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(1)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			loss_i=lambda encoded_images, images: self.loss_percep((1+encoded_images)/2.0, (1+images)/2.0)/ encoded_images.shape[0]
			g_perceploss_on_encoder = loss_i(encoded_images,images)
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight*g_loss_on_discriminator + self.encoder_percepweight * g_perceploss_on_encoder+\
				self.encoder_weight * g_loss_on_encoder + self.decoder_weight * g_loss_on_decoder

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			#"g_dinoloss_on_discriminator":g_dinoloss_on_discriminator,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_perceploss_on_encoder": g_perceploss_on_encoder,
			"g_loss_on_encoder":g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		error_rate /= batch_size
		return error_rate
	
	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.vqmodel.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)
	def load_model_ed(self, path_encoder_decoder: str):
		self.vqmodel.load_state_dict(torch.load(path_encoder_decoder))

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.load_state_dict(torch.load(path_discriminator))

class VQmodel(nn.Module):
	def __init__(self, encoder,decoder, vqvae_aug):
		super(VQmodel, self).__init__()
		self.encoder = encoder
		self.vqvae_aug = vqvae_aug
		self.decoder = decoder

	def forward(self, images, messages):
		encoded_images = self.encoder(images, messages)
		vimage, emb_loss, info = self.vqvae_aug.encode(encoded_images)
		noised_images = self.vqvae_aug.decode(vimage)
		decoded_messages = self.decoder(noised_images)
		return encoded_images, noised_images, decoded_messages
	

			
	
	


	

    

