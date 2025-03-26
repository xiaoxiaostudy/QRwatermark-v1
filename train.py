import numpy as np
import time,os,json
import torch
import argparse
from torch.utils.data import DataLoader
from save_images import *
from network.Network import *
from util.MyDataloader import MyDataset

'''
train
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
	parser = argparse.ArgumentParser()

	def aa(*args,**kwargs):
		group.add_argument(*args,**kwargs)

	group = parser.add_argument_group('Experiments parameters')
	aa("--train_dir",type=str,default="datasets/train")
	aa("--val_dir",type=str,default="datasets/test")
	aa("--output_dir",type=str,default="results/", help="Output directory for logs and images")
	aa("--train_continue",type=bool,default=False)
	aa("--train_continue_path",type=str,default="")
	aa("--train_continue_epoch",type = int , default = 1)
	aa("--ckpt_path",type=str,default="ckpt/")
	aa("--vqvae",type=str,default="vq-f4",help="vq-f4 , vq-f4-noattn , vq-f8 , vq-f8-n256 , vq-f16")

	group = parser.add_argument_group("Image and message size")
	aa("--image_H", type = int , default = 256)
	aa("--image_W", type = int , default = 256)
	aa("--message_length", type = int, default = 64)

	group = parser.add_argument_group("Optimization parameters")
	aa("--epochs", type=int, default=300, help="Number of epochs for optimization. (Default: 300)")
	aa("--lr",type = float,default=5e-4)
	aa("--encoder_weight", type=float, default=10)
	aa("--encoder_percepweight", type=float, default=0.1)
	aa("--discriminator_weight", type=float, default=0.001)
	aa("--decoder_weight", type=float, default=10)

	group = parser.add_argument_group('Loader parameters')
	aa("--batch_size", type=int, default=4, help="Batch size. ")
	aa("--workers", type=int, default=6, help="Number of workers for data loading.")
	return parser
	
def main(params):


	network = Network(H=params.image_H, 
				   W=params.image_W, 
				   message_length=params.message_length,
				   device=device,
				   batch_size=params.batch_size, 
				   lr=params.lr,
				   encoder_weight=params.encoder_weight,
				   encoder_percepweight=params.encoder_percepweight,
				   discriminator_weight=params.discriminator_weight,
				   decoder_weight=params.decoder_weight,
				   ckpt_path=params.ckpt_path,
				   vqvae=params.vqvae)


	train_dataset = MyDataset(params.train_dir, params.image_H, params.image_W)
	train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.workers, pin_memory=True)

	val_dataset = MyDataset(params.val_dir, params.image_H, params.image_W)
	val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.workers, pin_memory=True)

	if params.train_continue:
		if params.data_augmentation:
			EC_path = "results/" + params.train_continue_path + "/models/EC_" + str(params.train_continue_epoch) + ".pth"
			D_path = "results/" + params.train_continue_path + "/models/D_" + str(params.train_continue_epoch) + ".pth"
			network.load_model(EC_path, D_path)
			begin_epoch = 0
			result_folder = params.output_dir + str(params.train_continue_epoch) + "_" +"random_rotation(4)/"
			if not os.path.exists(result_folder): os.mkdir(result_folder)
			if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
			if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
			with open(result_folder + "/train_params.txt", "w") as file:
				content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
				file.write(content)
				params_dict = vars(params) 
				json.dump(params_dict, file, indent=4)
				file.write("\n")
			with open(result_folder + "/train_log.txt", "w") as file:
				content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
				file.write(content)
			with open(result_folder + "/val_log.txt", "w") as file:
				content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
				file.write(content)
		
		else:
			EC_path = "results/" + params.train_continue_path + "/models/EC_" + str(params.train_continue_epoch) + ".pth"
			D_path = "results/" + params.train_continue_path + "/models/D_" + str(params.train_continue_epoch) + ".pth"
			network.load_model(EC_path, D_path)
			result_folder = "results/" + params.train_continue_path + "/"
			begin_epoch = params.train_continue_epoch
	else:
		begin_epoch = 0
		result_folder = params.output_dir + time.strftime( "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/"
		if not os.path.exists(result_folder): os.mkdir(result_folder)
		if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
		if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
		with open(result_folder + "/train_params.txt", "w") as file:
			content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
			file.write(content)
			params_dict = vars(params) 
			json.dump(params_dict, file, indent=4)
			file.write("\n")
		with open(result_folder + "/train_log.txt", "w") as file:
			content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
			file.write(content)
		with open(result_folder + "/val_log.txt", "w") as file:
			content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",time.localtime()) + "-----------------------\n"
			file.write(content)
	
	print("\nStart training : \n\n")
	print("result_folder:",result_folder)
	
	save_images_number = 2
	W = params.image_W 
	H = params.image_H 

	# Add best model tracking variables
	best_error_rate = float('inf')
	best_psnr = float('-inf')
	best_epoch = 0

	for epoch in range(begin_epoch,params.epochs):
	
		running_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		#"g_dinoloss_on_discriminator":0.0,
		"g_loss_on_discriminator": 0.0,
		"g_perceploss_on_encoder": 0.0,
		"g_loss_on_encoder":0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0
		}

		start_time = time.time()

		'''
		train
		'''
		num = 0
		total_batches = len(train_dataloader)
		for batch_idx, images in enumerate(train_dataloader):
			image = images.to(device)
			message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], params.message_length))).to(device)

			result = network.train(image, message)

			for key in result:
				running_result[key] += float(result[key])

			num += 1
			
			# Add progress printing
			if batch_idx % 10 == 0:  # Print every 10 batches
				print(f"\rTraining: {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%) - "
					  f"Loss: {running_result['g_loss']/num:.4f} - "
					  f"Error Rate: {running_result['error_rate']/num:.4f} - "
					  f"PSNR: {running_result['psnr']/num:.2f} - "
					  f"SSIM: {running_result['ssim']/num:.4f}", end='')

		'''
		train results
		'''
		content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
		for key in running_result:
			content += key + "=" + str(running_result[key] / num) + ","
		content += "\n"

		with open(result_folder + "/train_log.txt", "a") as file:
			file.write(content)
		print(content)

		'''
		validation
		'''

		val_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		#"g_dinoloss_on_discriminator":0.0,
		"g_loss_on_discriminator": 0.0,
		"g_perceploss_on_encoder": 0.0,
		"g_loss_on_encoder":0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0
		}

		start_time = time.time()

		saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
		saved_all = None

		num = 0
		total_val_batches = len(val_dataloader)
		for i, images in enumerate(val_dataloader):
			image = images.to(device)
			message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], params.message_length))).to(device)

			result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

			for key in result:
				val_result[key] += float(result[key])

			num += 1
			
			# Add validation progress printing
			if i % 10 == 0:  # Print every 10 batches
				print(f"\rValidating: {i+1}/{total_val_batches} ({(i+1)/total_val_batches*100:.1f}%) - "
					  f"Loss: {val_result['g_loss']/num:.4f} - "
					  f"Error Rate: {val_result['error_rate']/num:.4f}", end='')

			if i in saved_iterations:
				if saved_all is None:
					saved_all = get_random_images(image, encoded_images, noised_images)
				else:
					saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

		save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

		'''
		validation results
		'''
		current_error_rate = val_result['error_rate'] / num
		current_psnr = val_result['psnr'] / num
		content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
		for key in val_result:
			content += key + "=" + str(val_result[key] / num) + ","
		content += "\n"

		with open(result_folder + "/val_log.txt", "a") as file:
			file.write(content)
		print(content)

		'''
		save model
		'''
		# Save model if it's the best so far
		if current_error_rate < best_error_rate:
			best_error_rate = current_error_rate
			best_psnr = current_psnr
			best_epoch = epoch
			path_model = result_folder + "models/"
			path_encoder_decoder = path_model + "best_{}_EC.pth".format(epoch)
			path_discriminator = path_model + "best_{}_D.pth".format(epoch)
			network.save_model(path_encoder_decoder, path_discriminator)
			print(f"New best model saved! (Epoch {epoch}, Error Rate: {best_error_rate:.4f}, PSNR: {best_psnr:.2f})")
		elif current_error_rate == best_error_rate and current_psnr > best_psnr:
			best_psnr = current_psnr
			best_epoch = epoch
			path_model = result_folder + "models/"
			path_encoder_decoder = path_model + "best_{}_EC.pth".format(epoch)
			path_discriminator = path_model + "best_{}_D.pth".format(epoch)
			network.save_model(path_encoder_decoder, path_discriminator)
			print(f"New best model saved! (Same Error Rate but better PSNR) (Epoch {epoch}, Error Rate: {best_error_rate:.4f}, PSNR: {best_psnr:.2f})")
				

		# Also save checkpoints every 10 epochs
		if (epoch + 1) % 10 == 0:
			path_model = result_folder + "models/"
			path_encoder_decoder = path_model + f"EC_epoch_{epoch}.pth"
			path_discriminator = path_model + f"D_epoch_{epoch}.pth"
			network.save_model(path_encoder_decoder, path_discriminator)

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)