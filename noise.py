import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
from io import BytesIO

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


def convert(image):
	images = image.squeeze(0)
	images = (images.cpu() + 1) / 2
	images = images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
	image_np = images.numpy().transpose(1, 2, 0)
	saved_image = Image.fromarray(image_np).convert("RGB")
	return saved_image

def convert_to_tensor(image,H,W):
	transform = transforms.Compose([
			transforms.Resize((H,W)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#0到1映射到-1到1
		])
	tensor = transform(image)
	return tensor

#-----------------------Identity-----------------------------
class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, image):
		return image

#-------------------Geometric distortions-------------------------
#旋转角度 （0，45）
class Rotation(nn.Module):
    def __init__(self,angle=5):
        super(Rotation,self).__init__()
        self.angle = angle

    def forward(self,image, angle=None):
        if angle is not None:
            self.angle = angle
        img = F.rotate(image, self.angle)
        return img



#随机剪裁矩形，剩下的遮罩
class Crop(nn.Module):

	def __init__(self, ratio):
		super(Crop, self).__init__()
		self.ratio = ratio

	def forward(self, image,ratio=None):
		if ratio is not None:
			self.ratio = ratio
		_,_, h, w = image.shape
		h_start = int(h * (1 - self.ratio) / 2)
		h_end = h - h_start
		w_start = int(w * (1 - self.ratio) / 2)
		w_end = w - w_start

		#h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.ratio,self.ratio)
		mask = torch.zeros_like(image)
		mask[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask
'''
#encodedimage中随机选择一块矩形用原图像替代
class Cropout(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Cropout, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image,encodedimage):
		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		output = encodedimage.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output
'''

#剪裁矩形并缩放回原来的大小(1,0.5)
class Resizedcrop(nn.Module):
    def __init__(self,scale):
        super(Resizedcrop,self).__init__()
        self.scale = scale

    def forward(self,image,scale=None):
        if scale is not None:
            self.scale = scale
        _,_, height, width = image.shape

        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(self.scale, self.scale), ratio=(1, 1))

        distorted_image = F.resized_crop(image, i, j, h, w, (height, width))
        return distorted_image


#随机抹去一个矩形区域
class Erasing(nn.Module):
    def __init__(self,scale):
        super(Erasing,self).__init__()
        self.scale = scale

    def forward(self,image,scale=None):
        if scale is not None:
            self.scale = scale
        i, j, h, w, v = T.RandomErasing.get_params(image, scale=(self.scale, self.scale), ratio=(1, 1), value=[0])
        distorted_image = F.erase(image, i, j, h, w, v)
        return distorted_image

#--------------------Photometric distortions-------------------
#调整亮度
class Brightness(nn.Module):
    def __init__(self, min_brightness=0.8, max_brightness=1.2):
        super(Brightness, self).__init__()
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def forward(self, image, min_brightness=None, max_brightness=None):
        if min_brightness is not None:
            self.min_brightness = min_brightness
            self.max_brightness = max_brightness
        brightness_scale = torch.empty(image.size(0)).uniform_(self.min_brightness, self.max_brightness)
        brightness_scale = brightness_scale.view(image.size(0), 1, 1, 1).to(image.device)
        adjusted_image = image * brightness_scale
        return adjusted_image.clamp(-1, 1)

'''
#调整对比度
class Contrast(nn.Module):
    def __init__(self, min_contrast=0.8, max_contrast=1.2):
        super(Contrast, self).__init__()
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def forward(self, image, min_contrast=None, max_contrast=None):
        if min_contrast is not None:
            self.min_contrast = min_contrast
            self.max_contrast = max_contrast
        
        contrast_scale = torch.empty(image.size(0)).uniform_(self.min_contrast, self.max_contrast)
        contrast_scale = contrast_scale.view(image.size(0), 1, 1).to(image.device)

        
        mean = image.mean(dim=(1, 2), keepdim=True)

        
        adjusted_image = (image - mean) * contrast_scale + mean

        
        return adjusted_image.clamp(0, 1)
'''
class Contrast(nn.Module):
    def __init__(self, contrast=0.5):
        super(Contrast, self).__init__()
        self.contrast = contrast
        

    def forward(self, image, contrast=None):
        if contrast is not None:
            self.contrast = contrast
        contrast_scale = torch.empty(image.size(0)).uniform_(self.contrast)
        contrast_scale = contrast_scale.view(image.size(0), 1, 1, 1).to(image.device)

        
        mean = image.mean(dim=(1, 2, 3), keepdim=True)

        
        adjusted_image = (image - mean) * contrast_scale + mean

        
        return adjusted_image.clamp(-1, 1)

  
#调整饱和度
class Saturation(nn.Module):
    def __init__(self,rnd_sat=1.0) -> None:
        super(Saturation,self).__init__()
        self.rnd_sat=torch.rand(1)[0] * 1.0 * rnd_sat
    
    def forward(self,image,rnd_sat=None):
        if rnd_sat is not None:
            self.rnd_sat = torch.rand(1)[0] * 1.0 * rnd_sat
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1,3, 1, 1).to(image.device)
        encoded_image_lum = torch.mean(image * sat_weight, dim=1, keepdim=True)
        image = (1 - self.rnd_sat) * image + self.rnd_sat * encoded_image_lum
        image.clamp(-1,1)
        return image

#---------------Degradation distortions----------------------

class Blurring(nn.Module):
    def __init__(self,N_blur = 7):
        super(Blurring,self).__init__()
        self.blur = N_blur

    def forward(self,image,N_blur=None):
        if N_blur is not None:
            self.blur = N_blur
        f = F.gaussian_blur(image,kernel_size=N_blur)

        return f

#添加高斯噪声
class Gnoise(nn.Module):
    def __init__(self,rnd_noise=0.02):
        super(Gnoise,self).__init__()
        self.rnd_noise = rnd_noise

    def forward(self,image,rnd_noise=None):
        if rnd_noise is not None:
            self.rnd_noise = rnd_noise
        noise = torch.normal(mean=0, std=self.rnd_noise, size=image.size(), dtype=torch.float32).to(image.device)
        image = image + noise
        image.clamp(-1,1)
        return image

#jpeg压缩
class JPEG(nn.Module):
    def __init__(self, quality=75):
        """
        初始化 JPEG 压缩模块。
        :param quality: JPEG 压缩的质量1 到 100,值越高质量越好，压缩比越低）
        """
        super(JPEG, self).__init__()
        self.quality = quality

    def forward(self, image,quality=None):
        """
        前向传递应用 JPEG 压缩。
        :param image: 输入图像（形状为 (batch_size, channels, height, width))
        :return: 经过 JPEG 压缩后的图像
        """
        #_, _, height, width = image.shape
        if quality is not None:
            self.quality = quality
        _, _, height, width = image.shape
        output_images = []

        for img in image:
            img_pil = convert(img)
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            img_pil_jpeg = Image.open(buffer)
            img_tensor =  convert_to_tensor(img_pil_jpeg,height,width)
            output_images.append(img_tensor)

        output = torch.stack(output_images).to(image.device)

        
        return output


#--------------Combined distortions--------------
#随机选一个
class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, image):
		id = get_random_int([0, len(self.list) - 1])
		return self.list[id](image)




