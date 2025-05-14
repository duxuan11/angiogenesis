
import torch


batch_size, num_objects, H, W = 2, 1, 224, 224
logits = torch.randn(batch_size, num_objects, H, W)  # 随机logits

upsampler = torch.hub.load("mhamilton723/FeatUp", 'vit').cuda() # image tensor has size 224x224
hr_feats = upsampler(logits) # feature map has size 256x256
hr_feats_14 = torch.nn.functional.interpolate(hr_feats, scale_factor=14/16, mode="bilinear") # feature map has size 224x224
