import torch
from models.birefnet import BiRefNet
import os
import argparse
from utils.utils import path_to_image
from dataloader import init_dataloader
from torchvision.utils import save_image

device = "cuda:0"
def inference(model, data_loader):
    model.eval()
    with torch.no_grad():

        for batch_idx, batch in enumerate(data_loader):
            batch_data = {k: v.to(device) for k, v in batch.items()}
            inputs, gts = batch_data['images'], batch_data['instances']
            points = batch_data['points']
            for i in range(points.shape[1]):
                points[0,i] = torch.tensor([1,1,100])

            prev_mask = torch.zeros_like(inputs, dtype=torch.float32)[:, :1, :, :]
            prompts = {'points': points, 'prev_mask': prev_mask}
            prompt_feats = model.get_prompt_feats(inputs.shape, prompts)
            scaled_preds = model(inputs,prompt_feats)
            pred = scaled_preds[-1].sigmoid()
            img = pred[0][0]
            img[ img >= 0.5] = 255
            img[ img < 0.5]  = 0
            img1 = pred[1][0]
            img1[ img1 >= 0.5] = 255
            img1[ img1 < 0.5]  = 0
            save_image(img, f'output/{batch_idx}.png')
            save_image(img1, f'output/{batch_idx}_0.png')

def check_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def main(weights,image_path):
    # Init model
    train_data, val_data=init_dataloader(False)
    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(weights, map_location='cpu')
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to("cuda:0")
    inference(model=model,data_loader=val_data)

if __name__ == '__main__':
    main("E:\\avatarget\\Git\\test\\gui\\models\\weights\\epoch_150.pth","E:\\avatarget\\Git\\test\\gui\\images\\1712710822.1444118_mask.jpg")
