import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
import vision_transformer as vits
from tqdm import tqdm
import gc


class loader():
    def load_images(path, method):
        if method == "ALL":
            filenames = []
            dir_list = os.listdir(path)
            for i in range(0, len(dir_list)):
                com_path = os.path.join(path, dir_list[i])
                if com_path[-4:] == ".jpg":
                    filenames.append(com_path)
                elif os.path.isdir(com_path):
                    filenames.extend(loader.load_images(com_path, "ALL"))

        elif method == "FOLDER":
            filenames = []
            for file in os.listdir(path):
                if file[-4:] == ".jpg":
                    filenames.append(path+file)

        elif method == "IMAGE":
            filenames = [path]

        return filenames


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)

state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth")
model.load_state_dict(state_dict, strict=True)

transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
folder = "Basalt/"
if not os.path.exists(folder):
    os.mkdir(folder)
# main
images = loader.load_images("Rocks/"+folder, "ALL")
for image in tqdm(images):
    # open image
    org_img = cv2.imread(image)
    img = transform(org_img)
    w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - \
        img.shape[2] % 8  # remove remain
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // 8
    h_featmap = img.shape[-1] // 8

    # del model
    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    model.load_state_dict(state_dict, strict=True)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        attentions = model.get_last_selfattention(img.to(device))
        nh = attentions.shape[1]  # number of head

        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(
            0), scale_factor=8, mode="nearest")[0].cpu().numpy()
        mask = attentions[5]
        proc_list = []
        for i in range(3):
            org_split = np.resize(
                cv2.split(org_img)[i], (mask.shape[0], mask.shape[1]))
            proc_list.append((org_split*mask).T)
        processed = np.array(proc_list).T*255
        cv2.imwrite(folder+image.split("/")[-1], processed)
    except:
        print(image.split("/")[-1], " can't be processed")
