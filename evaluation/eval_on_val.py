# camera-ready

import sys
from tkinter import Image

sys.path.append("./bruinbot")
from datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("./bruinbot/model")
from deeplabv3 import DeepLabV3

sys.path.append("./bruinbot/utils")
from utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from PIL import Image

batch_size = 2

network = DeepLabV3("eval_val", project_dir="./bruinbot")
network.load_state_dict(torch.load("./bruinbot/pretrained_models/model_13_2_2_2_epoch_580.pth", map_location=torch.device('cpu')))

# val_dataset = DatasetVal(cityscapes_data_path="./bruinbot/data/cityscapes",
#                          cityscapes_meta_path="./bruinbot/data/cityscapes/meta")

# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_val_batches:", num_val_batches)

# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=1)

# with open("./bruinbot/data/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
#     class_weights = np.array(pickle.load(file))
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor))

# # loss function
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
# for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):

with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
    imgs = Image.open("./bruinbot/images/sidewalk.png")
    # imgs = Variable(imgs) # (shape: (batch_size, 3, img_h, img_w))
    # label_imgs = Variable(label_imgs.type(torch.LongTensor)) # (shape: (batch_size, img_h, img_w))

    outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

    ########################################################################
    # save data for visualization:
    ########################################################################
    outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    for i in range(pred_label_imgs.shape[0]):
        if i == 0:
            pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
            # img_id = img_ids[i]
            img = imgs[i] # (shape: (3, img_h, img_w))

            img = img.data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
            img = img*np.array([0.229, 0.224, 0.225])
            img = img + np.array([0.485, 0.456, 0.406])
            img = img*255.0
            img = img.astype(np.uint8)

            pred_label_img_color = label_img_to_color(pred_label_img)
            overlayed_img = 0.35*img + 0.65*pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)

            cv2.imwrite(network.model_dir + "/" + "_overlayed.png", overlayed_img)