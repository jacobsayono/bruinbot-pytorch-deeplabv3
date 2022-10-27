import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append("./utils")
from utils import label_img_to_color # Need to figure out a way to use this

# sys.path.append(".")
# from datasets import DatasetVal

sys.path.append("./model")
from deeplabv3 import DeepLabV3

model = DeepLabV3("eval_val", project_dir=".")
model.load_state_dict(torch.load("./pretrained_models/model_13_2_2_2_epoch_580.pth", map_location=torch.device('cpu')))

model.eval()

# Define input image
input_image = Image.open('images/sidewalk.png').convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

with torch.no_grad():
    output = model(input_batch)[0]
output_predictions = output.argmax(0)

# Colors adopted from utils.py
colors = torch.tensor(
        [[128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32],
        [81,  0, 81]]
    )
colors = colors.numpy().astype("uint8")


r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

# Overlay mask over image
plt.imshow(input_image, cmap='gray')
plt.imshow(r, cmap='jet', alpha=0.75)
plt.show()