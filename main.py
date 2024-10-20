from PIL import Image
import torch
from torchvision import transforms
import torchvision
from torch import nn
import torch.nn.functional as F

from flask import Flask, request, render_template
app = Flask(__name__)


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ['processed_lsd_jpgs',
 'processed_osf_jpgs',
 'processed_spider_jpgs',
 'processed_tseg_jpgs']


model = torchvision.models.efficientnet_b0(weights=weights).to(device)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(class_names),
              bias=True).to(device))

model.load_state_dict(torch.load('model_weights.pth',weights_only=True))


model2 = torchvision.models.efficientnet_b0(weights=weights).to(device)
for param in model2.features.parameters():
    param.requires_grad = False

model2.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, 
              out_features=2,
              bias=True).to(device))

model2.load_state_dict(torch.load('model_Spine_Or_No.pth',weights_only=True))


def predict_image_class(image_path, model, class_names, device='cpu'):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_batch)

        _, predicted_idx = torch.max(output, 1)

    predicted_class = class_names[predicted_idx.item()]

    return predicted_class

def predict_image_class2(image_path, model ,device='cpu'):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image)
    batch = tensor.unsqueeze(0)
    
    batch = batch.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(batch)
        probabilities = F.softmax(output, dim=1)
        
        _, predicted_idx = torch.max(output, 1)
        confidence = probabilities[0][predicted_idx].item() * 100  
    
    return predicted_idx.item(), confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  if 'image' not in request.files:
        return "No image is uploaded !"
  file = request.files["image"]
  file.save("static/input.jpg")
  if(predict_image_class2('static/input.jpg', model2, device='cpu')[0] == 0):
      return render_template("error.html")
  return render_template("output.html", result=predict_image_class('static/input.jpg', model,class_names, device='cpu'))

app.run()