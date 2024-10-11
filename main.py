from PIL import Image
import torch
from torchvision import transforms
import torchvision
from torch import nn

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

model.load_state_dict(torch.load('model_final_weights.pth'))


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


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  if 'image' not in request.files:
        return "No image is uploaded !"
  file = request.files["image"]
  file.save("static/input.jpg");
  return render_template("output.html", result=predict_image_class('static/input.jpg', model, class_names, device='cpu'))

app.run()