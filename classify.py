import json
import sys
import cv2
from PIL import Image
import torch
from torchvision import transforms
import urllib

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f"Reading from file {input_file}")
print(f"Writing to file {output_file}")

# Load GoogLeNet model
model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True)
model.eval()

# Load ImageNet labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    with urllib.request.urlopen(url) as f:
        categories = [s.decode("utf-8").strip() for s in f.readlines()]
except:
    categories = [f"class_{i}" for i in range(1000)]

# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_labels = []

cap = cv2.VideoCapture(input_file)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to RGB and process
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = preprocess(img_pil)
    input_batch = input_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    label = categories[top_catid[0]]
    all_labels.append(label)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    i += 1

# Get unique labels
return_labels = list(set(all_labels))

cap.release()
cv2.destroyAllWindows()

data = {"labels": return_labels}

with open(output_file, "a") as json_file:
    json_file.write("\n")
    print(f"Writing to file {output_file}")
    json.dump(data, json_file)
