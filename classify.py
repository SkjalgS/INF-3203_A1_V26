import json
import sys
import cv2
import urllib
from typing import List
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import GoogLeNet_Weights


class ImageClassificationPipeline:
    """Image classification pipeline using a pre-trained GoogLeNet model."""

    def __init__(self) -> None:
        """Initialize the image classification pipeline."""
        self.device = "cpu"
        self.tfms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = (
            torch.hub.load(
                "pytorch/vision:v0.10.0", "googlenet", weights=GoogLeNet_Weights.DEFAULT
            )
            .eval()
            .to(self.device)
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.categories = self._load_imagenet_labels()
        self.all_labels = []

    def _load_imagenet_labels(self) -> List[str]:
        """Load ImageNet labels from a remote file.

        Returns:
            List[str]: List of ImageNet class labels.
        """
        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        try:
            with urllib.request.urlopen(url) as f:
                return [s.decode("utf-8").strip() for s in f.readlines()]

        except Exception as e:
            print(f"Could not download labels file: {e}")
            return [f"class_{i}" for i in range(1000)]

    def __call__(self, img: np.ndarray) -> str:
        """Classify an image and return the predicted label.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            str: Predicted label.
        """

        # Convert BGR to RGB and process
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.tfms(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        label = self.categories[top_catid[0]]
        self.all_labels.append(label)

        return label


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: Missing required arguments")
        print("Usage: python classify.py <input_video> <output_json>")
        print("  input_video: Path to the input video file")
        print("  output_json: Path to the output JSON file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading from file {input_file}")
    print(f"Writing to file {output_file}")

    classifier = ImageClassificationPipeline()

    cap = cv2.VideoCapture(input_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        classifier(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i += 1

    # Get unique labels
    return_labels = list(set(classifier.all_labels))

    cap.release()
    cv2.destroyAllWindows()

    data = {"labels": return_labels}

    with open(output_file, "a") as json_file:
        json_file.write("\n")
        print(f"Writing to file {output_file}")
        json.dump(data, json_file)
