import json
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from matplotlib import cm
from torchvision import transforms
from functools import partial

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f"Reading from file {input_file}")
print(f"Writing to file {output_file}")

utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)


torch.load = partial(torch.load, map_location=torch.device("cpu"))


class ObjectDetectionPipeline:
    def __init__(self, threshold=0.5, device="cpu", cmap_name="tab10_r"):
        self.tfms = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = (
            torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd")
            .eval()
            .to(device)
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.threshold = threshold
        self.cmap = cm.get_cmap(cmap_name)
        self.classes_to_labels = utils.get_coco_object_dictionary()
        self.all_labels = []

    @staticmethod
    def _crop_img(img):
        if len(img.shape) == 3:
            y = img.shape[0]
            x = img.shape[1]
        elif len(img.shape) == 4:
            y = img.shape[1]
            x = img.shape[2]
        else:
            raise ValueError(f"Image shape: {img.shape} invalid")

        out_size = min((y, x))
        startx = x // 2 - out_size // 2
        starty = y // 2 - out_size // 2

        if len(img.shape) == 3:
            return img[starty : starty + out_size, startx : startx + out_size]
        elif len(img.shape) == 4:
            return img[:, starty : starty + out_size, startx : startx + out_size]

    def _plot_boxes(self, output_img, labels, boxes):
        for label, (x1, y1, x2, y2) in zip(labels, boxes):
            if (x2 - x1) * (y2 - y1) < 0.25:

                x1 = int(x1 * output_img.shape[1])
                y1 = int(y1 * output_img.shape[0])
                x2 = int(x2 * output_img.shape[1])
                y2 = int(y2 * output_img.shape[0])

                rgba = self.cmap(label)
                bgr = rgba[2] * 255, rgba[1] * 255, rgba[0] * 255
                cv2.rectangle(output_img, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    output_img,
                    self.classes_to_labels[label - 1],
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return output_img

    def __call__(self, img):
        if type(img) == np.ndarray:
            img_tens = (
                self.tfms(Image.fromarray(img[:, :, ::-1])).unsqueeze(0).to(self.device)
            )

            results = utils.decode_results(self.model(img_tens))
            # print("results ", results)
            boxes, labels, conf = utils.pick_best(results[0], self.threshold)
            # print("labels ", labels)
            [self.all_labels.append(x) for x in labels]

            output_img = self._crop_img(img)

            return self._plot_boxes(output_img, labels, boxes)

        elif type(img) == list:
            if len(img) == 0:
                return None

            tens_batch = torch.cat(
                [self.tfms(Image.fromarray(x[:, :, ::-1])).unsqueeze(0) for x in img]
            ).to(self.device)
            results = utils.decode_results(self.model(tens_batch))
            # print("results "+results)
            output_imgs = []
            for im, result in zip(img, results):
                boxes, labels, conf = utils.pick_best(result, self.threshold)
                # print("labels "+labels)
                output_imgs.append(self._plot_boxes(self._crop_img(im), labels, boxes))

            return output_imgs

        else:
            raise TypeError(f"Type {type(img)} not understood")


obj_detect = ObjectDetectionPipeline(device="cpu", threshold=0.5)


cap = cv2.VideoCapture(input_file)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# cap = get_youtube_cap(url)
# pbar = tqdm(total=total_frames)

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    obj_detect(frame)[:, :, ::-1]

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    i += 1

return_labels = list(
    set([obj_detect.classes_to_labels[x - 1] for x in obj_detect.all_labels])
)

cap.release()
cv2.destroyAllWindows()

data = {"labels": return_labels[0]}

with open(output_file, "a") as json_file:
    print(f"Writing to file {output_file}")
    json.dump(data, json_file)
