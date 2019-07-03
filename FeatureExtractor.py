import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2


class FeatureExtractor:

    def __init__(self):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        # check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # send model to GPU
        self.model.to(device)
        # enable evaluation mode (no training)
        self.model.eval()
        # variable for predictions
        self.predictions = None

    def feed(self, frame):
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        frame = data_transforms(frame)
        frame = frame.float().cuda()

        with torch.no_grad():
            predictions = self.model([frame])
        self.predictions = predictions

    def get_features(self):
        if self.predictions is not None:
            # boxes = self.predictions[0]['boxes']
            # labels = self.predictions[0]['labels']
            scores = self.predictions[0]['scores']
            keypoint = self.predictions[0]['keypoints']
            scores = scores.cpu()
            keypoint = keypoint.cpu()
            if len(['' for s in scores if s > 0.8]) != 1:
                return None  # skip this frame if more or less than one person was detected
            # creating features from key-points
            features = [
                [
                    np.linalg.norm(k[0] - k[2]),  # right eyes - nose distance
                    np.linalg.norm(k[0] - k[1]),  # left eyes - nose distance
                    np.linalg.norm(k[3] - k[4]),  # ear distance
                    abs(k[3][0] - k[4][0]),  # ear distance x
                    abs(k[3][1] - k[4][1]),  # ear distance y
                    np.linalg.norm(k[5] - k[6]),  # shoulder distance
                    abs(k[5][0] - k[6][0]),  # shoulder distance x
                    np.linalg.norm(k[11] - k[12]),  # hip distance
                    abs(k[11][0] - k[12][0]),  # hip distance x
                    abs(k[11][1] - k[12][1]),  # hip distance y
                    np.linalg.norm(k[15] - k[16]),  # ankle distance
                    np.linalg.norm(k[7] - k[8]),  # elbow distance
                    np.linalg.norm(k[13] - k[14]),  # knee distance
                ]
                for i, k in enumerate(keypoint)
                if scores[i] > 0.8
            ]
            return features

    def get_center(self):
        boxes = self.predictions[0]['boxes']
        scores = self.predictions[0]['scores']
        keypoint = self.predictions[0]['keypoints']
        boxes.cpu()
        scores.cpu()
        if len(['' for s in scores if s > 0.8]) != 1:
            return None  # skip this frame if more or less than one person was detected

        # delete
        mean = []
        meany = []
        for i, box in enumerate(boxes):
            if scores[i] > 0.8:
                for k in keypoint[i]:
                    x, y, visibility = k
                    mean.append(int(x))
                    meany.append(int(y))
        x = np.mean(mean)
        y = np.mean(meany)
        return x, y
        # delete

    def get_gran_truth(self, center, next_center):
        x_diff, y_diff = np.subtract(next_center, center)
        # print("x diff: {}, y diff: {}".format(x_diff, y_diff))
        if abs(x_diff) > abs(y_diff):
            if x_diff > 0:
                return 'RIGHT'
            else:
                return 'LEFT'
        if abs(x_diff) < abs(y_diff):
            if y_diff < 0:
                return 'UP'
            else:
                return 'DOWN'
