
CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
    'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
    'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
    'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush']

COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# feature set for our model


FEATURE_SET = [
    'nose L eyes dist',
    'nose R eyes dist',
    'ear dist',
    'ear dist X',
    'ear dist Y',
    'shoulders dist',
    'shoulders dist X',
    'hips dist',
    'hips dist X',
    'hips dist Y',
    'ankles dist',
    'elbow dist',
    'knee dist',
]

FEATURE_NUMB = len(FEATURE_SET)

#######
# Q_LEARNING
#######

ACTIONS = {
    'UP':    0,
    'DOWN':  1,
    'LEFT':  2,
    'RIGHT': 3
}

ACTIONS_NUMB = len(ACTIONS)

GAMMA = 1
EPSILON = 0.025
LEARNING_RATE = 0.008

###### best params
# RL 0.008 30x2500, epsilon = 0.025



EPOCH = 2500
BATCH_SIZE = 30

#######
# VIDEO CONF
#######

# VIDEO_STREAM_PATH = './videos/sample.mp4'
VIDEO_STREAM_PATH = './videos/simple.mp4'

DATASET_PATH = './datasets/balanced.csv'


TO_SKIP = 10  # frame to skip to have a good accuracy on traj prediction

#######
# GENERAL SETTINGS
#######

DEBUG = True
TEST = True
STATUS_BAR = True
