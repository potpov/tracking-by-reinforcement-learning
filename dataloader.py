import pandas as pd
import pickle as pkl
import cv2
import conf
import FeatureExtractor
import os


def skip_some(vs, to_skip=conf.TO_SKIP):
    for i in range(to_skip):
        vs.read()


feature_ext = FeatureExtractor.FeatureExtractor()

# loading video
vs = cv2.VideoCapture(conf.VIDEO_STREAM_PATH)
assert vs.isOpened(), 'Cannot capture source, bad video path?'

length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
_, current = vs.read()
skip_some(vs)
loading = conf.TO_SKIP  # counter for the status bar
retval, next = vs.read()  # for gran truth

# creating data-set
df = pd.DataFrame()

while retval:
    frame = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
    next_frame = cv2.cvtColor(next, cv2.COLOR_BGR2RGB)  # for gran truth
    # CURRENT FRAME PART
    feature_ext.feed(frame)
    features = feature_ext.get_features()
    if not features:
        current = next
        skip_some(vs)
        loading += conf.TO_SKIP
        retval, next = vs.read()
        continue  # skip this frame
    center = feature_ext.get_center()

    # NEXT FRAME PART
    feature_ext.feed(next_frame)
    next_center = feature_ext.get_center()
    while next_center is None:
        skip_some(vs)
        loading += conf.TO_SKIP
        retval, next = vs.read()
        next_frame = cv2.cvtColor(next, cv2.COLOR_BGR2RGB)
        feature_ext.feed(next_frame)
        next_center = feature_ext.get_center()

    gran_truth = feature_ext.get_gran_truth(center, next_center)

    df = df.append(
        pd.concat(
            [
                pd.Series({k: v for k, v in zip(conf.FEATURE_SET, features[0])}),
                pd.Series({'gran truth': gran_truth})
            ], axis=0
        ), ignore_index=True
    )

    # LOADING BAR
    if conf.STATUS_BAR:
        loading += 1
        os.system('clear')
        print("{}% processed".format((loading*100)//length))

    # DEBUG STUFF
    if conf.DEBUG:
        cv2.circle(frame, (int(center[0]), int(center[1])), 1, (0, 255, 0), 2)
        cv2.circle(frame, (int(next_center[0]), int(next_center[1])), 1, (0, 255, 255), 2)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)

    # saving dataset
    df = df[conf.FEATURE_SET + ['gran truth']]  # ensure grantruth is the last column
    df.to_csv(conf.DATASET_PATH, index=False)
    # moving to next frame
    current = next
    skip_some(vs)
    loading += conf.TO_SKIP
    retval, next = vs.read()
