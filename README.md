tracking by reinforcement learning

# main idea

the main idea behind this project is to exploit detection to create the training-set for our network.

first part is linked to dataloader and feature extractor.
for each frame of my training video i used RES-net to extract the keypoints of a single person; starting
from that keynotes i saved some interesting distances and computed the grand truth checking the direction 
of the person in the next frame. of course you can edit the feature extractor or just use your own dataset;
just mind you have to fix the number of features and classes on the conf.py file.

second part is made by reinforcement.
i created a little nn using pytorch and used it as value function.
class dataloader loads data, agent class makes use of the nn to predict a batch of states coming from 
the dataloader. agent class can use the e-greedy or greedy policy.
finally the Q-learning formula is used to update the network.

# performances

accuracy on this model is around 63% due to the lack of time the training set has kinda few entries and 
the features could be extracted in a better way. addictonal improvements to boorst performances might be the
following:
- crop each bounding box during detection and save it as a thumbnail
- add a conv layer to the head of the network and feed it with the thumbnails
- make prediction according to the images rather than from the distances of keypoints.
- correct camera distortion to increase the accuracy of the grand truth.

_ it's IMPORTANT to have a belanced dataset in order to make this model works! this can be achieved by calling the following istruction on your pandas dataframe before loading it_

```python
g = df.groupby('gran truth') 
df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
```

check out if your dataset is balanced by using:

```python
print(df['gran truth'].value_counts())
```


# disclaim

this is a toy software, it can be used to implement more complex Q-learning algorithm and solve
harder tasks. it's plain to understand that tracking a single person is not a difficoult task nowadays.

i also uploaded an alternative version of this software which use just the nn with cross entropy instead of reinforcement learning.

# how to install
 
just install the dipendences by running:
```pip install -r requirements.txt```

after you took your own videos with a fixed camera, set up your video path in conf.py and run
```python dataloader.py``` 
to extract your feature and save them as .csv
to train the network run
```python reinforcement.py``` 
i left a weight file and a dataset to have an example of how it works.
