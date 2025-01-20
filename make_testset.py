import os
import pandas as pd
import numpy as np

interpolated_csv = 'path/to/interpolated.csv'
train_csv = 'path/to/train.csv'
val_csv = 'path/to/val.csv'

interpolated = pd.read_csv(interpolated_csv)
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

# Delete rows in interpolated that are in train or val