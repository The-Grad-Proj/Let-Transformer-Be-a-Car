import pandas as pd
import numpy as np

interpolated_csv = 'F:/Graduation_Project/comma-experiment/interpolated_comma.csv'
train_csv = 'F:/Graduation_Project/comma-experiment/train_split.csv'
val_csv = 'F:/Graduation_Project/comma-experiment/val_split.csv'

interpolated = pd.read_csv(interpolated_csv)
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

# iloc the trained rows
train = train.iloc[0:218740]
val = val.iloc[0:24305]

print('Interpolated:', len(interpolated))
print('Train:', len(train))
print('Val:', len(val))

# Delete rows in interpolated that are in train or val
print('Before:', len(interpolated))
interpolated = interpolated[~interpolated['filename'].isin(train['filename'])]
interpolated = interpolated[~interpolated['filename'].isin(val['filename'])]
print('After:', len(interpolated))

# Sort by timestamp to ensure continuity
interpolated = interpolated.sort_values(by='timestamp')

# Choose the first 24305 rows from interpolated
interpolated = interpolated.iloc[0:24305]

# Check for faulty samples
faulty_count = 0
valid_rows = []
for i in range(0, len(interpolated), 5):
    chunk = interpolated.iloc[i:i + 5]
    mean_timestamp = chunk['timestamp'].mean()
    median_timestamp = chunk['timestamp'].median()
    if abs(mean_timestamp - median_timestamp) > 5:
        faulty_count += 1
    else:
        valid_rows.append(chunk)

print(f'Number of faulty samples: {faulty_count}')

# Concatenate the valid rows back into a single DataFrame
shuffled_interpolated = pd.concat(valid_rows).reset_index(drop=True)

# Create the new DataFrame with the required columns
new_df = pd.DataFrame()
new_df['frame_id'] = shuffled_interpolated['filename'].str.extract(r'preprocessed_frames\\(.+)\.jpg')[0]
new_df['steering_angle'] = shuffled_interpolated['angle']

# Add the 'public' column with 50% 0s and 50% 1s
new_df['public'] = np.random.choice([0, 1], size=len(new_df))

# Save to new CSV
new_df.to_csv('F:/Graduation_Project/comma-experiment/test_split.csv', index=False)