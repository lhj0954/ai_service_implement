import pandas as pd

df = pd.read_csv('../datasets/402_facebook.csv')

video = df[df['status_type'] == 'video']
video['positive_ratio'] = (video['num_likes'] + video['num_wows']) / video['num_reactions']


result = video[
    (video['positive_ratio'] > 0.4) &
    (video['positive_ratio'] < 0.5)
]

print(len(result))
