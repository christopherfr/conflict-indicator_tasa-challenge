import pandas as pd
import re
import numpy as np

def get_features():
    # Read the data
    df = pd.read_csv('Data/speakers.txt', header = None)
    # Set the regex pattern
    pattern = r"(.+)'s CI (.+) in (\d+) when sitting next to (.+)\."
    # Initialize the DataFrame
    df_features = pd.DataFrame({'a':[], 'b':[], 'ci':[]})
    # Extract the features row by row
    for index, row in df.iterrows():
        groups = re.findall(pattern, row[0])[0]

        speaker_A = groups[0]
        speaker_B = groups[3]
        ci = int(groups[2])
        if (groups[1] == 'decreases'):
            ci = ci*-1

        row_features = pd.DataFrame({'a':[speaker_A], 'b':[speaker_B], 'ci':[ci]})
        df_features = df_features.append(row_features, ignore_index = True)

    # Set the data types of the columns
    df_features['a'] = df_features['a'].astype('category')
    df_features['b'] = df_features['b'].astype('category')
    df_features['ci'] = df_features['ci'].astype(np.int16)
    return df_features

def main():
    df = get_features()
    filename = 'features.csv'
    df.to_csv(filename, index = False)

if __name__ == '__main__':
    main()