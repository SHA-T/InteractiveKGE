import pandas as pd

"""
This script extends the train set under the specified data sub-folder with inverse triples.
"""

DATASET_NAME = 'Yamanishi_INV'

if __name__ == '__main__':
    df = pd.read_csv(f"../data/{DATASET_NAME}/train.txt", sep='\t',
                     names=['head', 'relation', 'tail'])
    print(df)

    df2 = pd.DataFrame()
    print(df2)

    for index, row in df.iterrows():
        inv = pd.DataFrame({'head': row['tail'],
                            'relation': ''.join(['INV', row['relation']]),
                            'tail': row['head']}, index=[0])
        df2 = pd.concat([df2, inv]).reset_index(drop=True)

    print(df2)

    df = pd.concat([df, df2]).reset_index(drop=True)
    print(df)

    # After checking the output file is correct, make sure to rename train_INV.txt to
    # train.txt and replace the old train.txt
    df.to_csv(f"../data/{DATASET_NAME}/train_INV.txt", sep='\t', index=False, header=False)
