import pandas as pd

"""
This script creates an entities.dict file from the train, valid and test set under the specified data sub-folder.
"""

DATASET_NAME = 'Yamanishi_with_Similarity_Information_for_Drugs_only_60'

if __name__ == '__main__':
    df_train = pd.read_csv(f"../data/{DATASET_NAME}/train.txt", sep='\t',
                           names=['head', 'relation', 'tail'])
    df_valid = pd.read_csv(f"../data/{DATASET_NAME}/valid.txt", sep='\t',
                           names=['head', 'relation', 'tail'])
    df_test = pd.read_csv(f"../data/{DATASET_NAME}/test.txt", sep='\t',
                          names=['head', 'relation', 'tail'])

    print(df_train)

    df_all = pd.concat([df_train, df_valid, df_test])
    df_entities = pd.concat([df_all['head'].drop_duplicates(), df_all['tail'].drop_duplicates()],
                            ignore_index=True).drop_duplicates()

    df_entities = df_entities.reset_index(drop=True)

    df_entities.to_csv(f"../data/{DATASET_NAME}/entities.dict", sep="\t", index=True,
                       header=False)
