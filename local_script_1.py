import numpy as np
import pandas as pd

if __name__ == "__main__":
    df_train = pd.read_csv("./data/Yamanishi_with_Similarity_Information/train.txt", sep="\t",
                           names=['head', 'relation', 'tail'])
    df_valid = pd.read_csv("./data/Yamanishi_with_Similarity_Information/valid.txt", sep="\t",
                           names=['head', 'relation', 'tail'])
    df_test = pd.read_csv("./data/Yamanishi_with_Similarity_Information/test.txt", sep="\t",
                           names=['head', 'relation', 'tail'])

    print(df_train)

    df_all = pd.concat([df_train, df_valid, df_test])
    df_entities = pd.concat([df_all['head'].drop_duplicates(), df_all['tail'].drop_duplicates()],
                            ignore_index=True).drop_duplicates()
    print(df_entities)

    df_entities = df_entities.reset_index(drop=True)

    df_entities.to_csv('./data/Yamanishi_with_Similarity_Information/entities.dict', sep="\t", index=True,
                       header=False)
