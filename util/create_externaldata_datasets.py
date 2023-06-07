"""
This module creates new datasets by integrating external data from the data_external/ directory (SOURCE PATH).
This assumes, that the external data was created by having been queried out of a knowledge graph such that it will
complement the data in the TARGET_PATH.
The dictionary files entities.dict and relations.dict in the new directory will be updated accordingly.
"""

from distutils.dir_util import copy_tree
import os
import pandas as pd


EXTERNAL_DATA_TYPE = "side_effects"
SOURCE_PATH = "../data_external/Drug_SideEffects/OnlyYamanishiDrugs_SideEffects"
TARGET_PATH = "../data_k_fold/yamanishi"
SUB_DATASET_NAMES = ['enzyme', 'ion_channel', 'nuclear_receptor', 'gpcr', 'whole_yamanishi']
NUMBER_OF_FOLDS = 5
UPSCALING_FACTOR = 10


if __name__ == '__main__':

    for sub_dataset in SUB_DATASET_NAMES:

        # Load external data as DataFrame
        df_ext = pd.read_csv(f'{SOURCE_PATH}/{sub_dataset}_{EXTERNAL_DATA_TYPE}.csv')
        relations_ext = df_ext['relation'].drop_duplicates()
        entities_ext = df_ext['tail'].drop_duplicates()
        # print(relations_ext)
        # print(entities_ext)
        # print(df_ext)

        # Create new directory for Yamanishi+ExternalData
        new_dir = f'{TARGET_PATH}/{sub_dataset}/with_{EXTERNAL_DATA_TYPE}'
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        # Copy files over from original
        copy_tree(f'{TARGET_PATH}/{sub_dataset}/original', new_dir)

        for fold in range(1, NUMBER_OF_FOLDS + 1):

            # Update relations.dict
            relations_int = pd.read_csv(f'{TARGET_PATH}/{sub_dataset}/original/{fold}/relations.dict',
                                        delimiter='\t', header=None)[1]
            relations_int = pd.concat([relations_int, relations_ext])
            relations_int = relations_int.reset_index(drop=True)
            relations_int.to_csv(f"{new_dir}/{fold}/relations.dict", sep="\t", index=True, header=False)

            # Update entities.dict
            entities_int = pd.read_csv(f'{TARGET_PATH}/{sub_dataset}/original/{fold}/entities.dict',
                                       delimiter='\t', header=None)[1]
            entities_int = pd.concat([entities_int, entities_ext])
            entities_int = entities_int.reset_index(drop=True)
            entities_int.to_csv(f"{new_dir}/{fold}/entities.dict", sep="\t", index=True, header=False)

            # Update train.txt
            df_int = pd.read_csv(f'{TARGET_PATH}/{sub_dataset}/original/{fold}/train.txt',
                                 delimiter='\t', names=['head', 'relation', 'tail'])
            if UPSCALING_FACTOR > 1:
                df_int = pd.concat([df_int] * int(UPSCALING_FACTOR))
            df_int = pd.concat([df_int, df_ext])
            df_int = df_int.reset_index(drop=True)
            df_int.to_csv(f"{new_dir}/{fold}/train.txt", sep="\t", index=False, header=False)
            print(df_int)
