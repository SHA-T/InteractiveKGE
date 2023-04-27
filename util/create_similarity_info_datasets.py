"""
This module looks at the Similarity Matrices for the Yamanishi datasets in the 'data_external/' directory, filters
for the top x% entries, converts those to triples and adds them to the original Yamanishi drug-target interaction
train sets in the 'data/' or 'data_k_fold/' directory. If you want to add pretrain mock data to the new directories,
you have to do this manually by adding a pretrain.txt mock dataset and add the new relations and mock entities to the
relations.dict and entities.dict file.
"""


from distutils.dir_util import copy_tree
import os
import numpy as np
import pandas as pd


TARGET_PATH = "../data_k_fold/yamanishi"
SUB_DATASET_NAME = 'whole_yamanishi'               # enzyme | gpcr | ion_channel | nuclear_receptor | whole_yamanishi
NUMBER_OF_FOLDS = 5
TOP_X_PERCENT = [0.05, 0.1, 0.2, 0.5, 1, 2]


def calc_thresh(matrix: pd.DataFrame, top_x_percent: float):
    """
    Calculates and returns threshold for similar entities.

    :param matrix:
        Similarity score matrix of entities.
    :param top_x_percent:
        The top X percent of matrix values that we choose to mark similar entities.
    :return:
        A threshold value in the matrix that a given percent of all other values in the matrix are higher than.
    """
    if top_x_percent <= 0 or top_x_percent > 100:
        raise ValueError("Please, provide a number between 0 and 100 for top_x_percent.")

    a = matrix.to_numpy()
    a[a >= 1] = None    # Remove '1's to avoid cases where we only get values on the main diagonal

    percentile = 100 - top_x_percent
    thresh = np.nanpercentile(a, percentile)

    return thresh


def get_similar_triples(matrix: pd.DataFrame, thresh: float):
    """
    Returns all entity pairs from the matrix, that have a similarity score higher than thresh.

    :param matrix:
        Similarity score matrix of entities.
    :param thresh:
        Similarity score, above which an entity pair is viewed as similar.
    :return:
    """
    a = matrix
    # Create list of tuples containing entity pairs with similarity score > thresh
    lst = a[a > thresh].stack().index.tolist()
    # Remove tuples, where both elements are the same
    lst = [item for item in lst if item[0] != item[1]]

    return lst


if __name__ == "__main__":
    # Uncomment this ONLY in your first ever use to
    # fix the issue, where all protein names in the original Yamanishi Similarity Matrices are missing the ':'
    # fin1 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/e_simmat_dg.txt", "rt")
    # fin2 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/gpcr_simmat_dg.txt", "rt")
    # fin3 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/ic_simmat_dg.txt", "rt")
    # fin4 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/nr_simmat_dg.txt", "rt")
    # data1 = fin1.read()
    # data2 = fin2.read()
    # data3 = fin3.read()
    # data4 = fin4.read()
    # data1 = data1.replace('hsa', 'hsa:')
    # data2 = data2.replace('hsa', 'hsa:')
    # data3 = data3.replace('hsa', 'hsa:')
    # data4 = data4.replace('hsa', 'hsa:')
    # fin1.close()
    # fin2.close()
    # fin3.close()
    # fin4.close()
    # fin1 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/e_simmat_dg.txt", "wt")
    # fin2 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/gpcr_simmat_dg.txt", "wt")
    # fin3 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/ic_simmat_dg.txt", "wt")
    # fin4 = open("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/nr_simmat_dg.txt", "wt")
    # fin1.write(data1)
    # fin2.write(data2)
    # fin3.write(data3)
    # fin4.write(data4)
    # fin1.close()
    # fin2.close()
    # fin3.close()
    # fin4.close()

    # Load all similarity matrices
    dfs = []
    if SUB_DATASET_NAME == 'whole_yamanishi' or SUB_DATASET_NAME == 'enzyme':
        df1 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_CompoundStructure/e_simmat_dc.txt",
                          delimiter='\t', index_col=0)
        df2 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/e_simmat_dg.txt",
                          delimiter='\t', index_col=0)
        dfs.extend([df1, df2])
    if SUB_DATASET_NAME == 'whole_yamanishi' or SUB_DATASET_NAME == 'gpcr':
        df3 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_CompoundStructure/gpcr_simmat_dc.txt",
                          delimiter='\t', index_col=0)
        df4 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/gpcr_simmat_dg.txt",
                          delimiter='\t', index_col=0)
        dfs.extend([df3, df4])
    if SUB_DATASET_NAME == 'whole_yamanishi' or SUB_DATASET_NAME == 'ion_channel':
        df5 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_CompoundStructure/ic_simmat_dc.txt",
                          delimiter='\t', index_col=0)
        df6 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/ic_simmat_dg.txt",
                          delimiter='\t', index_col=0)
        dfs.extend([df5, df6])
    if SUB_DATASET_NAME == 'whole_yamanishi' or SUB_DATASET_NAME == 'nuclear_receptor':
        df7 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_CompoundStructure/nr_simmat_dc.txt",
                          delimiter='\t', index_col=0)
        df8 = pd.read_csv("../data_external/Yamanishi_SimilarityMatrix_ProteinSequence/nr_simmat_dg.txt",
                          delimiter='\t', index_col=0)
        dfs.extend([df7, df8])
    print("Number of relevant similarity matrices:", len(dfs))

    for top_x_percent in TOP_X_PERCENT:

        # Combined list of all similar entity pairs
        entity_pairs = []
        for df in dfs:
            thresh = calc_thresh(df, top_x_percent)
            entity_pairs.extend(get_similar_triples(df, thresh))
        print("\nCombined list of all similar entity pairs:\n", entity_pairs, "\nCount:", len(entity_pairs))

        # Convert list of tuples to DataFrame and insert relation
        df = pd.DataFrame.from_records(entity_pairs, columns=['head', 'tail'])
        df.insert(loc=1, column='relation', value="sameAs")
        print("\nDataframe of all similar entity pairs:\n", df)

        # Remove all rows that contain entities that are not present in the Yamanishi train set (Use entities.dict file)
        train_set_entities = pd.read_csv(f"{TARGET_PATH}/{SUB_DATASET_NAME}/original/1/entities.dict",
                                         sep='\t', header=None, names=['idx', 'entity'], index_col=0)
        train_set_entities = train_set_entities.entity.tolist()
        df = df[df['head'].isin(train_set_entities) & df['tail'].isin(train_set_entities)]
        df.reset_index(drop=True, inplace=True)
        print(df)

        # Create directory
        target_path = f"{TARGET_PATH}/{SUB_DATASET_NAME}/with_similarity_information_top{str(top_x_percent)}pct"
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        # Copy files over from original
        copy_tree(f"{TARGET_PATH}/{SUB_DATASET_NAME}/original", target_path)
        # Append (mode='a') similarity triples dataframe to train.txt for all folds
        for fold in range(1, NUMBER_OF_FOLDS + 1):
            df.to_csv(f"{target_path}/{fold}/train.txt",
                      mode='a', sep='\t', header=False, index=False)
