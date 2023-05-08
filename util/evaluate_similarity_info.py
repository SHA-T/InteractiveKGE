"""
This script can be used to evaluate how useful the injected similarity information is.
This script displays entities (present in the similarity triples) that have missing links to neighbors of a similar
entity in the train set, which (the missing links) are present in the test set. Thus, these missing links can be found
easily with the injection of the similarity data.
"""


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json


TARGET_PATH = "../data_k_fold/yamanishi"
SUB_DATASET_NAME = 'enzyme'                 # enzyme | gpcr | ion_channel | nuclear_receptor | whole_yamanishi
FOLD = 1                                    # 1 | 2 | ... | k
TOP_X_PERCENT = 0.1


def set_default(obj):
    """
    Returns a list version of obj, if obj is a set.
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':

    base_path = f"{TARGET_PATH}/{SUB_DATASET_NAME}"

    # DataFrame Representation
    df_train_ori = pd.read_csv(f"{base_path}/original/{FOLD}/train.txt",
                               sep='\t', names=['head', 'relation', 'tail'])
    df_train_sim = pd.read_csv(f"{base_path}/with_similarity_information_top{str(TOP_X_PERCENT)}pct/{FOLD}/train.txt",
                               sep='\t', names=['head', 'relation', 'tail'])
    df_test = pd.read_csv(f"{base_path}/original/{FOLD}/test.txt",
                          sep='\t', names=['head', 'relation', 'tail'])
    df_sim_only = pd.concat([df_train_ori, df_train_sim]).drop_duplicates(keep=False)

    # Graph Representation
    gr_train = nx.from_pandas_edgelist(df_train_ori, 'head', 'tail')
    gr_test = nx.from_pandas_edgelist(df_test, 'head', 'tail')

    ml = {}

    for index, row in df_sim_only.iterrows():
        # Direct neighbors of head & tail in train & test dataset
        neighbors_head_train = set([n for n in gr_train.neighbors(row['head'])])
        neighbors_tail_train = set([n for n in gr_train.neighbors(row['tail'])])
        try:
            neighbors_head_test = set([n for n in gr_test.neighbors(row['head'])])
        except nx.exception.NetworkXError:
            neighbors_head_test = set()
        try:
            neighbors_tail_test = set([n for n in gr_test.neighbors(row['tail'])])
        except nx.exception.NetworkXError:
            neighbors_tail_test = set()

        # Missing links to neighbors of a similar entity in train set, that are present in test set
        ml_head = neighbors_head_test.intersection(neighbors_tail_train).difference(neighbors_head_train)
        ml_tail = neighbors_tail_test.intersection(neighbors_head_train).difference(neighbors_tail_train)

        # Add those missing links to the dictionary
        if row['head'] in ml:
            ml[row['head']].update(ml_head)
        else:
            ml[row['head']] = ml_head
        if row['tail'] in ml:
            ml[row['tail']].update(ml_tail)
        else:
            ml[row['tail']] = ml_tail

    # Delete empty keys and pretty print
    ml = {k: v for k, v in ml.items() if v}
    print("\nDictionary of entities (present in the similarity triples) that have missing links to neighbors")
    print("of a similar entity in the train set, which (the missing links) are present in the test set.")
    print("Thus, these are missing links that can be found easily with the injection of the similarity data:")
    print(f"\n{SUB_DATASET_NAME}, threshold for injected similarity data: {TOP_X_PERCENT}%, Fold {FOLD}:\n")
    print(json.dumps(ml, indent=4, default=set_default))

    # plt.figure(figsize=(10, 8))
    # nx.draw(gr_test, with_labels=True)
    # plt.show()
