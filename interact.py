import glob
import sys
import os
import shutil
import pathlib
import time
from random import randint

import numpy as np
import pandas as pd
import subprocess
from matplotlib import pyplot as plt


DEFAULT_DATASET_NAME = "countries_neighb_UsaSpaDen"


def main(dataset_name):

    # remove old embeddings
    if os.path.exists(f"models/{dataset_name}_2Dim"):
        shutil.rmtree(f"models/{dataset_name}_2Dim")

    # load data
    df = pd.read_csv(f"data/{dataset_name}/train.txt", sep="\t", header=None)

    # run the KGE training in a separate subprocess
    cmd = ["python", "-u", "codes/run.py",
           "--do_train", "--do_valid", "--do_test",
           "--data_path", "data/countries_S1",
           "--model", "TransE",
           "--valid_steps", "100",
           "--save_checkpoint_steps", "10",
           "-n", "2", "-b", "8", "-d", "2",
           "-g", "2.0", "-a", "1.0", "-adv",
           "-lr", "0.02", "--max_steps", "1000",
           "-save", f"models/{dataset_name}_2Dim", "--test_batch_size", "8"]
    # subprocess.run(cmd)
    pid = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    time.sleep(0.5)

    # main process (track embeddings, plot, interact)
    count = 0
    history = []  # currently, not the whole embedding history will be saved, only the embeddings of the last epoch
    colors = []
    annotation_list = []
    init = False
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(dataset_name)
    while pid.poll() is None:  # while subprocess is still alive

        # locate embedding files and then load latest embedding (.npy)
        embedding_history_paths = sorted(list(pathlib.Path(f"models/{dataset_name}_2Dim").glob("entity*.npy")))
        if len(embedding_history_paths) > count:
            emb = np.load(embedding_history_paths[-1], allow_pickle=True)

        if len(embedding_history_paths) == 1:
            # set header for each dimension
            headers = []
            dim_size = emb.shape[1]
            for x in range(dim_size):
                headers.append("d" + str(x))

            # fetching indices of used entities:
            tmp = pd.read_csv(f"data/{dataset_name}/entities.dict", sep="\t", header=None)
            tmp.columns = ["key", "value"]
            tmp_filtered = tmp["value"].isin(set(np.concatenate((df[0].values, df[2].values))))
            filtered_entities = tmp[tmp_filtered]
            indices_of_used_entities = filtered_entities['key'].values
            names_of_used_entities = filtered_entities['value'].values

            init = True

        if len(embedding_history_paths) > count and init:
            #
            transe_historical_embeddings = np.take(emb, indices_of_used_entities, axis=0)
            df_transe_historical_embeddings = pd.DataFrame(data=transe_historical_embeddings,
                                                           index=names_of_used_entities, columns=headers)
            if history:
                history[0] = df_transe_historical_embeddings
            else:
                history.append(df_transe_historical_embeddings)

                # create colors for plot
                for i in range(len(history[0].index)):
                    colors.append('#%06X' % randint(0, 0xFFFFFF))

                xCoor = []
                yCoor = []
                xCoor.append(history[0]["d0"].values)
                yCoor.append(history[0]["d1"].values)

                # scatter initialization
                sc = ax.scatter(xCoor[0], yCoor[0], c=colors, s=150)

            """update plots"""
            # old plots but small
            ax.scatter(xCoor[0], yCoor[0], c=colors, s=15)

            xCoor[0] = history[0]["d0"].values
            yCoor[0] = history[0]["d1"].values

            # update x- and y-values
            sc.set_offsets(np.c_[xCoor[0], yCoor[0]])
            """"""

            """annotations"""
            # remove old annotations
            for i, a in enumerate(annotation_list):
                a.remove()
            annotation_list[:] = []
            # set new annotations
            for j, txt in enumerate(list(history[0].index)):
                annotations = ax.annotate(txt, (xCoor[0][j], yCoor[0][j]))
                annotation_list.append(annotations)
            """"""

            fig.canvas.draw_idle()
            plt.pause(1)  # need some time to capture events like mouse interactions

            count = len(embedding_history_paths)

    plt.waitforbuttonpress()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main(DEFAULT_DATASET_NAME)
    elif os.path.exists(f"data/{sys.argv[1]}"):
        main(sys.argv[1])
    else:
        print("As the first argument, please provide the name of a folder under the 'data' directory (=dataset name).")