import glob
import os
import pathlib
from random import randint

import numpy as np
import pandas as pd
import subprocess
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # load data
    df = pd.read_csv("data/countries_S1/train.txt", sep="\t", header=None)

    # run the KGE training in a seperate process
    cmd = ["python", "-u", "codes/run.py",
           "--do_train", "--do_valid", "--do_test",
           "--data_path", "data/countries_S1",
           "--model", "TransE",
           "--valid_steps", "50",
           "--save_checkpoint_steps", "1",
           "-n", "2", "-b", "8", "-d", "2",
           "-g", "2.0", "-a", "1.0", "-adv",
           "-lr", "0.1", "--max_steps", "250",
           "-save", "models/Task3_Countries_2Dim", "--test_batch_size", "8"]
    # subprocess.run(cmd)
    pid = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    # load latest embedding (.npy)
    count = 0
    history = []
    colors = []
    annotation_list = []
    init = False
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    # sc = ax.scatter([], [])
    ax.set_title("Neighbors of:\n united_states | spain | denmark")
    while pid.poll() is None:   # while subprocess is still alive
        embedding_history_paths = sorted(list(pathlib.Path('models/Task3_Countries_2Dim').glob('entity*.npy')))
        if len(embedding_history_paths) > count:
            emb = np.load(embedding_history_paths[-1], allow_pickle=True)

        if len(embedding_history_paths) == 1:
            headers = []
            dim_size = emb.shape[1]
            for x in range(dim_size):
                headers.append("d" + str(x))

            # fetching indices of used entities:
            tmp = pd.read_csv("data/countries_S1/entities.dict", sep="\t", header=None)
            tmp.columns = ["key", "value"]
            tmp_filtered = tmp["value"].isin(set(np.concatenate((df[0].values, df[2].values))))
            filtered_entities = tmp[tmp_filtered]
            indices_of_used_entities = filtered_entities['key'].values
            names_of_used_entities = filtered_entities['value'].values

            init = True

        if len(embedding_history_paths) > count and init:
            #
            transe_historical_embeddings = np.take(emb, indices_of_used_entities, axis=0)
            df_transe_historical_embeddings = pd.DataFrame(data=transe_historical_embeddings, index=names_of_used_entities, columns=headers)
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

            # update plots:

            # old plots but small
            ax.scatter(xCoor[0], yCoor[0], c=colors, s=15)

            xCoor[0] = history[0]["d0"].values
            yCoor[0] = history[0]["d1"].values

            # update x- and y-values
            sc.set_offsets(np.c_[xCoor[0], yCoor[0]])

            # annotations:
            # remove old annotations
            for i, a in enumerate(annotation_list):
                a.remove()
            annotation_list[:] = []
            # set new annotations
            for j, txt in enumerate(list(history[0].index)):
                annotations = ax.annotate(txt, (xCoor[0][j], yCoor[0][j]))
                annotation_list.append(annotations)

            fig.canvas.draw_idle()
            plt.pause(0.1)      # need some time to capture events like mouse interactions

            count = len(embedding_history_paths)

    plt.waitforbuttonpress()
