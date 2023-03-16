# Interactive Knowledge Graph Embedding

This extends the git repo [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) to make it interactive by creating a live embedding visualization that allows for adding new entities during training.
For more information on how the actual embedding works click the link.

## 1. Changes to KnowledgeGraphEmbedding Repo

Some changes were made to the existing code of [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). These are fixing seeds and modifying some code parts such that it is possible to store embeddings during the training. 
The exact code changes are documented in `scripts.ipynb`.

Additionally to that the module `interact.py` was created for the purpose of this extended repo to visualize embeddings and add new entities - both live during training.

Also, the `run.py` module was changed to allow for a new optional feature: **Pretraining**.
Pretraining allows for the training of the semantics of new relations with the help of a mock dataset that should be saved as _pretrain.txt_ under the corresponding data directory. 
An example for such a file can be looked up in the _data/Yamanishi_with_Similarity_Information_ directory. 
While the regular training phase uses the data in _train.txt_, pretraining precedes regular training and only uses the data in _pretrain.txt_. 
Pretraining lasts only a small number of epochs before regular training starts. 
After Pretraining is completed the embeddings of the new relations will be freezed, so that the learned semantics don't get corrupted. 
Currently it is assumed that all relations in the _relations.dict_ file, except the first listed relation, are new relations (if pretraining is activated).

## 2. How To Use

For detailed information on how to use the `run.py` for KGE training without visualization refer to [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). You don't have to use the [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) repo, since this repo (InteractiveKGE) provides the same features and even more.
To use Pretraining set the `--do_pretrain` argument.

```
python -u codes/run.py 
    --do_pretrain 
    --do_train 
    --do_valid 
    --do_test 
    --data_path data/countries_S1 
    --model TransE 
    ...
    ...
    ...
```

Like training without visualization, training with visualization can be executed on any of the datasets in the _data/_ directory. 
To choose the dataset, pass the name of a sub-folder under the _data/_ directory as the first command line argument.

```
python interact.py countries_neighb_UsaSpaDen
```

If you do not pass any argument, by default the `countries_neighb_UsaSpaDen` dataset will be chosen. 
This dataset is a simplified version of the `countries_S1` dataset, that is filtered for the countries USA, Spain and Denmark and all of their neighbors.
You can add your own datasets and run the Interactive KGE on them. Therefore, add your datasets to the `data/` directory, 
but keep the structure and format same as the other datasets.

**Implemented** Features:
- [x] Free dataset selection
- [x] Running the Knowledge Graph Embedding in a sub process
- [x] Tracking and Plotting the embeddings in the main process
- [x] Pretraining (only accessible if training without visualization)

**In Progress** Features:
- [ ] By clicking into the embedding space, you pause the training and create a new entity at the clicked coordinates. Choose the relations of the new entity in the dropdown menu.
- [ ] Change training hyperparameters for the Knowledge Graph Embedding (Currently they can be changed by modifying the `interact.py` module)