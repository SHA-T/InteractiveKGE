# Interactive Knowledge Graph Embedding

This extends the git repo [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) to make it interactive by creating a live embedding visualization that allows for adding new entities during training.
For more information on how the actual embedding works click the link.

## 1. Changes to KnowledgeGraphEmbedding Repo

Some changes were made to the existing code of [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). These are fixing seeds and modifying some code parts such that it is possible to store embeddings during the training. 
The exact code changes are documented in `scripts.ipynb`.

Additionally to that the module `interact.py` was created for the purpose of this extended repo to visualize embeddings and add new entities - both live during training.

## 2. How it works

The Interactive KGE can be executed on any of the datasets in the `data/` directory. 
To choose the dataset, pass the name of a sub-folder under the `data/` directory as the first command line argument.

```
python interact.py countries_neighb_UsaSpaDen
```

If you do not pass any argument, by default the `countries_neighb_UsaSpaDen` dataset will be chosen. 
This dataset is a simplified version of the `countries_S1` dataset, that is filtered for the countries USA, Spain and Denmark and all of their neighbors.

**Features implemented**: 
- Running the Knowledge Graph Embedding in a sub process
- Tracking and Plotting the embeddings in the main process

**Features in progress**: 
- By clicking into the embedding space, you pause the training and create a new entity at the clicked coordinates. Choose the relations of the new entity in the dropdown menu.
- Change training hyperparameters for the Knowledge Graph Embedding (Currently they can be changed by modifying the `interact.py` module)