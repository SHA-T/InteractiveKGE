# Interactive Knowledge Graph Embedding

This extends the git repo [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) to make it interactive by creating a live embedding visualization that allows for adding new entities during training. For more information on how the actual embedding works click the link.

Running `python interact.py` will run the Knowledge Graph Embedding in a separate subprocess. In the main process the embeddings will be tracked and plotted.

In progress: By clicking into the embedding space, you pause the training and create a new entity at the clicked coordinates. Choose the relations of the new entity in the dropdown menu.
