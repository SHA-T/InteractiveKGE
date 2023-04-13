# About The Data

The **sub-folders** contain subsets of the Yamanishi data¹ aswell as a merged dataset from these subsets:
- enzyme
- gpcr
- ion_channel
- nuclear_receptor
- whole_yamanishi

Each subset contains connections between drugs and a targets/proteins of a certain type: 
enzyme, gpcr, ion channel or nuclear receptor, respectively.

Each sub-folder contains **sub-sub-folders** of the following structure:
- original
- with_similarity_information_top`x`pct

The "original"-set contains $k$ folds of the original data. 

The "with_similarity_information_top`x`pct"-set contains the same $k$ folds, but with added similarity information to 
the train set (train.txt). The similarity information is picked from the top $x$ percent entries in the similarity 
matrices¹ with the highest values. The two entities belonging to that entry are viewed as similar and therefore are 
connected through the relation "sameAs". To learn the semantics of the new relation "sameAs", the model can be 
pre-trained on the "pretrain.txt" mock data. Thus, each fold (1, 2, ..., k) in the 
"with_similarity_information_top`x`pct"-set differs from the respective fold in the "original"-set by the following 
four files:
- pretrain.txt (Added. Contains mock data for pretraining semantics of new relations.)
- train.txt (Changed. Triples, representing the similarity information, are added.)
- entities.dict (Changed. New entities, present in the mock data, are added.)
- relations.dict (Changed. New relations, present in the mock data, are added.)

The "valid.txt" and "test.txt" remain unchanged.

---

¹ Yamanishi data, contains drug-target interactions and similarity information between drugs and targets separately, 
source http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/