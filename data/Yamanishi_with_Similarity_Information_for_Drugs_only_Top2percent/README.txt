This "Yamanishi_with_Similarity_Information" directory differs from the regular Yamanishi data in the "Yamanishi" directory as follows:

- Similarity information (source: http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) between the entities was added to the train set (train.txt). Similar drugs are connected through the newly introduced relation "sameAs", if they share a similarity score within the top 2%.

To learn the semantics of the new relation "sameAs", the model can be pre-trained on the "pretrain.txt" mock data.