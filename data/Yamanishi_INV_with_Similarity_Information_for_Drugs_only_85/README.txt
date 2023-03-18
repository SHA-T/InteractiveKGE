This "Yamanishi_INV_with_Similarity_Information" directory differs from the regular Yamanishi data in the "Yamanishi" directory as follows:

- For each triple (h, r, t) in the train set an inverse triple (t, INVr, h) is added
- Similarity information (source: http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) between the DRUGS was added to the train set (train.txt). Similar drugs are connected through the newly introduced relation "sameAs", if they share a similarity score of at least 0.85 (85%)

To learn the semantics of the new relation "sameAs", the model can be pre-trained on the "pretrain.txt" mock data.