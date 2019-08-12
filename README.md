# Video-Action-Transformer-Network-Pytorch-
Pytorch Implementation of the paper Video Action Transformer Network   
*Rohit Girdhar, Joao Carreira, Carl Doersch, Andrew Zisserman*


Retasked Video transformer (uses resnet as base) 
**transformer_v1.py** is more like real transoformer, **transformer.py** more true to what paper advertises
Usage : 
```
from transformer_v1 import Semi_Transformer
model = Semi_Transformer(num_classes=num_classes , num_frames = max_seq_len)
outputs, features = model(imgs) # outputs is the classification layer output (do cross entropy loss)
                                #features are used as video embedding
```


In case you find any discrepency, please raise an issue
