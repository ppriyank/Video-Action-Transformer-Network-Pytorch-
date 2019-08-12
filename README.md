# Video-Action-Transformer-Network-Pytorch-
Pytorch Implementation of the paper Video Action Transformer Network   
*Rohit Girdhar, Joao Carreira, Carl Doersch, Andrew Zisserman*


Retasked Video transformer (uses resnet as base)
Usage : 
```
from transformer_v1 import Semi_Transformer
model = Semi_Transformer(num_classes=num_classes , num_frames = max_seq_len)
outputs, features = model(imgs) # outputs is the classification layer output (do cross entropy loss)
                                #features are used as video embedding
```
