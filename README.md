# Video-Action-Transformer-Network-Pytorch-
Pytorch and Tensorflow Implementation of the paper Video Action Transformer Network   
*Rohit Girdhar, Joao Carreira, Carl Doersch, Andrew Zisserman*


Retasked Video transformer (uses resnet as base) 
**transformer_v1.py** is more like real transformer, **transformer.py** more true to what paper advertises
Usage : 
```
from transformer_v1 import Semi_Transformer
model = Semi_Transformer(num_classes=num_classes , num_frames = max_seq_len)
outputs, features = model(imgs) # outputs is the classification layer output (do cross entropy loss)
                                #features are used as video embedding
                                
##################### or ###################
from transformer_v2 import Semi_Transformer
model = Semi_Transformer(num_classes=625 , seq_len = max_seq_len)
```


In case you find any discrepency, please raise an issue. If any one was able to reproduce the paper results kindly help me with this [issue](https://github.com/ppriyank/Video-Action-Transformer-Network-Pytorch-/issues/2). If possible please meantion the changes needs to be further added. 

