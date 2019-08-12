import keras
import tensorflow as tf 
import numpy as np 


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)  
  sines = np.sin(angle_rads[:, 0::2])
  cosines = np.cos(angle_rads[:, 1::2])
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  pos_encoding = pos_encoding[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq,seq_length): 
	return tf.cast(tf.sequence_mask(seq.shape[0], maxlen=seq_length),  dtype = tf.int32 ) 


def create_look_ahead_mask(size):
  mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask=None):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.sqrt(dk)
  if mask is not None:
  	#mask is the  relevant values from k to  keep , [q.size  , k .size]
    scaled_attention_logits += tf.log(tf.to_float(mask)) 
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0    
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)        
  
  def split_heads(self, x):
    x = tf.reshape(x, (-1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[1, 0, 2])
  
  def call(self, v, k, q, mask):
    q = self.wq(q)  # (seq_len, d_model)
    k = self.wk(k)  # (seq_len, d_model)
    v = self.wv(v)  # (seq_len, d_model)    
    q = self.split_heads(q)  # (num_heads, seq_len_q, depth)
    k = self.split_heads(k)  # (num_heads, seq_len_k, depth)
    v = self.split_heads(v)  # (num_heads, seq_len_v, depth)
    # scaled_attention.shape == ( num_heads, seq_len_v, depth)
    # attention_weights.shape == ( num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[1, 0, 2])  #  (seq_len_q, num_heads, depth)
    concat_attention = tf.reshape(scaled_attention, (-1, self.d_model))  # (seq_len_q, d_model)
    output = self.dense(concat_attention)  # (seq_len_q, d_model)       
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (seq_len, dff)
      tf.keras.layers.Dense(d_model)  # ( seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.dropout1 = tf.nn.dropout(input_ , 1-rate)
    self.dropout2 = tf.nn.dropout(input_ , 1-rate)
    
  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask)  # ( input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1= tf.contrib.layers.layer_norm(x + attn_output)
    ffn_output = self.ffn(out1)  # ( input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = tf.contrib.layers.layer_norm(out1 + ffn_output)
    return out2



class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.dropout1 = tf.nn.dropout(input_ , 1-rate)
    self.dropout2 = tf.nn.dropout(input_ , 1-rate)
    self.dropout3 = tf.nn.dropout(input_ , 1-rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # ( target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = tf.contrib.layers.layer_norm(x + attn1)
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # ( target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = tf.contrib.layers.layer_norm(attn2 + out1)
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = tf.contrib.layers.layer_norm(ffn_output + out2)
    return out3, attn_weights_block1, attn_weights_block2



class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    self.pos_encoding = tf.reshape(pos_encoding, [-1, self.d_model])
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.nn.dropout(input_ , 1-rate)
  
  def call(self, x, training, mask):
    seq_len = tf.shape(x)[0]
    #sess.run(seq_len)
    # adding embedding and position encoding.
    x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[ :seq_len, :]
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    return x  # ( input_seq_len, d_model)



class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
               rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    pos_encoding = positional_encoding(target_vocab_size, self.d_model)
    self.pos_encoding = tf.reshape(pos_encoding, [-1, self.d_model])
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.nn.dropout(input_ , 1-rate)
    
  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[0]
    attention_weights = {}
    x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[ :seq_len, :]
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    # x.shape == ( target_seq_len, d_model)
    return x, attention_weights





class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)
    self.final_layer = tf.keras.layers.Dense(d_model)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_model)
    return final_output, attention_weights
