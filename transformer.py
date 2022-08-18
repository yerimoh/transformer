import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        # 1000^(2i/d_model)
        ## pow -> 제곱값 계산
        ## tf.cast -> 부동소수점형에서 정수형으로 바꾼 경우 소수점 버림을 한다.
        angles = 1 / tf.pow(10000, (2*(i // 2)) / tf.cast(d_model, tf.float32)) 
        # pos/1000^(2i/d_model)
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angles_rads = self.get_angles(position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i = tf.range(d_model, 
                                                  dtype=tf.float32)[tf.newaxis, :],
                                     d_model = d_model)
        
        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angles_rads[:, 0::2])
        
        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angles_rads[:, 1::2])
        
        angles_rads = np.zeros(angles_rads.shape)
        angles_rads[:, 0::2] = sines
        angles_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angles_rads)
        
        
        # print(pos_encoding) # 변경 전
        pos_encoding = pos_encoding[tf.newaxis, ...] # tensor에 하나의 차원이 추가됨
        # print('\n', pos_encoding, tf.float32) # 변경 후
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
  
  
# Scaeld-Dot Product Attention
# 그냥 QK가 아니라 rootdk를 사용했기 떄문에 scaled라고 붙임
def scaled_dot_product_attention(query, key, value, mask):
    # query 크기: (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기: (batct_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기: (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask: (batch_size, 1, 1, key의 문장 길이)
    
    # QK 
    # [2단계] 점수 계산, Attention score 행렬
    # 아 단어와 입력 문장 속 다른 단어들에 얼마나 집중을 해야할 지 결정(얼마나 연관되는지를 보여줌)
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # QK/rootdk
    # [3단계] scaling 
    # dk의 루트 값으로 나눔
    # 두 벡터의 내적값을 스케일링하는 값으로 루트를 씌움 -> 더 안정적인 값이 됨
    # K 벡터의 차원을 나타내는 dk 에 루트를 씌움
    # 논문에선 dk = dmodel/num-head = 64이므로 root씌우면 8임
    depth = tf.cast(tf.shape(key)[-1], tf.float32) # K 벡터의 차원구하기 
    logis = matmul_qk / tf.math.sqrt(depth)
    
    # [+ 단계] 마스킹 
    # decoder에서 쓰임 따로 만들기 귀찮아서 여기다가 추가~
    # Attention score matrix의 마스킹 할 위치에 매우 작은 음수값을 넣음
    # 매우 작은 값이므로 softmax 함수를 지나면 행렬의 해당 위치 값은 0이 됨
    if mask is not None:
        logis += (mask * -1e9)
        
    # softmax(QK/rootdk)
    # [4단계] softmax 
    # 이 함수는 마지막 차원인 key의 문장 길이 방향으로 수행됨
    # attention weight: (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logis, axis=-1)
    
    # softmax(QK/rootdk) *  V
    # [5단계] output
    # output: (batch_size, num_heads, query의 문장 길이, d_model / num_heads)
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights
  
from tensorflow.keras.layers import Dense, Layer
class MultiHeadAttention(Layer):
    
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        
        # assert는 뒤의 조건이 True가 아니면 AssertError를 발생
        assert d_model % self.num_heads == 0
        
        # d_model을 num_heads 로 나눈 값.
        # 논문 기준: 64
        self.depth = d_model // self.num_heads
        
        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)
        
        # W0에 해당하는 밀집층 정의
        self.dense = Dense(units=d_model)
        
    
    # num_heads 개수만큼 q, k, v를 split 하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q: (batch_size, query의 문장 길이, d_model)
        # k: (batch_size, key의 문장 길이, d_model)
        # v: (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있음
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 3. scaled-dot product attention
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # 4. 헤드 연결(concatenate) 하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # 5. W0에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)
        
        return outputs
      
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :] 
  
from tensorflow.keras.layers import Dropout, LayerNormalization
from tensorflow.keras.layers import Embedding

def encoder_layer(dff, d_model, num_heads, dropout, name='encoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    
    # 인코더는 패딩 마스크 사용
    ## 인코더의 입력으로 들어가는 문장에는 Padding이 있을 수 있으므로, 
    ## Attention 시 패딩 토큰을 제외하도록 패딩 마스크를 사용함(padding은 decoder에만 있음)
    ## 이는 Multi-Head Attention함수의 mask 인자값으로 padding_mask가 들어가는 이유
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    # Multi-head Attention (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttention(d_model, num_heads, name='attention')({
        'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask 
    }) # 패딩 마스크 사용, Q = K = V
    
    # Dropout + Add & Norm
    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Position-wise FFNN (두번째 서브층)
    outputs = Dense(units=dff, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    
    # Dropout + Add & Norm
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)
    
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
  
  
# 쌓기
def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='encoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    
    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)
    
    # 인코더를 num_layer에 쌓기
    # 논문에서 i=6
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name=f'encoder_layer_{i}')([outputs, padding_mask])
        
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
  
  
# 디코더의 첫번째 서브층에서 미래 토큰을 mask 하는 함수
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩마스크도 포함
    
    return tf.maximum(look_ahead_mask, padding_mask)
  
def decoder_layer(dff, d_model, num_heads, dropout, name='decoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    
    # look-ahead mask(첫 번째 서브층)
    # 마스크 존재
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    
    # padding mask(두 번째 서브층)
    # 마스트 없음
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    # Multi-Head Attention (첫 번째 서브층 / masked self-attention)
    attention1 = MultiHeadAttention(d_model, num_heads, name='attention_1')(inputs={
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': look_ahead_mask # 룩-어헤드 마스크
    })
    
    # 잔차 연결과 층 정규화
    attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    
    # Multi-Head Attention (두 번째 서브층 / Decoder-Encoder Attention)
    attention2 = MultiHeadAttention(d_model, num_heads, name='attention_2')(inputs={
        'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask 
    }) # 패딩 마스크,  Q != K = V
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = Dropout(rate=dropout)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention2 + attention1)
    
    # Position-wise FFNN (세 번째 서브층)
    outputs = Dense(units=dff, activation='relu')(attention2)
    outputs = Dense(units=d_model)(outputs)
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention2)
    
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)
  
  
  
# 쌓기
def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    
    # 디코더는 룩어헤드 마스크(첫 번째 서브층)와 패딩 마스크(두 번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)
    
    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, 
                                d_model=d_model, 
                                num_heads=num_heads, 
                                dropout=dropout, 
                                name=f'decoder_layer_{i}')(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)
  
def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='transformer'):
    
    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    
    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')
    
    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
    
    # 디코더의 룩어헤드 마스크(첫 번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
    
    # 디코더의 패딩 마스크(두 번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)
    
    # 인코더의 출력은 enc_outputs. 디코더로 전달됨
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])
    # 인코더의 입력은 입력문장과 패딩 마스크
    
    # 디코더의 출력은 dec_outputs, 출력층으로 전달됨
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[dec_inputs,
                                                                                                                                               enc_outputs,
                                                                                                                                               look_ahead_mask,
                                                                                                                                               dec_padding_mask])
    
    # 다음 단어 예측을 위한 출력층
    outputs = Dense(units=vocab_size, name='outputs')(dec_outputs)
    
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
  
