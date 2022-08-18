# transformer

  
small_transformer = transformer(vocab_size = 10000, # 임의로 정함
                                num_layers = 4, 
                                dff = 512,
                                d_model = 128,
                                num_heads = 4,
                                dropout = 0.1,
                                name="small_transformer")

tf.keras.utils.plot_model(small_transformer, to_file='small_transformer.png', show_shapes=True)
