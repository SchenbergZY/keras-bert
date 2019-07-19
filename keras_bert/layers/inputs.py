from keras_bert.backend import keras


def get_inputs(seq_len):
    """Get input layers.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Masked']
    #a = [keras.layers.Input(
    #    shape=(seq_len,),
    #    name='Input-%s' % name,
    #) for name in names]
    #return a
    aa = keras.layers.Input(
        shape=(seq_len,),
        name='Input-Token',
    )
    bb = keras.layers.Input(
        shape=(seq_len,),
        name='Input-Segment',
    )
    cc = keras.layers.Input(
        shape=(seq_len,),
        name='Input-Masked',
    )
    aa_mask = keras.layers.Mask(0,name='Input-Token-mask')(aa)
    bb_mask = keras.layers.Mask(0,name='Input-Segment-mask')(bb)
    return [aa_mask,bb_mask,cc]
