import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

datatype=tf.float32

def conv1d(name, inputs, kernel, ch_in, ch_out, dilation):
    y=tf.pad(inputs, tf.constant([[0,0], [0,0], [dilation*(kernel-1), 0], [0,0]]))
    y=tf.layers.conv2d(y, ch_out, (1, kernel), padding='VALID', data_format='channels_last', dilation_rate=(1, dilation), bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer())
    return y

def conv1d_pad(name, inputs, kernel, ch_in, ch_out, dilation, relu=False):
    prev_inputs=tf.placeholder(datatype, [None, 1, dilation*(kernel-1), ch_in])
    y=tf.concat([inputs, prev_inputs], 2)
    next_inputs=tf.slice(y, [0,0,y.get_shape()[2]-dilation*(kernel-1),0], [y.get_shape()[0],y.get_shape()[1],dilation*(kernel-1),y.get_shape()[3]])
    y=tf.layers.conv2d(y, ch_out, (1, kernel), padding='VALID', data_format='channels_last', dilation_rate=(1, dilation), bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer(), activation=(tf.nn.relu if relu else None))

    return prev_inputs, next_inputs, y

def conv11(name, inputs, ch_in, ch_out):
    y=tf.layers.conv2d(inputs, ch_out, (1, 1), padding='VALID', data_format='channels_last', bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer())
    return y
    
def complex_gate(mask, x):
    mask_real, mask_imag=tf.split(mask, num_or_size_splits=2, axis=3)
    mask_abss=mask_real*mask_real+mask_imag*mask_imag
    mask_rabs=tf.math.rsqrt(mask_abss)
    mask_tanh=tf.math.tanh(mask_rabs*mask_abss)*mask_rabs
    mask_real=mask_real*mask_tanh
    mask_imag=mask_imag*mask_tanh
    
    x_real, x_imag=tf.split(x, num_or_size_splits=2, axis=3)
    res_real=x_real*mask_real-x_imag*mask_imag
    res_imag=x_real*mask_imag+x_imag*mask_real
    
    return tf.concat([res_real, res_imag], 3)
    
# ch_in: 18 (9*2); ch: 512 (256*2); synth_mid:128 (64*2); synth_hid: 192 (96*2); block_size:32; freq: 32; kernel: 3; synth_layer: 4; synth_rep:4  
def model_pad_strided(T, ch_in, ch_mid, ch, synth_mid, synth_hid, block_size, freq, kernel, synth_layer, synth_rep, sess):
    F=block_size
    x=tf.placeholder(datatype, [1, T, F, ch_in])
    
    xs=[]
    ys=[]
    # layers
    def forward(x): #B, T, F, C
        #set_session(sess)
        assert(freq==x.shape[2])
        with tf.variable_scope("model3"):
            y=tf.reshape(x, (-1,1,T,F*ch_in))
            y=conv11('map', y, ch_in*F, ch)
            
            yshort=y

            y=conv11('map2', y, ch, synth_mid)

            res_sum=None
            for i in range(synth_rep):
                dilation=1
                for j in range(synth_layer):
                    yres=y
                    xin, xout, y=conv1d_pad('synth'+str(i)+str(j), y, kernel, synth_mid, synth_hid, dilation, relu=True)
                    xs.append(xin)
                    ys.append(xout)
                    
                    y1=conv11('2synth'+str(i)+str(j), y, synth_hid, synth_mid)
                    if j==synth_layer-1:
                        if i!=synth_rep-1:
                            if res_sum is None:
                                res_sum=conv11('3synth'+str(i)+str(j), y, synth_hid, synth_mid)
                            else:
                                ysum=conv11('3synth'+str(i)+str(j), y, synth_hid, synth_mid)
                                ysum=tf.image.resize_nearest_neighbor(ysum, (1,T))
                                res_sum=res_sum+ysum
                            y=y1+yres
                            y=tf.layers.conv2d(y, synth_mid, (1, 2), padding='VALID', strides=(1,2), data_format='channels_last', bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer())
                        else:
                            ysum=tf.image.resize_nearest_neighbor(y1, (1,T))
                            y=res_sum+ysum
                    else:
                        y=y1+yres
                    dilation*=kernel
                        

            y=conv11('reduce1', y, synth_mid, synth_hid)
            y=tf.nn.relu(y)
            y=conv11('reduce2', y, synth_hid, ch)
            y=complex_gate(y, yshort)
            y=conv11('final', y, ch, F)
            return xs, ys, y
    
    return x, forward
    