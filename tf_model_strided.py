import tensorflow.compat.v1 as tf
import time
import numpy as np
from tensorflow.keras import datasets, layers, models, losses

tf.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()

tf.disable_eager_execution()
tf.compat.v1.disable_eager_execution()

tf.reset_default_graph()
tf.compat.v1.reset_default_graph()

datatype=tf.float32

print ("TF VERSION ",tf.__version__)

def conv1d(name, inputs, kernel, chan_in, chan_out, dilation):
    y=tf.pad(inputs, tf.constant([[0,0], [0,0], [dilation*(kernel-1), 0], [0,0]]))
    y=tf.layers.conv2d(y, chan_out, (1, kernel), padding='VALID', data_format='channels_last', dilation_rate=(1, dilation), bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer())
    return y

def conv1d_pad(name, inputs, kernel, chan_in, chan_out, dilation, relu=False):
    prev_inputs=tf.placeholder(datatype, [1, 1, dilation*(kernel-1), chan_in])
    # print ('SHAPE1 ',inputs.shape)
    # print ('SHAPE2 ',prev_inputs.shape)
    y=tf.concat([inputs, prev_inputs], 2)
    # print ('SHAPE3 ',y.shape)
    next_inputs=tf.slice(y, 
        [0,0,y.get_shape()[2]-dilation*(kernel-1),0], 
        [y.get_shape()[0],y.get_shape()[1],dilation*(kernel-1),y.get_shape()[3]])
    y=tf.layers.conv2d(y, chan_out, (1, kernel), padding='VALID', data_format='channels_last', dilation_rate=(1, dilation), bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer(), activation=(tf.nn.relu if relu else None))

    return prev_inputs, next_inputs, y

def conv11(name, inputs, chan_in, chan_out):
    print ('SHAPE 2 ',inputs.shape)
    print ('CHAN_OUT ',chan_out)
    y=tf.layers.conv2d(inputs, chan_out, (1, 1), 
        padding='VALID', data_format='channels_last', 
        bias_initializer=tf.random_normal_initializer(), 
        kernel_initializer=tf.random_normal_initializer())
    print ("CONV11 done")
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
def model_pad_strided(x, T, chan_in, chan_mid, chan, synth_mid, synth_hid, block_size, freq, kernel, synth_layer, synth_rep, sess):
    F=block_size
    x=tf.placeholder(datatype, [1, T, F, chan_in])

    xs=[]
    ys=[]
    # layers
    def forward(x): #B, T, F, C
        #set_session(sess)
        assert(freq==x.shape[2])
        # with tf.variable_scope("model3"):
        print ('SHAPE 1 ',x.shape)
        y=tf.reshape(x, (-1,1,T,F*chan_in))

        y=conv11('map', y, chan_in*F, chan)
        return y
        # yshort=y

        # y=conv11('map2', y, chan, synth_mid)

        # res_sum=None
        # for i in range(synth_rep):
        #     dilation=1
        #     for j in range(synth_layer):
        #         yres=y
        #         xin, xout, y=conv1d_pad('synth'+str(i)+str(j), y, kernel, synth_mid, synth_hid, dilation, relu=True)
        #         xs.append(xin)
        #         ys.append(xout)
                
        #         y1=conv11('2synth'+str(i)+str(j), y, synth_hid, synth_mid)
        #         if j==synth_layer-1:
        #             if i!=synth_rep-1:
        #                 if res_sum is None:
        #                     res_sum=conv11('3synth'+str(i)+str(j), y, synth_hid, synth_mid)
        #                 else:
        #                     ysum=conv11('3synth'+str(i)+str(j), y, synth_hid, synth_mid)
        #                     ysum=tf.image.resize_nearest_neighbor(ysum, (1,T))
        #                     res_sum=res_sum+ysum
        #                 y=y1+yres
        #                 y=tf.layers.conv2d(y, synth_mid, (1, 2), padding='VALID', strides=(1,2), data_format='channels_last', bias_initializer=tf.random_normal_initializer(), kernel_initializer=tf.random_normal_initializer())
        #             else:
        #                 ysum=tf.image.resize_nearest_neighbor(y1, (1,T))
        #                 y=res_sum+ysum
        #         else:
        #             y=y1+yres
        #         dilation*=kernel
                    
        # y=conv11('reduce1', y, synth_mid, synth_hid)
        # y=tf.nn.relu(y)
        # y=conv11('reduce2', y, synth_hid, chan)
        # y=complex_gate(y, yshort)
        # y=conv11('final', y, chan, F)
        # return xs, ys, y
    
    return x, forward

# input_tensor = tf.zeros([1,192,32,18],dtype=datatype)
# x,myforward=model_pad_strided(input_tensor, 192, 18, 0, 512, 128, 192, 32, 32, 3, 4, 4, 0)
# xs,ys,y=myforward(x)

class DeepBeamModel(tf.keras.Model):
  def __init__(self):
    super(DeepBeamModel, self).__init__(name='')

  def call(self, input_tensor, training=False):
    x,myforward=model_pad_strided(input_tensor, 192, 18, 0, 512, 128, 192, 32, 32, 3, 4, 4, 0)
    y=myforward(x)
    return y

print ("EAGER ",tf.executing_eagerly())
block = DeepBeamModel()
input_tensor = tf.zeros([1,192,32,18],dtype=datatype)
block.build((1,192,32,18))
block.summary()

# s=time.time()
block(input_tensor=input_tensor)
# print ((time.time()-s)*1000)
block.save('model.pb',save_format="tf")
