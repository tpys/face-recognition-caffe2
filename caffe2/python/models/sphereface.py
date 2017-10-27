## @package resnet
# Module caffe2.python.example.resnet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, brew
import os
'''
Utility for creating sphereface
See "SphereFace: Deep Hypersphere Embedding for Face Recognition" by Weiyang, Liu et. al. 2017
'''


xavier = ("XavierFill", {})
msra = ("MSRAFill", {})
gaussian = ("GaussianFill", {"mean" : 0.0, "std" : 0.01})
constant = ("ConstantFill", {"value" : 0.0})

class SpherefaceBuilder():
    '''
    Helper class for sphereface.
    '''

    def __init__(self, model, in_blob, in_dim):
        self.model = model
        self.prev_blob = in_blob
        self.prev_dim = in_dim
        self.comp_count = 1
        self.comp_idx = 1

    def conv_prelu(self, in_blob, out_dim,
                   kernel=3,
                   stride=1,
                   pad=1,
                   no_bias=0,
                   weight_init=msra,
                   bias_init=constant):
        self.prev_blob = brew.conv(self.model,
                                   in_blob,
                                   "conv{}_{}".format(self.comp_idx, self.comp_count),
                                   self.prev_dim,
                                   out_dim,
                                   weight_init=weight_init,
                                   bias_init=bias_init,
                                   kernel=kernel,
                                   stride=stride,
                                   pad=pad,
                                   no_bias=no_bias)
        # bn = brew.spatial_bn(self.model, self.prev_blob, "bn{}_{}".format(self.comp_idx, self.comp_count), out_dim)
        self.comp_count += 1
        return brew.relu(self.model, self.prev_blob, self.prev_blob)

    def add_residual_block(self, in_blob, out_dim):
        self.conv_prelu(in_blob, out_dim)
        self.prev_dim = out_dim
        self.conv_prelu(self.prev_blob, out_dim)
        self.prev_dim = out_dim
        self.prev_blob = brew.sum(self.model, [self.prev_blob, in_blob], "res{}_{}".format(self.comp_idx, self.comp_count - 1))
        return self.prev_blob

    def create_convolution(self, out_dim, num_block):
        self.reset_count(1)
        self.conv_prelu(self.prev_blob, out_dim,
                        kernel=3,
                        stride=2,
                        pad=1,
                        no_bias=0,
                        weight_init=msra)
        self.prev_dim = out_dim
        for i in range(num_block):
            self.prev_blob = self.add_residual_block(self.prev_blob, out_dim)
        self.comp_idx += 1
        return self.prev_blob

    def reset_count(self, val):
        self.comp_count = val


def create_net(model,
             data,
             label=None,
             in_dim=3,
             class_num=10575,
             feature_dim=512,
             is_test=False,
             no_loss=False,
             fp16_data=False):
    if fp16_data:
	    data = model.FloatToHalf(data, data + "_fp16")

    builder = SpherefaceBuilder(model, data, in_dim)
    filters = [64, 128, 256, 512]
    blocks = [1, 2, 4, 1]
    for i in range(0, 4):
        builder.create_convolution(filters[i], blocks[i])

    fc5 = brew.fc(model, builder.prev_blob, "fc5", 512 * 8 * 8, feature_dim)

    if fp16_data:
        fc5 = model.net.HalfToFloat(fc5, fc5 + '_fp32')

    if is_test:
        return fc5

    # last_out = builder.prev_blob
    if label is not None:
        output = brew.lsoftmax(model, [fc5, label], "fc6",
                                feature_dim,
                                class_num,
                                margin=3,
                                base=float(10000),
                                lambda_min=float(5))

        fc6 = output[0]

        if fp16_data:
        	fc6 = model.net.HalfToFloat(fc6, fc6 + '_fp32')

        if no_loss:
            return fc6

        softmax, loss = model.SoftmaxWithLoss([fc6, label], ['softmax', 'loss'])
        return softmax, loss