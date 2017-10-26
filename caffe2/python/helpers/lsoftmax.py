# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package fc
# Module caffe2.python.helpers.fc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags


def _lsoftmax(model, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None,
              WeightInitializer=None, enable_tensor_core=False, **kwargs):
    assert len(blob_in) == 2, print("a list with length two, which contain X and Label")
    X = blob_in[0]
    L = blob_in[1]

    WeightInitializer = initializers.update_initializer(
        WeightInitializer, weight_init, ("XavierFill", {})
    )
    if not model.init_params:
        WeightInitializer = initializers.ExternalInitializer()

    blob_out = blob_out or model.net.NextName()
    weight = model.create_param(
        param_name=blob_out + '_w',
        shape=[dim_out, dim_in],
        initializer=WeightInitializer,
        tags=ParameterTags.WEIGHT
    )

    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    output = [blob_out, "lambda", "x_norm", "w_norm"]

    return op_call([X, weight, L], output, **kwargs)



def lsoftmax(model, *args, **kwargs):
    return _lsoftmax(model, model.net.LSoftmaxWithLoss, *args, **kwargs)