# Face Recognition for Caffe2
Caffe2 is a lightweight, modular, speed, so i want to do face recognition experiment with it, like light cnn, center loss, large margin softmax, Angular softmax and so on.

## Reference
[origin_lsoftmax](https://github.com/wy1iu/LargeMargin_Softmax_Loss.git)
[mxnet_lsoftmax](https://github.com/luoyetx/mx-lsoftmax.git)

## Files
- Original Caffe2 library
- Large margin softmax operator
  * caffe2/operators/lsoftmax_with_loss.h
  * caffe2/operators/lsoftmax_with_loss.cc
  * caffe2/operators/lsoftmax_with_loss.cu
  * caffe2/python/operator_test/lsoftmax_with_loss_test.py
- face_example
  * caffe2/python/examples/sphereface_trainer.py
  * caffe2/python/models/sphereface.py
- mnist_example
  * caffe2/python/examples/mnist_trainer.py

## Result
- mnist_example for lsoftmax 
  * margin = 1  <p align='center'><img src='caffe2/python/examples/result/mnist/distance-margin-1.png' style='max-width:600px'></img></p> 
  * margin = 2  <p align='center'><img src='caffe2/python/examples/result/mnist/distance-margin-2.png' style='max-width:600px'</img></p>
  * margin = 3  <p align='center'><img src='caffe2/python/examples/result/mnist/distance-margin-3.png' style='max-width:600px'</img></p>
  * margin = 4  <p align='center'><img src='caffe2/python/examples/result/mnist/distance-margin-4.png' style='max-width:600px'</img></p>
  
## Build
Use my version of  caffe2, follow origin caffe2 installation [Installation](http://caffe2.ai/docs/getting-started.html)
