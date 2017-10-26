import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects

from caffe2.python import core, model_helper, workspace, brew, optimizer
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python import net_drawer

core.GlobalInit(['caffe2', '--caffe2_log_level=-1'])


def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    mean = np.array(128, dtype=np.float32)
    workspace.FeedBlob("mean", mean)
    data = model.Sub([data, "mean"], "data", broadcast=1)
    data = model.Scale(data, data, scale=float(1./128))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label


def AddLeNetModel(model, data, label=None, no_loss=False, embedding_size=2, class_num=10, margin=0):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.
    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=32, kernel=5)
    prelu1 = brew.relu(model, conv1, "prelu1")

    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, prelu1, 'pool1', kernel=2, stride=2)

    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=32, dim_out=64, kernel=5)
    prelu2 = brew.relu(model, conv2, "prelu2")

    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, prelu2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=64 * 4 * 4, dim_out=256)
    prelu3 = brew.relu(model, fc3, "prelu3")

    embedding = brew.fc(model, prelu3, 'embedding', 256, embedding_size)    
    if no_loss:
        return embedding

    if label is not None:
        output = brew.lsoftmax(model, [embedding, label], "fc4", embedding_size, class_num,
                                margin=margin,
                                base=float(10),#200
                                lambda_min=float(0)) #5
        fc4 = output[0]
        softmax, loss = model.SoftmaxWithLoss([fc4, label], ['softmax', 'loss'])
        accuracy = brew.accuracy(model, [softmax, label], "accuracy")

        return [loss, accuracy]
    else:
        fc4 = brew.fc(model, embedding, 'fc4', embedding_size, class_num)
        return brew.softmax(model, fc4, "softmax")


def AddTrainingOperators(model, loss):
    model.AddGradientOperators([loss])
    optimizer.add_weight_decay(model, 5e-4)
    stepsz = int(10 * 60000 / 128)
    opt = optimizer.build_sgd(model, 
        base_learning_rate=0.01, 
        policy="step", 
        stepsize=stepsz, 
        gamma=0.1, 
        momentum=0.9)
    # opt = optimizer.build_yellowfin(model)

    return opt


def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
        # Now, if we really want to be verbose, we can summarize EVERY blob
        # that the model produces; it is probably not a good idea, because that
        # is going to take time - summarization do not come for free. For this
        # demo, we will only show how to summarize the parameters and their
        # gradients.

def TrainTest(args):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        
    np.random.seed(123)  # make test deterministic
    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
    data, label = AddInput(
        train_model,
        batch_size=128,
        db=os.path.join(args.data_folder, 'mnist-train-nchw-lmdb'),
        db_type='lmdb')

    loss, _ = AddLeNetModel(train_model, data, label, margin=args.margin)
    AddTrainingOperators(train_model, loss)
    # AddBookkeepingOperators(train_model)

    # Testing model. We will set the batch size to 100, so that the testing
    # pass is 100 iterations (10,000 images in total).
    # For the testing model, we need the data input part, the main LeNetModel
    # part, and an accuracy part. Note that init_params is set False because
    # we will be using the parameters obtained from the train model.
    test_model = model_helper.ModelHelper(name="mnist_test", arg_scope=arg_scope, init_params=False)
    data, label = AddInput(
        test_model,
        batch_size=100,
        db=os.path.join(args.data_folder, 'mnist-test-nchw-lmdb'),
        db_type='lmdb')

    AddLeNetModel(test_model, data, label, margin=args.margin)

    # Deployment model. We simply need the main LeNetModel part.
    deploy_model = model_helper.ModelHelper(name="mnist_deploy", arg_scope=arg_scope, init_params=False)
    AddLeNetModel(deploy_model, data, no_loss=True, margin=args.margin)
    # You may wonder what happens with the param_init_net part of the deploy_model.
    # No, we will not use them, since during deployment time we will not randomly
    # initialize the parameters, but load the parameters from the db.
    # The parameter initialization network only needs to be run once.
    workspace.RunNetOnce(train_model.param_init_net)
    # creating the network
    workspace.CreateNet(train_model.net, overwrite=True)
    # set the number of iterations and track the accuracy & loss

    # print (str(train_model.param_init_net.Proto()))
    graph = net_drawer.GetPydotGraphMinimal(train_model.net.Proto(),
                                            "mnist", 
                                            rankdir="LR", 
                                            minimal_dependency=True)

    graph.write(os.path.join(args.save_folder, "mnist.pdf"), format='pdf')

    total_iters = 15000
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)
    # Now, we will manually run the network for 200 iterations.
    for i in range(total_iters):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.FetchBlob('accuracy')
        loss[i] = workspace.FetchBlob('loss')
        if i % 10 == 0:
            print("#Iteration: {}, lambda: {:.3f}, loss: {:.3f}, train_acc: {:.3f}".
                  format(i, workspace.FetchBlob('lambda').tolist(), loss[i], accuracy[i]))

    # run a test pass on the test net
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    test_accuracy = np.zeros(100)
    embeds = []
    labels = []
    for i in range(100):
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracy[i] = workspace.FetchBlob('accuracy')
        embeds.append(workspace.FetchBlob('embedding'))
        labels.append(workspace.FetchBlob('label'))

    embeds = np.vstack(embeds)
    labels = np.hstack(labels)
    # vis, plot code from https://github.com/pangyupo/mxnet_center_loss
    num = len(labels)
    names = dict()
    for i in range(10):
        names[i]=str(i)
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeds[:,0], embeds[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(embeds[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    fname = "distance-margin-{}.png".format(args.margin)
    plt.savefig(os.path.join(args.save_folder, fname))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(loss, 'b')
    ax2.plot(accuracy, 'r')
    ax2.legend(('Loss', 'Accuracy'), loc='upper right')
    plt.show()

    fname = "loss-margin-{}.png".format(args.margin)
    plt.savefig(os.path.join(args.save_folder, fname))
    print('test_accuracy: %f' % test_accuracy.mean())


def main():
    parser = argparse.ArgumentParser(description="Caffe2: MNIST training")
    parser.add_argument("--data_folder",
                        type=str,
                        default="dataset/",
                        help="Path to training and test data",
                        required=False)

    parser.add_argument("--save_folder",
                    type=str,
                    default="result/mnist/",
                    help="where to save training result",
                    required=False)

    parser.add_argument("--num_labels",
                        type=int,
                        default=10,
                        help="Number of labels")

    parser.add_argument("--embedding_size",
                        type=int,
                        default=2,
                        help="It's feature dim")

    parser.add_argument("--margin",
                        type=int,
                        default=4,
                        help="L-softmax margin type")

    args = parser.parse_args()
    TrainTest(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=-1'])
    main()
