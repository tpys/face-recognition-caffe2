import os
import caffe2.python.dataset.data_util as data_util
import caffe2.python.dataset.lfw as lfw

script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    data_dir = '/media/tpys/ssd/lfw_align_128x128/'
    output_dir = '/media/tpys/ssd/lfw_align_128x128_lmdb'
    lfw_test = True
    shuffle = False
    use_caffe = True;
    face_dataset = data_util.get_dataset(data_dir)
    name2label = {face_dataset[i].name:i for i in range(len(face_dataset))}

    list_file = list_file = os.path.join(data_dir, 'list.txt')
    if os.path.isfile(list_file):
        os.remove(list_file)

    print ('class num: {}'.format(len(name2label.items())))
    if lfw_test:
        lfw_pairs = os.path.join(script_dir, "lfw_pairs.txt")
        pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
        paths, actual_issame = lfw.get_paths(os.path.expanduser(data_dir), pairs, 'jpg')
        print ('pair num: {}'.format(len(pairs)))
        with open(list_file, 'w') as f:
            for i, path in zip(range(len(paths)), paths):
                name = os.path.basename(os.path.dirname(path))
                sample = '{:s}/{:s} {:d}\n'.format(name,
                                                   os.path.basename(path),
                                                   name2label[name])
                # sample = '{:s} {:d}\n'.format(path, i)
                # print (sample.strip())
                f.write(sample)
    else:
        list_file = os.path.join(data_dir, 'list.txt')
        with open(list_file, 'w') as f:
            for image_class in face_dataset:
                for img in image_class.image_paths:
                    sample = '{:s}/{:s} {:d}\n'.format(image_class.name, os.path.basename(img), name2label[image_class.name])
                    # sample = '{:s} {:d}\n'.format(img, name2label[image_class.name])
                    f.write(sample)

    app_file = "/home/tpys/tools/caffe/build/tools/convert_imageset"
    if use_caffe:
        if shuffle:
            cmd = '{:s} {:s} {:s} {:s} -shuffle'.format(app_file, data_dir, list_file, output_dir)
        else:
            cmd = '{:s} {:s} {:s} {:s}'.format(app_file, data_dir, list_file, output_dir)

    else:
        cmd = '{:s} -input_folder {:s} ' \
              '-list_file {:s} ' \
              '-output_db_name {:s} ' \
              '-shuffle {:s}'.format(app_file,
                                     data_dir,
                                     list_file,
                                     output_dir,
                                     shuffle)

    print (cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()