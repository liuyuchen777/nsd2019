import numpy as np

def Load_file(dataset = 'MNIST', mode = 'train'):
    if dataset == 'MNIST':
        return Load_MNIST(mode)
    if dataset == 'cifar10':
        return Load_cifar10(mode)

def Load_MNIST(mode = 'train'):
    if mode == 'train':
        file_header = 'train'
    else:
        file_header = 't10k'
    with open('data/MNIST/'+file_header+'-images.idx3-ubyte','rb') as f:
        byte = f.read(4)
        magic_num  = int.from_bytes(byte, byteorder='big')    #magic num 2051 for test_x
        byte = f.read(4)
        image_num  = int.from_bytes(byte, byteorder='big')    #num of images
        byte = f.read(4)
        row_num    = int.from_bytes(byte, byteorder='big')    #num of rows
        byte = f.read(4)
        column_num = int.from_bytes(byte, byteorder='big')    #num of columns
        byte = f.read(image_num * row_num * column_num)       #all images
        data_image = np.frombuffer(byte, dtype = np.uint8, count = -1)
        data_image = data_image.reshape((image_num,1,row_num,column_num)) # Data_number * Channel * Height * Width
    with open('data/MNIST/'+file_header+'-labels.idx1-ubyte','rb') as f:
        byte = f.read(4)
        magic_num  = int.from_bytes(byte, byteorder='big')    #magic num 2051 for test_x
        byte = f.read(4)
        image_num  = int.from_bytes(byte, byteorder='big')    #num of images
        byte = f.read(image_num * row_num * column_num)       #all images
        data_label = np.frombuffer(byte, dtype = np.uint8, count = -1)
        data_label = data_label.reshape((image_num,1)) # Data_number * 1
    return data_image,data_label


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def Load_cifar10(mode = 'train'):
    data_image = []
    data_label = []
    if mode == 'train':
        for i in range(5):
            data = unpickle('data/cifar-10-batches-py/data_batch_'+str(i+1))
            data_image.append(np.asarray(data[b'data']))
            data_label.append(np.asarray(data[b'labels']))
    else:
        data = unpickle('data/cifar-10-batches-py/test_batch')
        data_image.append(np.asarray(data[b'data']))
        data_label.append(np.asarray(data[b'labels']))
    data_image = np.asarray(data_image)
    data_label = np.asarray(data_label)

    return data_image.reshape(-1,3,32,32), data_label.reshape(-1,1)

    return data_image,data_label


def Show_data(images, labels):
    import matplotlib.pyplot as plt
    images_num  = images.shape[0]
    for i in range(images_num):
        plt.imshow(images[i][0])
        print(labels[i][0])
        plt.show()