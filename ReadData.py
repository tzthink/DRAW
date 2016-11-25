import cPickle
import matplotlib.pyplot as plt
import numpy as np
import random


class ReadData:
    file_name_array = ['Dataset/cifar-10-batches-py/data_batch_1',
                       'Dataset/cifar-10-batches-py/data_batch_2',
                       'Dataset/cifar-10-batches-py/data_batch_3',
                       'Dataset/cifar-10-batches-py/data_batch_4',
                       'Dataset/cifar-10-batches-py/data_batch_5',
                       'Dataset/cifar-10-batches-py/test_batch']

    file_dict = {}
    data_dict = {}
    grayscale_data_dict = {}
    label_dict = {}
    label_name_dict = {}
    data_name_list = []

    grayInOneArray = np.array(0)
    grayInOneArrayLen = 0
    batch_pointer = 0

    def __init__(self):
        self.generate_file_dict()
        self.generate_data_dict()
        self.generate_data_name_list()
        self.generate_grayscale_data()
        self.generate_label_dict()
        self.generate_label_name_dict()
        self.merge_all_gray_data_to_one_array()

    def get_grayscale_data_dict(self):
        return self.grayscale_data_dict

    def get_label_dict(self):
        return self.label_dict

    def get_label_name_dict(self):
        return self.label_name_dict

    def get_data_name_list(self):
        return self.data_name_list

    def unpickle(self, file):
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def generate_file_dict(self):
        for file in self.file_name_array:
            self.file_dict.update({file: self.unpickle(file)})

    # data_dict: dict of training data and testing data
    def generate_data_dict(self):
        for file_dict_value in self.file_dict.values():
            list = []
            for i in range(file_dict_value['data'].shape[0]):
                data = file_dict_value['data'][i].astype('float32')
                data = data.reshape((3, 32, 32)).transpose(1, 2, 0)  # (32, 32, 3)
                list.append(data)
            self.data_dict.update({file_dict_value['batch_label']: list})

    # data_name_list: keys of data_dict
    def generate_data_name_list(self):
        self.data_name_list = self.data_dict.keys()
        # print data_name_list
        # ['training batch 2 of 5', 'training batch 3 of 5', 'training batch 4 of 5',
        # 'training batch 5 of 5', 'training batch 1 of 5', 'testing batch 1 of 1']

    # grayscale data
    def generate_grayscale_data(self):
        for data_name in self.data_name_list:
            list = []
            for i in range(len(self.data_dict[data_name])):
                data = self.data_dict[data_name][i][:, :, 0] + \
                       self.data_dict[data_name][i][:, :, 1] + \
                       self.data_dict[data_name][i][:, :, 2]
                data = data / 3
                list.append(data)
            self.grayscale_data_dict.update({data_name: list})

    # label dict
    def generate_label_dict(self):
        for file_dict_value in self.file_dict.values():
            list = []
            for i in range(len(file_dict_value['labels'])):
                label = file_dict_value['labels'][i]
                list.append(label)
            self.label_dict.update({file_dict_value['batch_label']: list})

    # label name dict
    def generate_label_name_dict(self):
        for file_dict_value in self.file_dict.values():
            list = []
            for i in range(len(file_dict_value['filenames'])):
                label_name = file_dict_value['filenames'][i]
                list.append(label_name)
            self.label_name_dict.update({file_dict_value['batch_label']: list})

    # all train gray data in one array
    def merge_all_gray_data_to_one_array(self):
        list = []
        for x in xrange(5):
            print
            for y in xrange(len(self.grayscale_data_dict[self.data_name_list[x]])):
                l = np.reshape(self.grayscale_data_dict[self.data_name_list[x]][y], newshape=(1024))
                l /= 255
                list.append(l)
        self.grayInOneArray = np.asarray(list)
        self.grayInOneArrayLen = self.grayInOneArray.shape[0]

    # get next batch
    def get_next_batch(self, batch_size):
        list = []
        for i in xrange(batch_size):
            list.append(self.grayInOneArray[self.batch_pointer])
            self.batch_pointer = (self.batch_pointer + 1) % self.grayInOneArrayLen
        return np.asarray(list)

    # get one image
    def get_one_image(self):
        r = random.randint(0, self.grayInOneArrayLen-1)
        return self.grayInOneArray[r]

    # show a image
    def show_a_image(self, image):
        show_image = np.reshape(image, newshape=[32, 32])
        plt.imshow(show_image, cmap='gray')
        plt.show()


# show image
# rd = ReadData()
# image = rd.grayscale_data_dict[rd.data_name_list[0]][10]
# plt.imshow(image)
# plt.show()
#
# print rd.grayscale_data_dict[rd.data_name_list[0]][11]  # (32, 32)



"""
    print type(dict)  # <type 'dict'>
    print len(dict)                     # 4
    print dict.keys()  # ['data', 'labels', 'batch_label', 'filenames']
    print type(dict['data'])  # <type 'numpy.ndarray'>
    print dict['data'].shape  # (10000, 3072)
    print dict['data'][0].shape  # (3072,)
    print len(dict['labels'])  # 10000
    print type(dict['labels'])  # <type 'list'>
    print type(dict['batch_label'])  # <type 'str'>
    print len(dict['filenames'])  # 10000
    print type(dict['filenames'])  # <type 'list'>
    """
