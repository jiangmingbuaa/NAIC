# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]
        pids = set(pids)
        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs = self.get_imagedata_info(query)
        num_gallery_imgs = len(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(-1, num_gallery_imgs))
        print("  ----------------------------------------")
