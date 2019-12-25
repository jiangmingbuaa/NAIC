# encoding: utf-8
"""
@author:  lizihan
@contact: lizihan233@qq.com
"""

import glob
import re

import os.path as osp
from .bases import BaseImageDataset


class NAIC_val(BaseImageDataset):

    dataset_dir = 'NAIC_val'

    def __init__(self, root='/root/share/dataset/reid/', verbose=True, **kwargs):
        super(NAIC_val, self).__init__()
        self.dataset_dir = osp.join(root, 'NAIC2')
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query_a')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_a')

        self._check_before_run()

        file_dir = '/root/share/dataset/reid/NAIC2/'
        train_file = osp.join(file_dir, 'train_val_list.txt')
        query_file = osp.join(file_dir, 'query_a_val_list.txt')
        gallery_file = osp.join(file_dir, 'gallery_a_val_list.txt')
        train = self._process_dir(self.train_dir, train_file, relabel=True)
        query = self._process_dir(self.query_dir, query_file, relabel=False)
        gallery = self._process_dir(self.gallery_dir, gallery_file, relabel=False)

        if verbose:
            print("=> NAIC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs= self.get_imagedata_info(self.query)
        self.num_gallery_imgs = len(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, label_list_name, relabel=False):
        dataset = []
        # import pdb
        # pdb.set_trace()
        if label_list_name != '':
            label_list = []
            with open(label_list_name, 'r') as f:
                label_list = f.read().split('\n')
            img_paths = []
            label = []
            pid_container = set()
            for name_label in label_list:
                # print(name_label)
                #import pdb
                #pdb.set_trace()
                img_paths.append(name_label.split(' ')[0])
                label.append(int(name_label.split(' ')[1]))
                pid_container.add(int(name_label.split(' ')[1]))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for i in range(len(img_paths)):
                pid = label[i]
                if relabel: pid = pid2label[pid]
                img_name = img_paths[i].split('/')[-1]
                dataset.append((img_paths[i], pid))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.png'))
            for img_path in img_paths:
                img_name = img_path.split('/')[-1]
                dataset.append((img_path, img_name))

        return dataset

dataset = NAIC_val()
# print(dataset)