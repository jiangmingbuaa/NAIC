# encoding: utf-8
"""
@author:  lizihan
@contact: lizihan233@qq.com
"""

import glob
import re

import os.path as osp
from .bases import BaseImageDataset


class NAIC(BaseImageDataset):

    dataset_dir = 'NAIC'

    def __init__(self, root='/root/share/dataset/reid', verbose=True, gallery_dir='gallery_B', query_dir='query_B', **kwargs):
        super(NAIC, self).__init__()
        root = root='/root/share/dataset/reid'
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = osp.join(root, 'NAIC2')
        # self.dataset_dir = root # /tmp/data
        self.train_dir = osp.join(self.dataset_dir, 'train') # train dir
        self.query_dir = osp.join(self.dataset_dir, 'query_b') # query dir
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_b') # gallery dir
        # self.query_dir = osp.join(self.dataset_dir, 'test', query_dir)
        # self.gallery_dir = osp.join(self.dataset_dir, 'test', gallery_dir)

        self._check_before_run()

        # train = self._process_dir(self.train_dir, 'train_all_list.txt', relabel=True) # train file list
        train = self._process_dir(self.train_dir, 'train_list.txt', relabel=True)
        # query = self._process_dir(self.query_dir, 'query_a_list.txt', relabel=False) # query file list
        query = self._process_dir(self.query_dir, '', relabel=False)
        gallery = self._process_dir(self.gallery_dir, '', relabel=False)

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
            with open(osp.join(self.dataset_dir, label_list_name), 'r') as f:
                label_list = f.read().split('\n')
            img_paths = []
            label = []
            pid_container = set()
            for name_label in label_list:
                # img_paths.append(osp.join(self.dataset_dir,name_label.split(' ')[0]))  ### to do: train/image/*.png
                if 'train' in name_label:
                    img_paths.append(osp.join(dir_path, 'image', name_label.split(' ')[0].split('/')[-1]))
                else:
                    img_paths.append(osp.join(self.dataset_dir,name_label.split(' ')[0]))
                # if 'query' in name_label or 'gallery' in name_label:
                #     img_paths.append(osp.join(self.dataset_dir,name_label.split(' ')[0]))
                # else:
                #     img_paths.append(osp.join(dir_path ,name_label.split(' ')[0].split('/')[1]))
                # img_paths.append(osp.join(dir_path ,name_label.split(' ')[0].split('/')[1]))
                label.append(int(name_label.split(' ')[1]))
                pid_container.add(int(name_label.split(' ')[1]))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            # import pdb
            # pdb.set_trace()
            for i in range(len(img_paths)):
                pid = label[i]
                if relabel: pid = pid2label[pid]
                img_name = img_paths[i].split('/')[-1]
                if label_list_name.find('query') >= 0:
                    dataset.append((img_paths[i], img_name))
                else:
                    dataset.append((img_paths[i], pid))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.png'))
            for img_path in img_paths:
                img_name = img_path.split('/')[-1]
                dataset.append((img_path, img_name))

        return dataset

dataset = NAIC()
# print(dataset)