import os.path as osp
import random

def main():
    dir_path = '/home1/lizihan/reID/data/NAIC/train_set'
    file_dir = '/home/jiangming/re-id/NAIC/reid_strong_baseline/data/datasets/'
    train_file = osp.join(file_dir, 'train_list.txt')
    label_list = []
    with open(train_file, 'r') as f:
        label_list = f.read().split('\n')
    img_paths = []
    label = []
    pid_container = set()
    for name_label in label_list:
        img_paths.append(osp.join(dir_path ,name_label.split(' ')[0].split('/')[1]))
        label.append(int(name_label.split(' ')[1]))
        pid_container.add(int(name_label.split(' ')[1]))
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    print('%d images\n'%len(label))
    count=[0]*len(pid_container)
    for i in label:
        count[i]+=1
    # large=[]
    # small=[]
    # for i,c in enumerate(count):
    #     if c>100:
    #         large.append([i,c])
    #     if c<2:
    #         small.append([i,c])
    train_list=[]
    query_list=[]
    gallery_list=[]
    last_label=0
    person=[]
    # import pdb
    # pdb.set_trace()
    for i in range(len(img_paths)):
        img = img_paths[i]
        l = label[i]
        if l != last_label:
            person_num = len(person)
            # import pdb
            # pdb.set_trace()
            if ((l-1) in [112, 1161, 514, 968, 760]) or ((l-1)%20==0 and person_num>1) or ((l-1)>3726 and (l-1)%10==0):
                index = [x for x in range(person_num)]
                random.shuffle(index)
                query_num = max(1, int(person_num*0.1))
                for i,idx in enumerate(index):
                    if i<query_num:
                        query_list.append(person[idx])
                    else:
                        gallery_list.append(person[idx])
            elif 1731<=(l-1)<=3726 and (l-1)%2==0:
                for p in person:
                    gallery_list.append(p)
            else:
                for p in person:
                    train_list.append(p)
            person = []
            last_label = l
        person.append(img+' '+str(l))
    
    print('%d training, %d query, %d gallery\n'%(len(train_list), len(query_list), len(gallery_list)))

    with open('train_val_list.txt', "w+") as f:
                for item in train_list:
                    f.write(item+"\n")
    with open('query_a_val_list.txt', "w+") as f:
                for item in query_list:
                    f.write(item+"\n")
    with open('gallery_a_val_list.txt', "w+") as f:
                for item in gallery_list:
                    f.write(item+"\n")

    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    main()
    # extend_gallery = []
    # dir_path = '/home1/lizihan/reID/data/NAIC/'
    # with open('query_a_list.txt', 'r') as f:
    #     label_list = f.read().split('\n')
    # for name_label in label_list:
    #     img = osp.join(dir_path ,name_label.split(' ')[0])
    #     label = name_label.split(' ')[1]
    #     extend_gallery.append(img+' '+label)
    # with open('extend_gallery.txt', "w+") as f:
    #             for item in extend_gallery:
    #                 f.write(item+"\n")

    # import glob
    # gallery_all = []
    # dir_path = '/home1/lizihan/reID/data/NAIC/gallery_a/'
    # img_paths = glob.glob(osp.join(dir_path, '*.png'))
    # # import pdb
    # # pdb.set_trace()
    # for img_path in img_paths:
    #     gallery_all.append(img_path+' 10000')
    # with open('gallery_all.txt', "w+") as f:
    #             for item in gallery_all:
    #                 f.write(item+"\n")