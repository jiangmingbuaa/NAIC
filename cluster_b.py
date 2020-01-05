import numpy as np
import json
from PIL import Image

if __name__=='__main__':
    with open('/tmp/data/model/NAIC_baseline_submission_2b.json', 'r') as f:
        result = json.load(f)
    
    # main()
    # img_path_query = '/root/share/dataset/reid/NAIC2/query_b/'
    # img_path_gallery = '/root/share/dataset/reid/NAIC2/gallery_b/'
    qf = np.load('/tmp/data/model/qf.npy')
    q_pids = np.load('/tmp/data/model/q_pids.npy')
    g_pids = np.load('/tmp/data/model/g_pids.npy')

    dist = np.dot(qf,qf.T)
    for i in range(dist.shape[0]):
        dist[i,i] = 0
    thre=0.6
    class_img = [i for i in range(qf.shape[0])]
    for i in range(qf.shape[0]):
        for j in range(i):
            if dist[i][j]>thre:
                class_img[j]=class_img[i]
    
    count={}
    for i in class_img:
        if not i in count:
            count[i]=1
        else:
            count[i]+=1
    print(np.unique(class_img).shape) 

    # import pdb
    # pdb.set_trace()

    top_k=3
    img_all=[]

    with open('/tmp/data/train/label/train_list.txt', 'r') as f:
        label_list = f.read().split('\n')
    for name_label in label_list:
        img_name, img_label = name_label.split(' ')
        img_all.append((img_name,img_label))
    
    for i,img_name in enumerate(q_pids):
        label = class_img[i]+10001
        # img_path = img_path_query+img_name
        # img = Image.open(img_path)
        # if np.array(img).mean()>=100:
        #     continue
        if count[class_img[i]]>1:
            continue
        img_all.append(('test/query_B/'+img_name,label))
        # img_all_val.append((img_path,label))
        for k in range(top_k):
            # img_path = img_path_gallery+result[img_name][k]
            # img = Image.open(img_path)
            img_all.append(('test/gallery_B/'+result[img_name][k],label))

    print(len(img_all))

    with open('/tmp/data/train/label/train_list.txt', "w+") as f:
        for item in img_all:
            f.write(item[0]+' '+str(item[1])+"\n")
