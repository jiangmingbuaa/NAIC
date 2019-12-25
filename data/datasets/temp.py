import os.path as osp
import glob

def main():
    data_dir='/root/share/dataset/reid/NAIC2/'
    img_paths = glob.glob(osp.join(data_dir, 'query_a/*.png'))
    label=10000
    query_list=[]
    for img_path in img_paths:
        # import pdb
        # pdb.set_trace()
        img_name = '/'.join(img_path.split('/')[-2:])
        query_list.append(img_name+' '+str(label))
        label+=1

    print(len(query_list))
    with open(osp.join(data_dir, 'query_a_list.txt'), "w+") as f:
        for item in query_list:
            f.write(item+"\n")

if __name__=='__main__':
    main()