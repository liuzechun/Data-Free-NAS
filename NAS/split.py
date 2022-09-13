import os
import random
import shutil

path = '/home/data/synthesized_images'
output_dir1 = '/home/data/val'
output_dir2 = '/home/data/train'

num_in1 = 8

files = os.listdir(path)
for file in files:
    f = path+"/"+file
    f1 = os.listdir(f)
    random.shuffle(f1)
    out_list1 = f1[:num_in1]
    out_list2 = f1[num_in1:]
    output1 = output_dir1+"/"+file
    output2 = output_dir2+"/"+file
    if not os.path.exists(output1):
        os.makedirs(output1)
    for i in range(len(out_list1)):
        src_filename = f +'/' + out_list1[i]
        dst_filename = output1 +'/' + out_list1[i]
        shutil.copyfile(src_filename, dst_filename)

    if not os.path.exists(output2):
        os.makedirs(output2)
    for i in range(len(out_list2)):
        src_filename = f +'/' + out_list2[i]
        dst_filename = output2 +'/' + out_list2[i]
        shutil.copyfile(src_filename, dst_filename)

