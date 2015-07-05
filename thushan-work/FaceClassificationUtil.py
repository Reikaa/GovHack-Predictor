__author__ = 'Thushan Ganegedara'

import shutil as sh
import os
import csv
from PIL import Image
import numpy as np
import pickle
import scipy
class FaceClassifUtil(object):

    def load_data_to_one_folder(self,data_dir):
        sub_dir_names = [x[0] for x in os.walk(data_dir)]
        for sub_d in sub_dir_names[1:]:
            f = []
            for (dirpath, dirnames, filenames) in os.walk(sub_d):
                f.extend(filenames)
                break
            for f_name in f:
                src = sub_d + '\\' + f_name
                dst = 'all_images\\' + f_name
                if '.jpg' in f_name:
                    sh.copyfile(src, dst)

    def load_data(self,img_dir,lbl_dir,file_name):

        def get_thumbnail(filename,size):
            try:
                im = Image.open(filename)
                im.thumbnail(size, Image.ANTIALIAS)
                img = im.convert('L')
                return img
            except IOError:
                print "cannot create thumbnail for '%s'" % filename

            return None

        def get_label(age,gender):
            age_val = 0
            gender_val = 0
            if age == '(0 2)':
                age_val=0
            elif age == '(4 6)':
                age_val=1
            elif age == '(8 13)':
                age_val = 2
            elif age == '(15 20)':
                age_val = 3
            elif age == '(25 32)':
                age_val = 4
            elif age == '(38 43)':
                age_val = 5
            elif age == '(48 53)':
                age_val = 6
            elif age == '(60 100)':
                age_val = 7

            if gender == 'f':
                gender_val = 0
            elif gender == 'm':
                gender_val = 1

            return gender_val*8 + age_val

        with open(lbl_dir+'\\'+file_name) as inputfile:
            results = list(csv.reader(inputfile))

        i_side = 48
        f_names = []
        f_ids = []
        labels = []
        imgs = []
        i = 0
        for s in results[1:]:

            if len(s)==2:
                new_s = s[0]+s[1]
            else:
                new_s = s[0]
            s_tokens = new_s.split('\t')

            if 'None' in s_tokens:
                continue

            f_name = s_tokens[1]
            f_id = s_tokens[2]
            img1 = get_thumbnail(img_dir+"\\"+'coarse_tilt_aligned_face.'+f_id+'.'+f_name,[i_side,i_side])
            if img1 is not None:
                arr = np.asarray(list(img1.getdata()))
                arr = arr/255.
                if arr.size<i_side**2:
                    gap = i_side**2 - arr.size
                    gap_arr = np.zeros((gap),dtype=np.float32)
                    arr = np.concatenate((arr,gap_arr))

                imgs.append(arr)
                f_names.append(s_tokens[1])
                f_ids.append(s_tokens[2])
                label = get_label(s_tokens[3],s_tokens[4])
                labels.append(label)


                #if i == 30:
                #    im = Image.fromarray(np.reshape(arr*255.,(64,64)))
                #    im = im.convert('RGB')
                #    im.save('test.jpg')

            else:
                continue

        return [imgs,labels]

if __name__ == '__main__':
    util = FaceClassifUtil()
    #util.load_data_to_one_folder('faces')

    f_names = ['fold_0_data','fold_1_data','fold_2_data','fold_3_data','fold_4_data']
    for f in f_names:
        [imgs,labels] = util.load_data("all_images","labels",f+'.txt')
        pickle.dump([imgs,labels], open(f+".pkl", "wb"))
        print 'done ' + f
