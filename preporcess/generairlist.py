import cv2
import os
import numpy as np
traindir='/Disk4/xkp/dataset/AIRS/trainval/train/label'
valdir='/Disk4/xkp/dataset/AIRS/trainval/val/label'
def generatorlist(maskdir):
    list= os.listdir(maskdir)
    list=[name for name in list if 'vis' in name]
    for name in list:

        patht=os.path.join(maskdir,name)
        mask=np.array(cv2.imread(patht,cv2.IMREAD_GRAYSCALE)/255.0,np.int)

        numpix=int(mask.shape[0]*mask.shape[1])
        precent=np.sum(mask)/numpix
        print(name,'   :   ',precent)



if __name__=='__main__':
    generatorlist(valdir)