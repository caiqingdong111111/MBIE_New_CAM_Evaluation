import os
import numpy as np
from PIL import Image

if __name__ == '__main__':

    ground_path = r'data\VOC2012\SegmentationClassAug' # ground truth mask folder
    trainlist = r'data\VOC2012\train.txt'
    cam_path = r'CAM_result_samples' # CAM results folder
    img_name_list = open(trainlist).read().splitlines()

    wholelabel = np.load(r'data\VOC2012\cls_labels.npy', allow_pickle=True).item()
    # the cls_labels.npy file records the categories associated with each image.

    class_count = np.zeros((20))
    class_backcount = np.zeros((20))
    class_confidence_value = np.zeros((20))
    class_back_confidence_value = np.zeros((20))

    for name in img_name_list:
        # read prediction cam
        predict_file = os.path.join(cam_path, '%s.npy' % name)
        predict_dict = np.load(predict_file, allow_pickle=True).item()

        gt_file = os.path.join(ground_path, '%s.png' % name)
        gt = np.array(Image.open(gt_file))
        gt_map = gt.flatten()
        gt_label = np.unique(gt_map)
        gt_label = np.delete(gt_label, np.where(gt_label == 255))

        class_index = {}
        for u_label in gt_label:
            class_index[u_label] = np.where(gt_map == u_label)[0]

        for key in predict_dict.keys():

            if (key + 1) in gt_label:
                cammap = predict_dict[key].flatten()
                thevalue = cammap[class_index[key + 1]].sum() / len(class_index[key + 1])
                class_confidence_value[key] += thevalue
                class_count[key] += 1
            if 0 in gt_label:
                class_back_confidence_value[key] += cammap[class_index[0]].sum() / len(class_index[0])
                class_backcount[key] += 1

    MBIEc = class_confidence_value / class_count
    MBIEcb = class_back_confidence_value / class_backcount
    # MBIEcb calculates the leakage activation values for a specific category,
    # thus it's counted based on the frequency of occurrences of a particular analogy.
    # This shares the same logic as MBIEc. Moreover, from the perspective of the category activation map,
    # pixels at the same spatial location have different values based on different categories.

    print('MBIEc', MBIEc)
    print('mMBIEc', MBIEc.mean())
    print('MBIEcb', MBIEcb.mean())
    print('MBIE', max(MBIEc.mean() - MBIEcb.mean(),0))
