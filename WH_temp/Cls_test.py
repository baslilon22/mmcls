import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from mmcls.apis import init_model, inference_model

from eval import ConfusionMatrix



def save_dict(dict_obj, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        return pickle.load(f)

def get_images_names(root):
    """支持图片的格式"""
    pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
    filelist = [f for f in os.listdir(root) if os.path.splitext(f)[-1] in pic_form]
    return filelist

def inference(model, img_dir):
    # 正样本类推理结果
    classifier_sku_list = model.CLASSES
    preds_List, labels_List = list(), list()
    labels = os.listdir(img_dir)
    for label in labels:
        imgs_path = os.path.join(img_dir, label)
        if not os.path.isdir(imgs_path):
            continue
        imgs_list = get_images_names(imgs_path)
        # 获取label索引
        label_i = classifier_sku_list.index(label)
        for filename in tqdm(imgs_list):
            img_path = os.path.join(imgs_path, filename)
            pred_label = inference_model(model, img_path)['pred_label']
            preds_List.append(pred_label)
            labels_List.append(label_i)
    preds = np.array(preds_List)
    labels_i = np.array(labels_List)
    return {'preds':preds, 'labels':labels_i}



def main():
    config_file = r'./work_dirs/swin_tiny/Experiment_negative/withNegative/jml_improve.py'
    checkpoint_file = r'./work_dirs//swin_tiny/Experiment_negative/withNegative/epoch_150.pth'
    test_dir = r'/data4/lj/Classification/1.7/WH_temp/test_data/test_withoutOther'
    out_dir = r'/data4/lj/Classification/1.7/WH_temp'
    device = 'cuda:4'
    model = init_model(config_file, checkpoint=checkpoint_file, device=device)
    
    # 测试:
    try:
        test_res = load_dict(os.path.join(out_dir, 'Cls_res.pkl'))
        print('---- Load Test result ----')
    except:
        test_res = inference(model, test_dir)
        save_dict(test_res, os.path.join(out_dir, 'Cls_res.pkl'))
        print('---- Get and Save Test result ----')
    preds, labels = test_res['preds'], test_res['labels']
    
    Mat = ConfusionMatrix.calculate(preds, labels, num_classes=len(model.CLASSES))
    print('混淆矩阵:', Mat.shape)
    print(Mat)
    
    
    # 测试集gt不包含others类, pd中的others视为negative
    others_i = model.CLASSES.index('yl_jml_others')
    index = model.CLASSES.index('negative')
    Mat[:, index] += Mat[:, others_i]
    Mat[:, others_i] = 0

    # ID指标:
    index = model.CLASSES.index('negative')
    ood_gts = Mat[index, :].sum()
    ood_pds = Mat[:, index].sum()
    id_gts = Mat.sum() - ood_gts
    
    acc = (Mat.trace() - Mat[index][index]) / id_gts
    tpr = 1 - (ood_pds - Mat[index][index]) / id_gts
    fpr = 1 - Mat[index][index] / ood_gts
    print('正样本数量: {}, acc:{}'.format(id_gts, acc))
    print('tpr:{}  fpr:{}'.format(tpr, fpr))
    
    
    # if index < Mat.shape[0] - 1:
    #     ID_Mat = torch.cat((Mat[0: index], Mat[index+1: ]), dim=0)
    # else:
    #     ID_Mat = Mat[0: index]
    # acc = (Mat.trace() - Mat[index][index]) / ID_Mat.sum()
    # print('ID 样本数量:{} acc:{}'.format(ID_Mat.sum(), acc))
    # tpr = (ID_Mat.sum() - ID_Mat[:, index].sum()) / ID_Mat.sum()
    # fpr = (Mat[index, :].sum() - Mat[index][index]) / Mat[index, :].sum()
    # print('tpr:{}  fpr:{}'.format(tpr, fpr))
    
    

if __name__ == '__main__':
    main()