import os
import mmcv
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
from mmcls.apis import init_model

from nnguide import NNGuideOODDetector
from vim import VIMOODDetector
from eval import compute_ood_performances


def inference_model(model, img, ReAct_c=None):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        img = data['img']                                                            # [B, C, H ,W]
        _rawfeas = model.extract_feat(img, stage='pre_logits')                       # [1, 768]
        if ReAct_c is not None:
            _rawfeas = _rawfeas.clip(max=ReAct_c)
        logits = model.head.simple_test(_rawfeas, softmax=False, post_process=False) # [1, 45]
    return _rawfeas, logits

def get_ReAct_c(model, train_dir, react_percentile=0.95):
    fea_List = list()
    labels = os.listdir(train_dir)
    for label in labels:
        imgs_path = os.path.join(train_dir, label)
        if not os.path.isdir(imgs_path) or label == '__pycache__':
            continue
        
        print(label)
        
        imgs_list = get_images_names(imgs_path)
        for filename in tqdm(imgs_list):
            img_path = os.path.join(imgs_path, filename)
            feas, _ = inference_model(model, img_path)
            fea_List.append(feas.detach().cpu())
    
    feas = torch.cat(fea_List, dim=0).numpy()       # [N, 768]
    c = np.quantile(feas, react_percentile)
    print(f"{((feas < c).mean()*100).round(2)}% of the units of train features are less than {c}")
    print(f"ReAct c = {c}")
    return c


def get_model_outputs(model, img_dir, ReAct_c=None):
    if ReAct_c is not None:
        print('==== Apply ReAct to extract features ====')
    num = 0
    
    classifier_sku_list = model.CLASSES
    classifier_sku_list.append('negative')
    fea_List, logit_List, label_List = list(), list(), list()
    labels = os.listdir(img_dir)
    for label in labels:
        imgs_path = os.path.join(img_dir, label)
        if not os.path.isdir(imgs_path) or label == '__pycache__':
            continue
        
        imgs_list = get_images_names(imgs_path)
        # 获取label索引
        label_i = classifier_sku_list.index(label)
        label_i = torch.from_numpy(np.array(label_i)).unsqueeze(0)
        for filename in tqdm(imgs_list):
            img_path = os.path.join(imgs_path, filename)
            feas, logits = inference_model(model, img_path, ReAct_c)
            feas = F.normalize(feas, dim=1)
            fea_List.append(feas.detach().cpu())
            logit_List.append(logits.detach().cpu())
            label_List.append(label_i)
            
            # num += 1
            # if num >= 500:
            #     break
    feas = torch.cat(fea_List, dim=0)       # [N, 768]
    logits = torch.cat(logit_List, dim=0)   # [N, num_classes]
    labels_i = torch.cat(label_List, dim=0) # [N]
    return {"feas": feas, "logits": logits, "labels": labels_i}

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

def main():
    config_file = r'./work_dirs/swin_tiny/Experiment_negative/withoutNegativeOther/jml_improve.py'
    checkpoint_file = r'./work_dirs/swin_tiny/Experiment_negative/withoutNegativeOther/epoch_200.pth'
    train_dir = r'/data4/lj/Dataset/JML/Experiment_negative/train_withoutNegativeOther'  # 限定为目录，按 label_name/*.jpg 存储
    test_dir = r'/data4/lj/Dataset/JML/Experiment_negative/test_withoutNegativeOther'
    out_dir = r'/data4/lj/Classification/1.7/WH_temp'
    device = 'cuda:4'
    model = init_model(config_file, checkpoint=checkpoint_file, device=device)
    
    #c = get_ReAct_c(model, train_dir)
    #c = float(0.37321757674217215)   # withoutNegativeOther
    c = None
    
    # 训练集提取特征
    try:
        train_res = load_dict(os.path.join(out_dir, 'Others_train_res.pkl'))
        print('---- Load Train Dataset Feature ----')
    except:
        train_res = get_model_outputs(model, train_dir, ReAct_c=c)
        save_dict(train_res, os.path.join(out_dir, 'Others_train_res.pkl'))
        print('---- Get and Save Train Dataset Feature ----')
    
    
    # ID提取特征
    try:
        test_res = load_dict(os.path.join(out_dir, 'Others_test_res.pkl'))
        print('---- Load Test Dataset Feature ----')
    except:
        test_res = get_model_outputs(model, test_dir, ReAct_c=c)
        save_dict(test_res, os.path.join(out_dir, 'Others_test_res.pkl'))
        print('---- Get and Save Test Dataset Feature ----')
    

    # OOD提取特征
    try:
        ood_res = load_dict(os.path.join(out_dir, 'Others_negative_res.pkl'))
        print('---- Load ODD Dataset Feature ----')
    except:
        ood_res = get_model_outputs(model, r'/data4/lj/Classification/1.7/WH_temp/test_data/Negative', ReAct_c=c)
        save_dict(ood_res, os.path.join(out_dir, 'Others_negative_res.pkl'))
        print('---- Get and Save ODD Dataset Feature ----')


    # nnguide
    try:
        ood_detector = load_dict(os.path.join(out_dir, 'Others_nnguide_detector.pkl'))
        print('---- Load OOD_Detector ----')
    except:
        ood_detector = NNGuideOODDetector()
        ood_detector.setup(train_res)  # 训练集特征作初始化
        save_dict(ood_detector, os.path.join(out_dir, 'Others_nnguide_detector.pkl'))
        print('---- Get and Save OOD_Detector ----')
    
    # vim
    # try:
    #     ood_detector = load_dict(os.path.join(out_dir, 'vim_detector.pkl'))
    #     print('---- Load OOD_Detector ----')
    # except:
    #     ood_detector = VIMOODDetector()
    #     ood_detector.setup(train_res)  # 训练集特征作初始化
    #     save_dict(ood_detector, os.path.join(out_dir, 'vim_detector.pkl'))
    #     print('---- Get and Save OOD_Detector ----')


    # Inference
    id_scores = ood_detector.infer(test_res)
    print('ID:')
    print('mean: ', id_scores.mean(), 'min: ', min(id_scores), ' max: ', max(id_scores))
    
    ood_scores = ood_detector.infer(ood_res)
    print('OD:')
    print('mean: ', ood_scores.mean(), 'min: ', min(ood_scores), ' max: ', max(ood_scores))
    
    
    
    # OOD测试：
    scores = torch.cat([id_scores, ood_scores], dim=0).numpy()
    detection_labels = torch.cat([torch.ones_like(test_res['labels']), torch.zeros_like(ood_res['labels'])], dim=0).numpy()
    opt_thr = compute_ood_performances(labels=detection_labels, scores=scores)
    
    # ID测试:
    id_logits = test_res['logits']
    preds_id = torch.max(id_logits, dim=-1)[1]
    
    ood_i = id_scores < opt_thr
    preds_id[ood_i] = -1
    
    acc = (preds_id == test_res['labels']).float().mean().numpy()
    print('ID 样本数量:{} acc:{}'.format(preds_id.shape[0], acc))


if __name__ == '__main__':
    main()