import torch.nn as nn
import cv2
import utils
import loss_function
import voc0712
import augmentations
import ssd_net_vgg
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import os
from visdom import Visdom
import time
import torch
import Config
import numpy as np
from detection import Detect
import pickle

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = Config.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
def xavier(param):
    nn.init.xavier_uniform_(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def test_net(save_folder,net,testset, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(Config.class_num)]

    det_file = os.path.join(save_folder, 'detections.pkl')
    '''
    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return
    '''
    for i in range(num_images):

        image = testset.pull_image(i)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = x.unsqueeze(0)     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx) 
        softmax = nn.Softmax(dim=-1)
        detect = Detect(Config.class_num, 0, 50, 0.01, 0.45) #构建检测类
        priors = utils.default_prior_box()
        loc,conf = y
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        detections = detect(
            loc.view(loc.size(0), -1, 4),
            softmax(conf.view(conf.size(0), -1,Config.class_num)),
            torch.cat([o.view(-1, 4) for o in priors], 0)
        ).data

        scale = torch.Tensor(image.shape[1::-1]).repeat(2)

        for j in range(detections.size(1)):
            inds = np.where(detections[0,j,:,0].cpu().numpy() > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = (detections[0,j,inds,1:]*scale).cpu().numpy()
            c_scores = detections[0,j,inds,0].cpu().numpy()
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i]=c_dets

        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,detections.size(1))])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1,detections.size(1)):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


    print('Evaluating detections')
    #if args.dataset == 'VOC':
    APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs, mAP
    #else:
        #testset.evaluate_detections(all_boxes, save_folder)