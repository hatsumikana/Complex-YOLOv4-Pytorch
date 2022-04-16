import argparse
import os
import time
import numpy as np
import sys
import warnings

from utils.visualization_utils import show_image_with_boxes

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')

from data_process.kitti_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2, rescale_boxes
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from data_process import kitti_data_utils, kitti_bev_utils
import config.kitti_config as cnf
import cv2

def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()

    # print model summary
    # model.print_network()

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            img_paths, imgs, targets = batch_data
            # Extract labels
            labels = targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)

            outputs = model(imgs)
            print('outputs: ', outputs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)

            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
            
            # pedestrain = 1.0
            # cars = 0.0
            # cyclist = 2.0
            
            if 1.0 in labels:
                index = list(ap_class).index(1)
                
                if precision[index] < 0.8:
                    print(f"\n-----Detected pedestrians poorly for {img_paths[0]}-----\n")
                    img_rgb = cv2.imread(img_paths[0])
                    
                    detections = outputs
                    img_bev = imgs
                    
                    img_bev = img_bev.squeeze() * 255
                    img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
                    img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

                    # Rescale boxes to original image
                    for detect in detections:
                        detect = rescale_boxes(detect, configs.img_size, img_bev.shape[:2])
                        for x, y, w, l, im, re, *_, cls_pred in detect:
                            yaw = np.arctan2(im, re)
                            # Draw rotated box
                            kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])
                    
                    calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
                    objects_pred = predictions_to_kitti_format(detections, calib, img_rgb.shape, configs.img_size)

                    filename = img_paths[0].split("\\")[-1]
                    file_ls = filename.split("/")
                    file_ls[-2] = "pred"
                    file_ls[-1] = filename[-10:-4] + ".txt"
                    pred_path = "/".join(file_ls)
                    pred_img_path = "/".join(file_ls[:-2]) + "/pred_images/"
                    
                    if configs.save_pred == True:
                        for obj in objects_pred:
                            obj_str = obj.to_kitti_format()
                            print(obj_str)

                            with open(pred_path, "a+") as f:
                                f.write(obj_str)
                                f.write("\n")

                    img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)
                    
                    if configs.show_image == True:
                        cv2.imshow(f"bev_{filename}", img_bev)
                        cv2.imshow(f"rgb_{filename}", img_rgb)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                    
                    if configs.save_image == True:
                        cv2.imwrite(f"{pred_img_path}bev_{file_ls[-1]}.png", img_bev)
                        cv2.imwrite(f"{pred_img_path}rgb_{file_ls[-1]}.png", img_rgb)

            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

    return precision, recall, AP, f1, ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    
    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--show_image', type=str, default=True,
                        help='True or False')
    parser.add_argument('--save_image', type=str, default=False,
                        help='True or False')
    parser.add_argument('--save_pred', type=str, default=False,
                        help='True or False')

    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    return configs


if __name__ == '__main__':
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)

    model = create_model(configs)
    # model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    model.load_state_dict(torch.load(configs.pretrained_path, map_location=configs.device))

    model = model.to(device=configs.device)

    model.eval()
    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {}\n".format(AP.mean()))
