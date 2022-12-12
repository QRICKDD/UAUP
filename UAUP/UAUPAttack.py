import sys

sys.path.append("..")
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
# from model_DBnet.pred_single import *
import AllConfig.GConfig as GConfig
from Tools.ImageIO import img_read, img_tensortocv2
from Tools.Baseimagetool import *
import random
from UAUP.Auxiliary import *
from Tools.Log import logger_config
import datetime
from Tools.EvalTool import DetectionIoUEvaluator,read_txt
from mmocr.utils.ocr import MMOCR
from PIL import Image
import numpy as np
from torchvision import transforms

to_tensor = transforms.ToTensor()


class RepeatAdvPatch_Attack():
    def __init__(self,
                 data_root, savedir, log_name,
                 eps=100 / 255, alpha=1 / 255, decay=1.0,
                 T=200, batch_size=8,
                 adv_patch_size=(1, 3, 100, 100), gap=20, mml_weight=1.0, dl_weight=1.0,
                 feamapLoss=False):

        # load DBnet
        self.ocr = MMOCR(det='DB_r18', recog=None)
        # hyper-parameters
        self.eps, self.alpha, self.decay = eps, alpha, decay

        # train settings
        self.T = T
        self.batch_size = batch_size

        # Loss
        self.feamapLoss = feamapLoss
        self.loss = nn.MSELoss()
        self.mml_weight, self.dl_weight = mml_weight, dl_weight

        # path process
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.train_dataset = init_train_dataset(data_root)
        self.test_dataset = init_test_dataset(data_root)

        # all gap
        self.shufflegap = len(self.train_dataset) // self.batch_size
        self.gap = gap

        # initiation
        self.adv_patch = torch.zeros(list(adv_patch_size))
        self.start_epoch = 1
        # recover adv patch
        recover_adv_path, recover_t = self.recover_adv_patch()
        if recover_adv_path != None:
            self.adv_patch = recover_adv_path
            self.t = recover_t


        self.logger = logger_config(log_filename=log_name)
        self.evaluator=DetectionIoUEvaluator()

    def recover_adv_patch(self):
        temp_save_path = os.path.join(self.savedir, "advtorch")
        if os.path.exists(temp_save_path):
            files = os.listdir(temp_save_path)
            if len(files) == 0:
                return None, None
            files = sorted(files, key=lambda x: int(x.split('.')[0].split("_")[-1]))
            t = int(files[-1].split("_")[-1])
            keyfile = os.path.join(temp_save_path, files[-1])
            return torch.load(keyfile), t
        return None, None

    def get_merge_image_list(self, patch: torch.Tensor, mask_list: list,
                             image_list: list):
        adv_image_list = []
        for mask, image in zip(mask_list, image_list):
            h,w=image.shape[2:]
            repeat_patch = repeat_4D(patch=patch, h_real=h, w_real=w)
            adv_image_list.append(image*(1-mask) + repeat_patch*mask)
        return adv_image_list


    def MultiMiddleLoss(self,feamap1,feamap2,x):
        loss = 0
        for f1,f2 in zip(feamap1,feamap2):
            loss += self.loss(f1.mean([2,3]),f1.mean([2,3]))/torch.norm(f1.mean([2,3])-f1.mean([2,3]),p=1,dim=(0,1))

        grad = torch.autograd.grad(loss,x,retain_graph=False)
        return grad,loss

    def DetectionLoss(self,mask,x1,x):
        loss = torch.norm(x1*mask,p=2)
        grad = torch.autograd.grad(loss, x, retain_graph=False)
        return grad,loss

    # 快捷初始化
    def inner_init_adv_patch_image(self, mask, image, hw, device):
        adv_patch = self.adv_patch.clone().detach()
        adv_patch = adv_patch.to(device)
        adv_patch.requires_grad = True
        image = image.to(device)
        adv_image = self.get_merge_image(adv_patch, mask=mask,
                                         image=image, hw=hw, device=device)
        return adv_patch, adv_image

    def train(self):
        momentum = 0

        print("start training-====================")
        shuff_ti = 0  # train_dataset_iter
        for t in range(self.t + 1, self.T):
            print("iter: ", t)
            if t % self.shufflegap == 0:
                random.shuffle(self.train_dataset)
                shuff_ti = 0
            batch_dataset = self.train_dataset[shuff_ti * self.batch_size: (shuff_ti + 1) * self.batch_size]
            shuff_ti += 1

            batch_mmLoss = 0
            batch_dLoss = 0
            sum_grad = torch.zeros_like(self.adv_patch)
            for [image, mask, gtpath] in batch_dataset:
                it_adv_patch=self.adv_patch.clone().detach().to(GConfig.DB_device)
                it_adv_patch.requires_grad=True
                image = image.to(GConfig.DB_device)
                mask = mask.to(GConfig.DB_device)
                image_d1, mask_d1 = Diverse_module_1(image, mask, t, self.gap)
                mask_t = extract_background(image_d1)  # character region
                h, w = image_d1.shape[2:]
                UAU = repeat_4D(patch=it_adv_patch, h_real=h, w_real=w)
                merge_image_d1 = UAU * mask_t + image * (1 - mask_t)
                image_d2, mask_d2, UAU_d2 = Diverse_module_2(image=merge_image_d1, mask=mask_d1, UAU=UAU,
                                                             now_ti=t, gap=self.gap)
                mmgrad, mmLoss = self.MultiMiddleLoss() * self.mml_weight  # need input tensor.clone()!!!!!
                batch_mmLoss += mmLoss
                sum_grad += mmgrad
                del (mmLoss)

                dlgrad, dLoss = self.DetectionLoss() * self.dl_weight  # need input tensor.clone()!!!!!
                batch_dLoss += dLoss
                sum_grad += dlgrad
                del (dLoss, image_d2, mask_d2, UAU_d2, merge_image_d1, UAU)

                # torch.cuda.empty_cache()

            # update grad
            grad = sum_grad / torch.mean(torch.abs(sum_grad), dim=(1), keepdim=True)  # 有待考证
            grad = grad + momentum * self.decay
            momentum = grad
            # update self.adv_patch
            temp_patch = self.adv_patch.clone().detach().cpu() + self.alpha * grad.sign()
            temp_patch = torch.clamp(temp_patch, min=-self.eps, max=0)
            self.adv_patch = temp_patch

            # update logger
            e = "iter:{}, batch_loss==mmLoss:{},dloss:{}===".format(t, batch_mmLoss / self.batch_size,
                                                                    batch_dLoss / self.batch_size)
            self.logger.info(e)

            # save adv_patch with
            temp_save_path = os.path.join(self.savedir, "advpatch")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            save_adv_patch_img(self.adv_patch + 1, os.path.join(temp_save_path, "advpatch_{}.png".format(t)))
            temp_torch_save_path = os.path.join(self.savedir, "advtorch")
            if os.path.exists(temp_torch_save_path) == False:
                os.makedirs(temp_torch_save_path)
            torch.save(self.adv_patch, os.path.join(temp_torch_save_path, "advpatch_{}".format(t)))

            # 保存epoch结果
            if t != 0 and t % 10 == 0:
                self.evauate_test_path(t)

    def evaluate_db_draw(self, adv_images, path, t):
        adv_images_cuda = self.list_to_cuda(adv_images, device)
        db_results = []
        for img in adv_images_cuda:
            with torch.no_grad():
                img = [tensor2np(img)]
                boxes, feamap = self.ocr.output(img,
                                               output='./', export='./',
                                               feaName=["neck.lateral_convs.0","neck.lateral_convs.1",
                                                        "neck.lateral_convs.2","neck.lateral_convs.3",
                                                        "bbox_head.binarize.7"])
                preds = feamap[-1]

            db_results.append(preds)
        hw_lists = get_image_hw(adv_images)
        i = 0
        save_name = "DB_{}_{}.jpg"
        for adv_image, pred, [h, w] in zip(adv_images, db_results, hw_lists):
            _, boxes = get_DB_dilateds_boxes(pred, h, w)
            adv_image_cv2 = img_tensortocv2(adv_image)
            DB_draw_box(adv_image_cv2, boxes=boxes, save_path=os.path.join(path, save_name.format(epoch, i)))
            i += 1
        P=1.0
        R=1.0
        F=1.0
        return P,R,F

    def evauate_test_path(self, t):
        save_dir_db = os.path.join(self.savedir, "eval", "orgin")
        if os.path.exists(save_dir_db) == False:
            os.makedirs(save_dir_db)
        save_resize_dir = os.path.join(self.savedir, "eval", "resize",str(t))
        if os.path.exists(save_resize_dir) == False:
            os.makedirs(save_resize_dir)

        resize_scales = [item / 10 for item in range(6, 21, 1)]

        hw_s = get_image_hw(self.test_images)
        mask_s = self.get_image_backgroud_mask(self.test_images)
        adv_images = self.get_merge_image_list(self.adv_patch.clone().detach().cpu(),
                                               mask_s, self.test_images, hw_s)
        self.evaluate_db_draw(adv_images=adv_images, path=save_dir_db, t=t)
        resize_adv_images = get_random_resize_image(adv_image_lists=adv_images, low=0.4, high=3)
        for re_sclae in resize_scales:
            P,R,F=self.evaluate_db_draw(resize_adv_images,os.path.join(save_resize_dir,str(re_sclae)),t)
            e="iter:{},scale:{},"
            self.logger.info()
        del (resize_adv_images, adv_images, hw_s, mask_s)


def tensor2np(img):
    return np.array(255*img.transpose(0,1).transpose(1,2),dtype=np.uint8)

if __name__ == '__main__':
    RAT = RepeatAdvPatch_Attack(data_root="../AllData/Data",
                                savedir='../result_save/150_150', log_name='150_150.log',
                                alpha=1 / 255, batch_size=20, gap=20, T=201,
                                mml_weight=1.0, dl_weight=1.0, eps=60 / 255, decay=1.0,
                                adv_patch_size=(1, 3, 150, 150),
                                feamapLoss=False)
    RAT.train()

