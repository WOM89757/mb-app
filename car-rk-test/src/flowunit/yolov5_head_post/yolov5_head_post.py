#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import _flowunit as modelbox
import numpy as np 
import json 
import cv2 

from yolov5_utils import *

class Yolov5Post(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.net_h = config.get_int('net_h', 416)
        self.net_w = config.get_int('net_w', 416)
        self.num_classes = config.get_int('num_classes', 80)
        self.num_grids = int((self.net_h / 32) * (self.net_w / 32)) * (1 + 2*2 + 4*4)
        self.conf_thre = config.get_float('conf_threshold', 0.3)
        self.nms_thre = config.get_float('iou_threshold', 0.4)
        self.anchors = {
            "in_feat": int(self.net_h/8),
            "in_feat2":int(self.net_h/16),
            "in_feat3":int(self.net_h/32),
        }

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # in_image = data_context.input("in_image")
        in_feat = data_context.input("in_feat")
        in_feat2 = data_context.input("in_feat2")
        in_feat3 = data_context.input("in_feat3")
        in_image = data_context.input("in_image")

        out_data = data_context.output("out_data")
        modelbox.debug("------yolov5 head post image size {}".format(in_image.size()))
        modelbox.debug("------yolov5 head post feat size {}".format(in_feat.size()))
        i = 0
        nbboxes = np.array([])
        nbboxes_cla = np.array([])
        nbboxes_scores = np.array([])
        nbboxes_num = np.array([])
        for buffer_feat, buffer_feat2, buffer_feat3, buffer_img in zip(in_feat, in_feat2, in_feat3, in_image):
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')
            modelbox.debug("get frame shape {} {} {}".format(channel, width, height))
            # # modelbox.info("get frame index: {}".format(frame_index))
            data = buffer_img.as_object()
            modelbox.debug("---buffer img type: {}".format(type(data)))
            img_data = np.array(buffer_img.as_object(), copy=False)
            # # print(img_data.shape)

            # # img_data = img_data.reshape((416, 416, channel))
            
            modelbox.debug("--------head post img shape {}", img_data.shape)
            feat_data = np.array(buffer_feat.as_object(), copy=False)
            feat_data2 = np.array(buffer_feat2.as_object(), copy=False)
            feat_data3 = np.array(buffer_feat3.as_object(), copy=False)
            modelbox.debug("--------feat data shape {}", feat_data.shape)
            modelbox.debug("--------feat2 data shape {}", feat_data2.shape)
            modelbox.debug("--------feat3 data shape {}", feat_data3.shape)
            # modelbox.info("-------- feat.shape[-2:]{}", feat_data.shape[-2:])
            feat_data = feat_data.reshape((3, self.anchors["in_feat"],self.anchors["in_feat"], self.num_classes + 5))
            feat_data2 = feat_data2.reshape((3, self.anchors["in_feat2"],self.anchors["in_feat2"], self.num_classes + 5))
            feat_data3 = feat_data3.reshape((3, self.anchors["in_feat3"],self.anchors["in_feat3"], self.num_classes + 5))
            # feat_data = feat_data.reshape((self.num_classes + 5, self.num_grids)).transpose()
            # feat_data = feat_data.reshape([3, -1]+list(feat_data.shape[-2:]))
            modelbox.debug("--------feat data reshap  {}", feat_data.shape)
            modelbox.debug("--------feat2 data reshap  {}", feat_data2.shape)
            modelbox.debug("--------feat3 data reshap  {}", feat_data3.shape)

            input_data = list()
            input_data.append(np.transpose(feat_data, (1, 2, 0, 3)))
            input_data.append(np.transpose(feat_data2, (1, 2, 0, 3)))
            input_data.append(np.transpose(feat_data3, (1, 2, 0, 3)))
            
            # input_data.append(np.transpose(feat_data, (0, 1, 2, 3)))
            # input_data.append(np.transpose(feat_data, (0, 1, 2, 3)))
            # input_data.append(np.transpose(feat_data, (0, 1, 2, 3)))
            # modelbox.info("--------input data reshap  {}", input_data[0].shape)

            bboxes, classes, scores = yolov5_post_process(input_data, self.conf_thre, self.nms_thre, self.net_h)
            
            # infer_data = "head result"
            # add_buffer = modelbox.Buffer(self.get_bind_device(), infer_data)

            ratio_w = self.net_w/width
            ratio_h = self.net_h/height

            # ratio = min(self.net_h/height, self.net_w/width)
            if bboxes is not None:
                _scores_pos = np.where(scores >= self.conf_thre)
                bboxes = bboxes[_scores_pos]
                classes = classes[_scores_pos]
                scores = scores[_scores_pos]

                nbboxes_num = np.append(nbboxes_num, len(bboxes))
                for box in bboxes:
                    box[np.where(box<0)] = 0
                    # xyxy
                    box[0] = box[0] / ratio_w
                    box[1] = box[1] / ratio_h
                    box[2] = box[2] / ratio_w
                    box[3] = box[3] / ratio_h
                # bboxes  = np.append(bboxes,bboxes, axis=0)
                modelbox.debug("--------head boxes size: {} ".format(len(bboxes)))
                # img_data = img_data.reshape((height, width, channel))

                # bboxes = bboxes / ratio
                modelbox.debug(" \n{}".format(bboxes))
                # img_out = draw(img_data, bboxes, scores, classes)
                # cv2.imwrite("/home/wm/code/car-rk-test/src/flowunit/yolov5_head_post/t1-{}.jpg".format(i), img_out)
                i = i+1
                # add_buffer = modelbox.Buffer(self.get_bind_device(), img_out)
                # add_buffer.copy_meta(buffer_img)
                # out_data.push_back(add_buffer)

                bboxes = bboxes.astype(int)
                nbboxes = np.append(nbboxes, bboxes)
                nbboxes_cla = np.append(nbboxes_cla, classes)
                nbboxes_scores = np.append(nbboxes_scores, scores)
                # add_buffer.set("bboxes", bboxes.flatten().tolist())
                # out_data.push_back(add_buffer)
            else:
                nbboxes_num = np.append(nbboxes_num, 0)
                modelbox.debug("--------head boxes size: 0 ")
                # bboxes = np.array([]).astype(int)
                # add_buffer.set("bboxes", "null")
                # out_data.push_back(add_buffer)
            
            # out_data.push_back(buffer_img)
            # continue



            # ratio = min(self.net_h / height, self.net_w / width)
            # bboxes = postprocess(feat_data, (self.net_h, self.net_w), self.num_classes, self.conf_thre, self.nms_thre, ratio)
            # if bboxes is not None:
            #     img_out = draw_bbox(img_data, bboxes)
            #     add_buffer = modelbox.Buffer(self.get_bind_device(), img_out)
            #     add_buffer.copy_meta(buffer_img)
            #     out_data.push_back(add_buffer)
            # else:
            #     out_data.push_back(buffer_img)
        
        infer_data = "head result"
        add_buffer = modelbox.Buffer(self.get_bind_device(), infer_data)
        nbboxes = nbboxes.astype(int)
        add_buffer.set("bboxes", nbboxes)
        add_buffer.set("bboxes_num", nbboxes_num)
        add_buffer.set("bboxes_classes", nbboxes_cla)
        add_buffer.set("bboxes_scores", nbboxes_scores)
        
        out_data.push_back(add_buffer)
        modelbox.debug("head post nbboxes num: {} \n {}".format(len(nbboxes_num), nbboxes_num))
        modelbox.debug("head post nbboxes size: {} \n {}".format(len(nbboxes), nbboxes))
        modelbox.debug("head post out buffer size: {}".format(out_data.size()))
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
