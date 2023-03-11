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
import cv2
import time

class DrawBoxes(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.car_classes = ['Car', 'Bus', 'Truck', 'Tricycle', 'Motorbike', 'Bicycle', 'Special', 'vehicle_Unknown']
        # self.car_classes = ['Car_Saloon', 'Car_SUV', 'Car_MPV', 'Car_Jeep', 'Car_Sports', 'Car_Taxi', 'Car_Police', 'Bus_Big', 'Bus_Middle',
        #     'Bus_Small', 'Bus_School', 'Bus_Bus', 'Bus_Ambulance', 'Truck_Big','Truck_Van', 'Truck_Engineering', 'Truck_Fueltank', 'Truck_Construction',
        #     'Truck_Fire', 'Truck_Garbage', 'Truck_Watering', 'Tricycle', 'Motorbike', 'Bicycle', 'Special_Military', 
        #     'Special_other', 'vehicle_Unknown']
        self.rentou_classes = ['TOUKUI','TOU','FXP', 'BS']

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_image_list = data_context.input("in_image")
        in_head_list = data_context.input("in_head")
        out_data_list = data_context.output("out_data")
        modelbox.debug("------draw boxes in head size {}".format(in_head_list.size()))
        modelbox.debug("------draw boxes in image size {}".format(in_image_list.size()))

        for image, head_buffer in zip(in_image_list, in_head_list):
            bboxes = image.get("bboxes")
            bboxes = np.array(bboxes).reshape(-1, 4)
            pgie_classes = image.get("bboxes_classes")
            pgie_scores = image.get("bboxes_scores")
            # modelbox.info("pgie_classes: {}".format(pgie_classes))
            # modelbox.info("pgie_scores: {}".format(pgie_scores))

            width = image.get("width")
            height = image.get("height")
            channel = image.get("channel")

            out_img = np.array(image.as_object(), dtype=np.uint8)
            # out_img = out_img.reshape(640, 640, channel)
            out_img = out_img.reshape(height, width, channel)

            head_boxes = head_buffer.get("bboxes")
            head_boxes = np.array(head_boxes).reshape(-1, 4)
            # head_boxes = in_head_list.get("bboxes")
            head_boxes_num = head_buffer.get("bboxes_num")
            head_boxes_num = head_boxes_num.astype(int)
            sgie_classes = head_buffer.get("bboxes_classes")
            sgie_classes = sgie_classes.astype(int)
            sgie_scores = head_buffer.get("bboxes_scores")
            # sgie_scores = sgie_scores.astype(int)

            # emotion = emotion.as_object().split(",")
            modelbox.debug("car_bboxes: {}".format(bboxes))
            h_frist = 0
            for ind, box in enumerate(bboxes):
                cl = pgie_classes[ind]
                score = pgie_scores[ind]
                # modelbox.info(type(box))
                # modelbox.info(" index {} {} {}".format(ind, head_boxes_num[ind], box[0]))
                # modelbox.info("{} {}".format(box[2], box[3]))
                cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                # cv2.putText(out_img, , (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


                # print('class: {}, score: {}'.format(self.car_classes[cl], score))
                cv2.putText(out_img, '{0} {1:.2f}'.format(self.car_classes[cl], score),
                                                        (box[0], box[1] - 6),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.6, (0, 255, 255), 2)

                head_fiter = head_boxes[h_frist:(h_frist+head_boxes_num[ind])]
                head_cl_fiter = sgie_classes[h_frist:(h_frist+head_boxes_num[ind])]
                head_sc_fiter = sgie_scores[h_frist:(h_frist+head_boxes_num[ind])]
                # modelbox.info("head fiter {}".format(head_fiter))
                for h_box, h_cl, h_sc in zip(head_fiter, head_cl_fiter, head_sc_fiter):
                    # modelbox.info(type(h_box))
                    # modelbox.info("hbox : {}".format(h_box))
                    # modelbox.info("h_cl : {}".format(h_cl))
                    # modelbox.info("h_sc : {}".format(h_sc))
                    h_top = h_box[0] + box[0]
                    h_left = h_box[1] + box[1]
                    h_right = h_box[2] + box[0]
                    h_bottom = h_box[3] + box[1]
                    cv2.rectangle(out_img, (h_top, h_left), (h_right, h_bottom), (0, 255, ), 1)
                    cv2.putText(out_img, '{0} {1:.2f}'.format(self.rentou_classes[h_cl], h_sc),
                                                        (h_top, h_left - 6),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.6, (0, 255, 255), 2)
                h_frist = h_frist + head_boxes_num[ind]
            
           
            # modelbox.info("head_bboxes: {}".format(head_boxes))
            # for box in head_boxes:
            #     modelbox.info(box)
            #     cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            
            # cv2.imwrite("/home/wm/code/car-rk-test/src/flowunit/draw_boxes/t1.jpg", out_img)
            add_buffer = modelbox.Buffer(self.get_bind_device(), out_img)
            add_buffer.copy_meta(image)
            out_data_list.push_back(add_buffer)


        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
