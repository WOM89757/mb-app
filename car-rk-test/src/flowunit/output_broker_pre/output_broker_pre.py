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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import _flowunit as modelbox
import json
import numpy as np
import cv2
from datetime import datetime


class Output_broker_preFlowUnit(modelbox.FlowUnit):
    # Derived from modelbox.FlowUnit
    def __init__(self):
        super().__init__()

    def open(self, config):
        # Open the flowunit to obtain configuration information
        output_broker_config_path = config.get_string("output_broker_config")
        with open(output_broker_config_path) as cfg_file:
            self.output_cfg = cfg_file.read()
        self.draw_results = config.get_bool("draw_results")
        self.save_violation_img = config.get_bool("save_results")
        # self.car_classes = ['Car', 'Bus', 'Truck', 'Tricycle', 'Motorbike', 'Bicycle', 'Special', 'vehicle_Unknown']
        self.car_classes = ['Car_Saloon', 'Car_SUV', 'Car_MPV', 'Car_Jeep', 'Car_Sports', 'Car_Taxi', 'Car_Police', 'Bus_Big', 'Bus_Middle',
            'Bus_Small', 'Bus_School', 'Bus_Bus', 'Bus_Ambulance', 'Truck_Big','Truck_Van', 'Truck_Engineering', 'Truck_Fueltank', 'Truck_Construction',
            'Truck_Fire', 'Truck_Garbage', 'Truck_Watering', 'Tricycle', 'Motorbike', 'Bicycle', 'Special_Military', 
            'Special_other', 'vehicle_Unknown']
        self.rentou_classes = ['TOUKUI','TOU','FXP', 'BS']
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # Process the data
        # input data
        in_data = data_context.input("input_data")
        out_event_data_list = data_context.output("out_event_data")

        session_ctx = data_context.get_session_context()
        session_cfg = session_ctx.get_session_config()
        session_cfg.set("flowunit.output_broker.config", self.output_cfg)
        session_cfg.set("config", self.output_cfg)
        
        
        # for buffer in in_data:
        #     result = buffer.as_object()
        #     print(result, type(result))

        #     # result_str = (json.dumps(result) + chr(0)).encode('utf-8').strip()
            # result_str = (json.dumps(result) + chr(0)).encode('utf-8')
            # add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
            # add_buffer.set("msg_name", 'voilation detect')
            # add_buffer.set("output_broker_names", 'webhook-wxs')
            # out_data.push_back(add_buffer)


        for buffer in in_data:
            bboxes = buffer.get("pgie_bboxes")
            bboxes = np.array(bboxes).reshape(-1, 4)
            pgie_classes = buffer.get("pgie_bboxes_classes")
            pgie_scores = buffer.get("pgie_bboxes_scores")
            # modelbox.info("pgie_classes: {}".format(pgie_classes))
            # modelbox.info("pgie_scores: {}".format(pgie_scores))
            pgie_track_ids = buffer.get("pgie_tracker_ids")
            frame_index = buffer.get('index')

                
            width = buffer.get("width")
            height = buffer.get("height")
            channel = buffer.get("channel")

            out_img = np.array(buffer.as_object(), dtype=np.uint8)
            # out_img = out_img.reshape(640, 640, channel)
            out_img = out_img.reshape(height, width, channel)

            head_boxes = buffer.get("sgie_bboxes")
            head_boxes = np.array(head_boxes).reshape(-1, 4)
            # head_boxes = in_head_list.get("bboxes")
            head_boxes_num = buffer.get("sgie_bboxes_num")
            head_boxes_num = head_boxes_num.astype(int)
            sgie_classes = buffer.get("sgie_bboxes_classes")
            sgie_classes = sgie_classes.astype(int)
            sgie_scores = buffer.get("sgie_bboxes_scores")
            # sgie_scores = sgie_scores.astype(int)

            # emotion = emotion.as_object().split(",")
            modelbox.debug("pgie_bboxes: {}".format(bboxes))
            h_frist = 0


            now = datetime.now()
            # "yyyy-MM-dd HH:mm:ss"
            detect_time = now.strftime("%Y-%m-%d %H:%M:%S")
            # print(detect_time)

            # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("/home/wm/code/modelbox-app/car-rk-test/src/flowunit/draw_boxes/t2.jpg", out_img)
            # buffer_image = cv2.imencode('.jpg', out_img)[1]
            # img_base64 = str(base64.b64encode(buffer_image))[2:-1]
            # modelbox.info("img base64 len: {}".format(len(img_base64)))
            vioaltion_num = 0
            for ind, box in enumerate(bboxes):
                boxcar_helmet_count = 0
                boxcar_non_helmet_count = 0

                motorcycle_helmet_count = 0
                motorcycle_non_helmet_count = 0

                sanlunche_helmet_count = 0
                sanlunche_non_helmet_count = 0
                find_steering = False

                cl = pgie_classes[ind]
                score = pgie_scores[ind]
                # modelbox.info(type(box))
                # modelbox.info(" index {} {} {}".format(ind, head_boxes_num[ind], box[0]))
                # modelbox.info("{} {}".format(box[2], box[3]))
                # cv2.putText(out_img, , (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


                # print('class: {}, score: {}'.format(self.car_classes[cl], score))
                trackid = pgie_track_ids[ind]
                trackid = trackid if trackid else ''

                head_fiter = head_boxes[h_frist:(h_frist+head_boxes_num[ind])]
                head_cl_fiter = sgie_classes[h_frist:(h_frist+head_boxes_num[ind])]
                head_sc_fiter = sgie_scores[h_frist:(h_frist+head_boxes_num[ind])]
                head_list = []
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
                    if self.draw_results:
                        #pgie
                        cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(out_img, '{0} {1} {2:.2f}'.format(trackid, self.car_classes[cl], score),
                                                                (box[0], box[1] - 6),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                0.6, (0, 255, 255), 2)

                        # sgie
                        cv2.rectangle(out_img, (h_top, h_left), (h_right, h_bottom), (0, 255, ), 1)
                        cv2.putText(out_img, '{0} {1:.2f}'.format(self.rentou_classes[h_cl], h_sc),
                                                            (h_top, h_left - 6),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.6, (0, 255, 255), 2)
                    head_iter = '"X": "{}", "Y": "{}", "W": "{}", "H":"{}", "KXD": "{}"'.format(h_top, h_left, h_right - h_top, h_bottom - h_left, int(h_sc*100))
                    head_list.append(head_iter)
                    # TODO violation
                    if 'Truck' in self.car_classes[cl]:
                        if 'TOUKUI' == self.rentou_classes[h_cl]:
                            boxcar_helmet_count = boxcar_helmet_count + 1
                        elif 'TOU' == self.rentou_classes[h_cl]:
                            boxcar_non_helmet_count = boxcar_non_helmet_count + 1

                    if 'Motorbike' == self.car_classes[cl]:
                        if 'TOUKUI' == self.rentou_classes[h_cl]:
                            motorcycle_helmet_count = motorcycle_helmet_count + 1
                        elif 'TOU' == self.rentou_classes[h_cl]:
                            motorcycle_non_helmet_count = motorcycle_non_helmet_count + 1
                    
                    if 'Tricycle' == self.car_classes[cl]:
                        if 'TOUKUI' == self.rentou_classes[h_cl]:
                            sanlunche_helmet_count = sanlunche_helmet_count + 1
                        elif 'TOU' == self.rentou_classes[h_cl]:
                            sanlunche_non_helmet_count = sanlunche_non_helmet_count + 1
                        elif 'FXP' == self.rentou_classes[h_cl]:
                            find_steering = True
                find_violation = False
                truck_total_estimated_people_cnt = boxcar_helmet_count + boxcar_non_helmet_count
                if truck_total_estimated_people_cnt > 0:
                    modelbox.info("detect truck overman {}".format(truck_total_estimated_people_cnt))
                
                moto_total_estimated_people_cnt = motorcycle_helmet_count + motorcycle_non_helmet_count
                if moto_total_estimated_people_cnt > 2:
                    modelbox.info("detect motobike overman {}".format(moto_total_estimated_people_cnt))
                if motorcycle_non_helmet_count > 0:
                    modelbox.info("detect motobike no helmet {}".format(motorcycle_non_helmet_count))
                    find_violation = True

                sanlunche_total_estimated_people_cnt = sanlunche_helmet_count + sanlunche_non_helmet_count
                if (sanlunche_total_estimated_people_cnt > 2 and find_steering) or (not find_steering and sanlunche_total_estimated_people_cnt > 1):
                    overman_num = sanlunche_total_estimated_people_cnt - 2 if find_steering else sanlunche_total_estimated_people_cnt - 1
                    modelbox.info("detect sanlunche overman {}".format(overman_num))
                
                h_frist = h_frist + head_boxes_num[ind]

                if find_violation and True:
                    # add buffer : one car info and full img
                    if self.save_violation_img:
                        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("/userdata/wm/code/modelbox-app/car-rk-test/src/flowunit/output_broker_pre/violation/v-{}-{}.jpg".format(frame_index, vioaltion_num), out_img)
                    vioaltion_num = vioaltion_num + 1

                    event_json = {
                        "sxjbh": "camera id",
                        # "tp": img_base64,
                        "tp": "base64 img",
                        "gcsj": detect_time,
                        "clgzid": trackid,
                        "wztz": {
                            "clwz": "{},{},{},{}".format(box[0], box[1], box[2] - box[0], box[3] - box[1])
                        },
                        "cxtz": {
                            "cllxgl": "11-99"
                        },
                        "jsxwtz": {
                            "mtcbdtk": "1_{}".format(int(score*100))
                        },
                        "fjsbxx": {
                            "ryxx": {
                                "s1": int(head_boxes_num[ind]),
                                "wzxx": head_list
                            }
                        }
                    }
                    # event_json = json.dumps(event_json, indent=4, ensure_ascii=False)
                    event_json = json.dumps(event_json, ensure_ascii=False)
                    # print(jsonString)


                    # event_buffer = modelbox.Buffer(self.get_bind_device(), event_json)
                    # # event_buffer.copy_meta(image)
                    # out_event_data_list.push_back(event_buffer)

                    result_str = event_json.encode('utf-8')
                    add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
                    add_buffer.set("msg_name", 'voilation detect')
                    add_buffer.set("output_broker_names", 'webhook-wxs')
                    out_event_data_list.push_back(add_buffer)
                else:
                    #TODO track object and report when it be removed
                    """
                    {
                        imgs
                        quality_scores
                        id

                    } track_object
                    


                    if find_object(to.id == track_object.id):
                        update track object
                    else:
                        track_objects.push_back(track_object)
                    
                    
                    if activate_removed:
                        remove_track = self.track_objects[remove_track_id]
                        init report remove_track json
                        detect result report to out_detect_date port

                    """
                    pass



# {
#     "sxjbh": "camera id",
#     "tp": "base64 img",
#     "gcsj": "yyyy-MM-dd HH:mm:ss",
#     "clgzid": "track id",
#     "wztz": {
#         "clwz": "x,y,w,h"
#     },
#     "cxtz": {
#         "cllxgl": "11-99"
#     },
#     "jsxwtz": {
#         "mtcbdtk": "1_80"
#     },
#     "fjsbxx": {
#         "ryxx": {
#             "s1": 3,
#             "wzxx": [
#                 {"X": "x", "Y": "y", "W": "w", "H":"h", "KXD": "60"},
#                 {"X": "x", "Y": "y", "W": "w", "H":"h", "KXD": "50"},
#                 {"X": "x", "Y": "y", "W": "w", "H":"h", "KXD": "30"}
#             ]
#         }
#     }
# }            




        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        # Close the flowunit
        return modelbox.Status()

    def data_pre(self, data_context):
        # Before streaming data starts
        return modelbox.Status()

    def data_post(self, data_context):
        # After streaming data ends
        return modelbox.Status()