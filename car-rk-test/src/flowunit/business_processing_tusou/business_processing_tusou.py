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
import os
import base64

class Business_processing_tusouFlowUnit(modelbox.FlowUnit):
    # Derived from modelbox.FlowUnit
    def __init__(self):
        super().__init__()


    def find_track(self, track_object):
        for iter in self.track_objects:
            if iter.id != track_object.id:
                continue
            return True, iter
        return False, None

    def add_track(self, track_object):
        if len(self.track_objects) > self.track_object_max_size:
            modelbox.info("track set is full, track object max size is: {}({})".format(len(self.track_objects), self.track_object_max_size))
            return;
        self.track_objects.append(track_object)

    class Track_Object:
        def __init__(self, id, x, y, w, h, scores, classes, img, frame_id, detect_time):
            self.id = id
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.scores = scores
            self.classes = classes
            self.img = img
            self.frame_id = frame_id
            self.detect_time = detect_time

        def update(self, other):
            # if (other.w * other.h) > (self.w * self.h):
                # return
            if other.scores > self.scores:
                self.x = other.x
                self.y = other.y
                self.w = other.w
                self.h = other.h
                self.scores = other.scores
                self.classes = other.classes
                self.img = other.img
                self.frame_id = other.frame_id
                self.detect_time = other.detect_time
        def __str__(self):
            return ("id: {}, x: {}, y: {}, w: {}, h: {}, scores: {}, classes: {}, frame_id: {}, detect_time: {}".format(self.id, self.x, self.y, self.w, self.h, self.scores, self.classes, self.frame_id, self.detect_time))

    def open(self, config):
        # Open the flowunit to obtain configuration information
        output_broker_config_path = config.get_string("output_broker_config")
        with open(output_broker_config_path) as cfg_file:
            self.output_cfg = cfg_file.read()
        self.draw_results = config.get_bool("draw_results")
        self.save_violation_img = config.get_bool("save_results")
        self.save_results_path = config.get_string("save_results_path")
        self.violation_event_root_path = os.path.join(self.save_results_path, "violation-event")
        self.objects_root_path = os.path.join(self.save_results_path, "objects")
        if not os.path.exists(self.save_results_path) or not os.path.exists(self.violation_event_root_path) or os.path.exists(self.objects_root_path):
            if not os.path.exists(self.violation_event_root_path):
                os.makedirs(self.violation_event_root_path)
            if not os.path.exists(self.objects_root_path):
                os.makedirs(self.objects_root_path)
        self.include_base64Img = config.get_bool("include_base64Img")

        # self.car_classes = ['Car', 'Bus', 'Truck', 'Tricycle', 'Motorbike', 'Bicycle', 'Special', 'vehicle_Unknown']
        self.car_classes = ['Car_Saloon', 'Car_SUV', 'Car_MPV', 'Car_Jeep', 'Car_Sports', 'Car_Taxi', 'Car_Police', 'Bus_Big', 'Bus_Middle',
            'Bus_Small', 'Bus_School', 'Bus_Bus', 'Bus_Ambulance', 'Truck_Big','Truck_Van', 'Truck_Engineering', 'Truck_Fueltank', 'Truck_Construction',
            'Truck_Fire', 'Truck_Garbage', 'Truck_Watering', 'Tricycle', 'Motorbike', 'Bicycle', 'Special_Military', 
            'Special_other', 'vehicle_Unknown']
        
        # 11	大型客车
        # 12	中型客车
        # 13	小型客车
        # 14	微型客车
        # 21	重中型货车
        # 22	轻微型货车
        # 23	三轮车
        # 30	摩托车
        # 40	挂车
        # 50	电动自行车
        # 60	拖拉机
        # 99	其他
        self.wus_classes = {
            "": 11,
            "": 12,
        }
        self.rentou_classes = ['TOUKUI','TOU','FXP', 'BS']
        self.track_object_max_size = 10
        self.track_objects = []
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # Process the data
        # input data
        in_data = data_context.input("input_data")
        out_object_data_list = data_context.output("out_object_data")
        out_event_data_list = data_context.output("out_event_data")
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
            try:
                bboxes = buffer.get("pgie_bboxes")
            except ValueError:
                continue
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
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

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
            result_null = 'null'
            img_base64 = "base64 img"
            if self.include_base64Img:
                buffer_image = cv2.imencode('.jpg', out_img)[1]
                img_base64 = str(base64.b64encode(buffer_image))[2:-1]
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
                
                if trackid != '':
                    new_track_object = self.Track_Object(trackid, box[0], box[1], box[3] - box[1], box[2] - box[0], score, cl, out_img, frame_index, detect_time);
                    has_track_object, curr_track = self.find_track(new_track_object) 
                    if has_track_object:
                        curr_track.update(new_track_object)
                    else:
                        self.add_track(new_track_object)

                if self.draw_results:
                    #pgie
                    cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(out_img, '{0} {1} {2:.2f}'.format(trackid, self.car_classes[cl], score),
                                                            (box[0], box[1] - 6),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.6, (0, 255, 255), 2)


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
                        # sgie
                        cv2.rectangle(out_img, (h_top, h_left), (h_right, h_bottom), (0, 255, ), 1)
                        cv2.putText(out_img, '{0} {1:.2f}'.format(self.rentou_classes[h_cl], h_sc),
                                                            (h_top, h_left - 6),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.6, (0, 255, 255), 2)
                    # head_iter = '"X": "{}", "Y": "{}", "W": "{}", "H":"{}", "KXD": "{}"'.format(h_top, h_left, h_right - h_top, h_bottom - h_left, int(h_sc*100))
                    # 以车辆左上角为原点
                    head_iter = '"X": "{}", "Y": "{}", "W": "{}", "H":"{}", "KXD": "{}"'.format(h_box[0], h_box[1], h_box[3] - h_box[1], h_box[2] - h_box[0], int(h_sc*100))
                    head_list.append(head_iter)
                    # violation
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
                        # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("{}/v-{}-{}.jpg".format(self.violation_event_root_path, frame_index, vioaltion_num), out_img)
                    vioaltion_num = vioaltion_num + 1

                    event_json = {
                        "sxjbh": "camera id",
                        "tp": img_base64,
                        "gcsj": detect_time,
                        "clgzid": trackid,
                        "wztz": {
                            "clwz": "{},{},{},{}".format(box[0], box[1], box[3] - box[1], box[2] - box[0])
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

                    # result_str = event_json.encode('utf-8')
                    result_str = event_json
                    add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
                    add_buffer.set("msg_name", 'voilation detect')
                    add_buffer.set("output_broker_names", 'webhook-wxs-violation')
                    out_event_data_list.push_back(add_buffer)
                
            # for iter in self.track_objects:
            #     modelbox.info("before {}".format(iter))
            # modelbox.info("track objects size {}".format(len(self.track_objects)))
            removed_track_objects = []
            if (len(self.track_objects) > 0):
                tmp_track_objects = []
                for iter in self.track_objects:
                    if (frame_index - iter.frame_id) < 30:
                        tmp_track_objects.append(iter)
                    else:
                        pass
                        # modelbox.info("removed track object {}".format(iter.id))
                removed_track_objects = list(set(self.track_objects).difference(set(tmp_track_objects)))
                # modelbox.info("removed ids is {}".format(removed_track_objects))
                self.track_objects = tmp_track_objects
            # for iter in self.track_objects:
            #     modelbox.info("after {}".format(iter))


            if len(removed_track_objects) > 0 and True:
                #TODO track object and report when it be removed
                for iter in removed_track_objects:

                    modelbox.info("removed track {}".format(iter))
                    if self.save_violation_img:
                        # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                        # iter_img = cv2.cvtColor(iter.img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("{}/r-{}-{}.jpg".format(self.objects_root_path, iter.frame_id, iter.id), iter.img)
                        cv2.imwrite("{}/r-{}-{}-leave.jpg".format(self.objects_root_path, iter.frame_id, iter.id), out_img)
                    report_json = {
                        "sxjbh": "camera id",
                        "tp": img_base64,
                        "gcsj": iter.detect_time,
                        "clgzid": iter.id,
                        "wztz": {
                            "clwz": "{},{},{},{}".format(iter.x, iter.y, iter.w, iter.h)
                        },
                        "cxtz": {
                            "cllxgl": "11-99"
                        },
                        "jsxwtz": {
                            "mtcbdtk": "0_{}".format(int(100))
                        },
                        "fjsbxx": ""
                    }
                    # event_json = json.dumps(event_json, indent=4, ensure_ascii=False)
                    report_json = json.dumps(report_json, ensure_ascii=False)
                    # print(jsonString)


                    # event_buffer = modelbox.Buffer(self.get_bind_device(), event_json)
                    # # event_buffer.copy_meta(image)
                    # out_event_data_list.push_back(event_buffer)

                    result_str = report_json
                    add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
                    add_buffer.set("msg_name", 'object detect')
                    add_buffer.set("output_broker_names", 'webhook-wxs-object')
                    out_object_data_list.push_back(add_buffer)


                """
                
                if activate_removed:
                    remove_track = self.track_objects[remove_track_id]
                    init report remove_track json
                    detect result report to out_detect_date port

                """
                # modelbox.info('event size: {}'.format(out_event_data_list.size()))
                # modelbox.info('object size: {}'.format(out_object_data_list.size()))
            if (out_object_data_list.size()*out_event_data_list.size() == 0) and (out_object_data_list.size()+out_event_data_list.size() > 0):
                add_buffer = modelbox.Buffer(self.get_bind_device(), result_null)
                if out_object_data_list.size() == 0:
                    out_object_data_list.push_back(add_buffer)
                    # modelbox.info("add out object")
                elif out_event_data_list.size() == 0:
                    out_event_data_list.push_back(add_buffer)
                    # modelbox.info("add out event")


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
        modelbox.info("--------start pre-------")
        session_ctx = data_context.get_session_context()
        session_cfg = session_ctx.get_session_config()
        session_cfg.set("flowunit.output_broker.config", self.output_cfg)
        return modelbox.Status()

    def data_post(self, data_context):
        # After streaming data ends
        return modelbox.Status()