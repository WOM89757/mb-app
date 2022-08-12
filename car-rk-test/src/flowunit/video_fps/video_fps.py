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
import numpy as np
import time
import cv2

class Video_fpsFlowUnit(modelbox.FlowUnit):
    # Derived from modelbox.FlowUnit
    def __init__(self):
        super().__init__()
        self.last_time = time.time()
        self.cur_fps = 0
        self.frame_count = 0

    def open(self, config):
        # Open the flowunit to obtain configuration information
        self.show_fps = config.get_bool("show_fps")
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # Process the data
        # input data
        in_data = data_context.input("input_data")
        # output data
        out_data = data_context.output("out_data")
        # modelbox.info("indata size {}".format(in_data.size()))
        for buffer_img in in_data:
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')
            frame_index = buffer_img.get('index')
            # modelbox.debug("get frame shape {} {} {}".format(channel, width, height))
            # modelbox.info("get frame index: {}".format(frame_index))

            current_time = time.time()
            # if not frame_index % 8 :
            #     self.cur_fps = 1/(current_time - self.last_time)
            #     modelbox.info("fps: {}".format(self.cur_fps))
            # self.last_time = current_time

            self.frame_count = self.frame_count + 1
            if (current_time -  self.last_time) > 1 :
                self.cur_fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
                # modelbox.debug("fps: {}".format(self.cur_fps))

            if self.show_fps:
                img_data = np.array(buffer_img.as_object(), copy=False)
                img_data = img_data.reshape((height, width, channel))
                cv2.putText(img_data, 'fps: {0}'.format(self.cur_fps),
                                                            (10, 20),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.6, (0, 255, 255), 2)

                add_buffer = modelbox.Buffer(self.get_bind_device(), img_data)
                add_buffer.copy_meta(buffer_img)
                out_data.push_back(add_buffer)
            else:
                out_data.push_back(buffer_img)

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