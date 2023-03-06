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

class ExpandBox(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def get_align(self, x, align=16, is_ceil=True):
        while (x % align) != 0:
            if is_ceil :
                x = x + 1
            else:
                x = x - 1
        return x

    def process(self, data_context):
        in_data_list = data_context.input("in_data")
        out_image_list = data_context.output("roi_image")

        for in_buffer in in_data_list:
            width = in_buffer.get("width")
            height = in_buffer.get("height")
            channel = in_buffer.get("channel")
            fmt = in_buffer.get("pix_fmt")

            # modelbox.info("expand box get frame shape {} {} {}".format(channel, width, height))
            bboxes = in_buffer.get("bboxes")
            modelbox.debug("bboxes size {}".format(len(bboxes)))
            modelbox.debug("bboxes {}".format(bboxes))
            img = np.array(in_buffer.as_object(), dtype=np.uint8)
            img = img.reshape(height, width, channel)
            # img = img.reshape(640, 640, channel)

            bboxes = np.array(bboxes).reshape(-1, 4)
            # 1228800 into shape (2232,4096,3)
            for box in bboxes:
                # img_roi = img[box[1]:box[3], box[0]:box[2]]
                # # cv2.imwrite("/home/wm/code/modelbox-app/car-rk-test/src/flowunit/expand_box/box-1.jpg", img_roi)
                # modelbox.info("box is {} ".format(box))
                # roi_height = box[3]-box[1]
                # roi_width = box[2]-box[0]
                # modelbox.info("width: {} {} height: {} {}".format(roi_width, roi_width%16, roi_height, roi_height %2))
                
                # for ind, b in enumerate(box):
                #     if ind % 2:
                #         if box[ind] % 6 != 0:
                #             box[ind] = self.get_align(box[ind], 6, (ind==2 and False))
                #     else:
                #         if box[ind] % 48 != 0:
                #             box[ind] = self.get_align(box[ind], 48, (ind==3 and False))
                
                roi_height = box[3]-box[1]
                roi_width = box[2]-box[0]
                modelbox.debug("width: {} {} height: {} {}".format(roi_width, roi_width%48, roi_height, roi_height %6))
                img_roi = img[box[1]:box[3], box[0]:box[2]]
                # cv2.imwrite("/home/wm/code/modelbox-app/car-rk-test/src/flowunit/expand_box/box-2.jpg", img_roi)
                # img_roi = img_roi[:, :, ::-1]

                # img_roi = img_roi.flatten()

                add_buffer = modelbox.Buffer(self.get_bind_device(), img_roi)
                add_buffer.copy_meta(in_buffer)
                add_buffer.set("pix_fmt", fmt)
                add_buffer.set("width", int(roi_width))
                add_buffer.set("height", int(roi_height))
                add_buffer.set("width_stride", int(roi_width))
                add_buffer.set("height_stride", int(roi_height))
                add_buffer.set("channel", channel)

                # modelbox.debug("expand box height:{} width:{}".format(add_buffer.get("height"), add_buffer.get("width")))
                out_image_list.push_back(add_buffer)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
