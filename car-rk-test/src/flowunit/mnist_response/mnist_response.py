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
import base64

class MnistResponseFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data = data_context.input("in_image")
        # in_image = data_context.input("in_image")
        out_data = data_context.output("out_data")

        for buffer in in_data:
            result_str = ''
            if buffer.has_error():
                error_msg = buffer.get_error_msg()
                result = {
                    "error_msg": str(error_msg)
                }
            else:
                width = buffer.get('width')
                height = buffer.get('height')
                channel = buffer.get('channel')
                buffer_img = buffer.as_object()
                modelbox.info("response buffer shape {} {} {}".format(channel, width, height))
                img_data = np.array(buffer_img, copy=False)
                # img_data = np.array(buffer_img, dtype=np.uint8, copy=False)

                img_data = img_data.reshape((height, width, channel))
                # img_data = img_data.reshape((width, height, channel))
                # img_data = img_data.reshape((640, 640, channel))
                # img_data = cv2.resize(img_data, (width, height))
                # img_data = cv2.resize(img_data, (height, width))
                
                # modelbox.info("--------img shape {}", img_data.shape)

                # cv2.imwrite("/home/wm/code/car-rk-test/src/flowunit/mnist_response/t2.jpg", img_data)
                buffer_image = cv2.imencode('.jpg', img_data)[1]
                img_base64 = str(base64.b64encode(buffer_image))[2:-1]
                modelbox.info("img base64 len: {}".format(len(img_base64)))
                
                result = {
                    "image_base64": img_base64
                }

                

            result_str = (json.dumps(result) + chr(0)).encode('utf-8').strip()
            add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
            out_data.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
