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
import base64
import json
import cv2

class MnistPreprocess(modelbox.FlowUnit):
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
        in_data = data_context.input("in_data")
        out_data = data_context.output("out_data")

        for buffer in in_data:
            # get image from request body
            request_body = json.loads(buffer.as_object().strip(chr(0)))
            
            if  request_body.get("image_base64"):
                img_base64 = request_body["image_base64"]
                img_file = base64.b64decode(img_base64)

                # reshape img
                img = cv2.imdecode(np.fromstring(img_file, np.uint8), cv2.IMREAD_COLOR)
                modelbox.debug("--------img size: {}".format(img.shape))
                o_height = img.shape[0]
                o_width  = img.shape[1]

                # if o_width % 48 != 0:
                #     o_width = self.get_align(o_width, 48, False)
                # if o_height % 6 != 0:
                #     o_height = self.get_align(o_height, 6, False)

                # img = cv2.resize(img, (o_height, o_width),cv2.co)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                modelbox.debug("--------ori img size: {} {} {}".format(o_height, o_width, type(o_width)))
                infer_data = img
                # img = cv2.resize(img, (28, 28))
                # infer_data = np.array([255 - img], dtype=np.float32)
                # infer_data = infer_data / 255
                
                # build buffer
                add_buffer = modelbox.Buffer(self.get_bind_device(), infer_data)
                add_buffer.set("height", o_height)
                add_buffer.set("width", o_width)
                add_buffer.set("pix_fmt", "rgb")
                add_buffer.set("channel", 3)
                out_data.push_back(add_buffer)
            else:
                error_msg = "wrong key of request_body"
                modelbox.error(error_msg)
                add_buffer = modelbox.Buffer(self.get_bind_device(), "")
                add_buffer.set_error("MnistPreprocess.BadRequest", error_msg)
                out_data.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
