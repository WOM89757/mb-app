/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODELBOX_FLOWUNIT_TRACKE_CPU_H_
#define MODELBOX_FLOWUNIT_TRACKE_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>
#include <modelbox/buffer.h>
#include "tracker/BYTETracker.h"

constexpr const char *FLOWUNIT_NAME = "tracke";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_VERSION = "1.0.0";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A tracker that get a track id for rectangle area on the input bboxes flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer type is image, the "
    "output port buffer type are image. \n"
    "\t  The input buffer contains the following meta fields:\n"
    "\t\tField Name: bboxes,              Type: vector<ssize_t>\n"
    "\t\tField Name: bboxes_classes,      Type: vector<ssize_t>\n"
    "\t\tField Name: bboxes_scores,       Type: vector<double>\n"
    "\t\tField Name: width,               Type: int32_t\n"
    "\t\tField Name: height,              Type: int32_t\n"
    "\t\tField Name: width_stride,        Type: int32_t\n"
    "\t\tField Name: height_stride,       Type: int32_t\n"
    "\t\tField Name: channel,             Type: int32_t\n"
    "\t\tField Name: pix_fmt,             Type: string\n"
    "\t\tField Name: layout,              Type: int32_t\n"
    "\t\tField Name: shape,               Type: vector<size_t>\n"
    "\t\tField Name: type,                Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint:";


class TrackeFlowUnit : public modelbox::FlowUnit {
 public:
  TrackeFlowUnit();
  ~TrackeFlowUnit() override;

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();
  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataGroupPre(std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataGroupPost(std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::shared_ptr<BYTETracker> tracker;
  std::mutex tracker_mtx_;
  float fps;
  bool init_flag;
};

#endif  // MODELBOX_FLOWUNIT_TRACKE_CPU_H_
