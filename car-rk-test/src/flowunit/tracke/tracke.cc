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

#include "tracke.h"
#include "modelbox/flowunit_api_helper.h"

TrackeFlowUnit::TrackeFlowUnit() = default;
TrackeFlowUnit::~TrackeFlowUnit() = default;

modelbox::Status TrackeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
    this->fps = opts->GetInt16("frame_rate");
    //TODO alter fps of byte tracker
    this->track_buffer = opts->GetInt16("track_buffer");
    this->track_thresh = opts->GetFloat("track_thresh");
    this->high_thresh = opts->GetFloat("high_thresh");
    this->match_thresh = opts->GetFloat("match_thresh");
    this->tracker = std::make_shared<BYTETracker>(this->fps, this->track_buffer, track_thresh, high_thresh, match_thresh);
    this->init_flag = false;

    return modelbox::STATUS_OK;
}

modelbox::Status TrackeFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status TrackeFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
}

modelbox::Status TrackeFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
    // input data
    auto input_bufs = data_ctx->Input("input_data");
    auto output_bufs = data_ctx->Output("output_data");


    // MBLOG_INFO << "tracke";
    for (uint k = 0; k < input_bufs->Size(); ++k) {
        auto input_buffer = input_bufs->At(k);
        std::vector<ssize_t> bboxes;
        input_buffer->Get("bboxes", bboxes);
        std::vector<double> scores;
        std::vector<ssize_t> classes;
        input_buffer->Get("bboxes_classes", classes);
        input_buffer->Get("bboxes_scores", scores);

        // if (!this->init_flag) {
        //     int num, den;
        //     input_buffer->Get("rate_num", num);
        //     input_buffer->Get("rate_den", den);
        //     this->fps = 1.0 * num / den;
        //     // MBLOG_INFO << "rate num " << num << " den " << den;
        //     this->init_flag = true;
        // }
        if (bboxes.empty()) {
            output_bufs->PushBack(input_buffer);
            continue;
        }

        
        int box_num = bboxes.size() / 4;
        std::vector<Object> objects;
        for(int i = 0; i < box_num; i++) {
            Object track_object;
            // x1, y1, x2, y2, score, label 
            int ind = i * 4;
            track_object.rect.x = bboxes[ind];
            track_object.rect.y = bboxes[ind + 1];
            track_object.rect.width = bboxes[ind + 2] - bboxes[ind];
            track_object.rect.height = bboxes[ind + 3] - bboxes[ind + 1];
            track_object.prob = scores[i];
            track_object.label = classes[i];
            objects.emplace_back(track_object);
        }

        std::vector<int> tracker_ids(objects.size(),0);
        std::lock_guard<std::mutex> lk(tracker_mtx_);
        std::vector<STrack> output_stracks = this->tracker->update(objects);
        // MBLOG_INFO << "output stacks size " << output_stracks.size();
        tracker_mtx_.unlock(); 

        for (uint i = 0; i < output_stracks.size(); i++) {
            tracker_ids[output_stracks[i].detect_id] = output_stracks[i].track_id;
        }


        // int32_t width = 0;
        // int32_t height = 0;
        // std::string pix_fmt;
        // bool exists = false;
        // exists = input_bufs->At(k)->Get("height", height);
        // if (!exists) {
        //     MBLOG_ERROR << "meta don't have key height";
        //     return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height"};
        // }

        // exists = input_bufs->At(k)->Get("width", width);
        // if (!exists) {
        //     MBLOG_ERROR << "meta don't have key width";
        //     return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width"};
        // }

        // auto input_data =
        //     static_cast<const u_char *>(input_bufs->ConstBufferData(k));

        // cv::Mat img_data(cv::Size(width, height), CV_8UC3);
        // memcpy(img_data.data, input_data, input_bufs->At(k)->GetBytes());

        // // draw  track id

        // for (int i = 0; i < output_stracks.size(); i++)
		// {
		// 	std::vector<float> tlwh = output_stracks[i].tlwh;
		// 	bool vertical = tlwh[2] / tlwh[3] > 1.6;
		// 	// if (tlwh[2] * tlwh[3] > 20 && !vertical)
		// 	if (1)
		// 	{
		// 		cv::Scalar s = tracker->get_color(output_stracks[i].track_id);
		// 		putText(img_data, format("%d", output_stracks[i].track_id), Point(tlwh[0] + 5, tlwh[1] + 15), 
        //                 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        //         rectangle(img_data, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
		// 	}
		// }
        // std::string filename = "/userdata/wm/code/modelbox-app/car-rk-test/src/flowunit/tracke/result/" + std::to_string(index) + ".jpg";
        // // MBLOG_INFO << "filename: " << filename;
        // cv::imwrite(filename, img_data);

        // for(int i = 0; i < box_num; i++) {
        //     int ind = i * 6;
        //     // MBLOG_INFO << "tracker id : " << tracker_ids[i];
        //     // cv::putText(img_data, format("%d", tracker_ids[i]), cv::Point(bboxes[ind], bboxes[ind + 1] - 5), 
        //     //                   0, 2, cv::Scalar(0, 255, 0), 2);
        //     // cv::rectangle(img_data, cv::Rect(bboxes[ind], bboxes[ind + 1], bboxes[ind + 2] - bboxes[ind], bboxes[ind + 3] - bboxes[ind + 1]), cv::Scalar(255, 0, 0), 2); 
        //     // x1, y1, x2, y2, score, label 
        //     // track_object.rect.x = bboxes[ind];
        //     // track_object.rect.y = bboxes[ind + 1];
        //     // track_object.rect.width = bboxes[ind + 2] - bboxes[ind];
        //     // track_object.rect.height = bboxes[ind + 3] - bboxes[ind + 1];
        //     // track_object.prob = scores[i];
        //     // track_object.label = classes[i];
        //     // objects.emplace_back(track_object);
        // }

        // auto output_buffer = std::make_shared<modelbox::Buffer>(GetBindDevice());
        // // output move image
        // output_buffer->Build(img_data.total() * img_data.elemSize());
        // auto output = output_buffer->MutableData();
        // memcpy(output, img_data.data, img_data.total() * img_data.elemSize());
        // output_buffer->CopyMeta(input_buffer);
        // output_buffer->Set("tracker_ids", tracker_ids);
        // output_bufs->PushBack(output_buffer);

        input_buffer->Set("tracker_ids", tracker_ids);
        output_bufs->PushBack(input_buffer);
    }

    return modelbox::STATUS_OK;
}

modelbox::Status TrackeFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
}

modelbox::Status TrackeFlowUnit::DataGroupPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
}

modelbox::Status TrackeFlowUnit::DataGroupPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(TrackeFlowUnit, desc) {
    /*set flowunit attributes*/
    desc.SetFlowUnitName(FLOWUNIT_NAME);
    desc.SetFlowUnitGroupType("Undefined");
    // input port
    desc.AddFlowUnitInput(modelbox::FlowUnitInput("input_data", FLOWUNIT_TYPE));
    // output port
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("output_data", FLOWUNIT_TYPE));
    desc.SetFlowType(modelbox::NORMAL);
    desc.SetDescription(FLOWUNIT_DESC);
    /*set flowunit parameter
    example code:
    desc.AddFlowUnitOption(modelbox::FlowUnitOption(
        "parameter0", "int", true, "640", "parameter0 describe detail"));
    desc.AddFlowUnitOption(modelbox::FlowUnitOption(
        "parameter1", "int", true, "480", "parameter1 describe detail"));
    */
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
    desc.Desc.SetName(FLOWUNIT_NAME);
    desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
    desc.Desc.SetType(FLOWUNIT_TYPE);
    desc.Desc.SetDescription(FLOWUNIT_DESC);
    desc.Desc.SetVersion(FLOWUNIT_VERSION);
}
