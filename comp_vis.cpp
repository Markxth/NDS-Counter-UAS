#include <stdio.h>
#include <cstdint>
#include <opencv2/opencv.hpp> 

typedef float float32_t;

void aim(int32_t camera_index, float32_t takeoff_alt) { 

    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolo26.onnx");
    cv::VideoCapture capture(camera_index);
    if(!capture.isOpened()) { 
        throw std::runtime_error("Camera not accessible.");
    }
    //height and width will not be above 1 million so we can use uint16_t
    uint16_t height = static_cast<uint16_t>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)); //MISRA compliant ; capture returns a double so it needs int conversion
    uint16_t width  = static_cast<uint16_t>(capture.get(cv::CAP_PROP_FRAME_WIDTH));

    std::vector<uint8_t>    classIds;
    std::vector<float32_t>  confidences;
    std::vector<cv::Rect>   boxes; //vector of rectangles

    while(true) {
        cv::Mat frame;
        capture >> frame;
        if(frame.empty()) { break; }

        //1.INFERENCE
        cv::Mat blob; //normalise, resize. by doing 1.0/255.0 we already got the 32 bit float we need by default for OpenCV
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640,480), cv::Scalar(), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames()); //we do not hard code as hard coding breaks if model structure changes for whatever reason

        //2. PARSE IMAGE - reshape + structure. 4d tensor into smth usable
        cv::Mat output = outputs[0];
        output = output.reshape(1, output.size[1]); // from [batch, channels, width, height] into [confidence scores per class + coordinates, boxes]
        cv::transpose(output, output);

        float32_t x_scale = static_cast<float32_t>(frame.cols) / 640.0F; // 0F to stay float and not be converted to double as it does the division which goes against MISRA
        float32_t y_scale = static_cast<float32_t>(frame.rows) / 640.0F;

        for(int32_t i = 0; i < output.rows; i++) {
            float32_t* row = output.ptr<float32_t>(i); //loop through rows
            cv::Mat scores(1, 80, CV_32F, row + 4);
            cv::Point classIdPoint;
            double maxScore; // MISRA deviation: cv::minMaxLoc requires double*
            cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);
            if(maxScore < 0.5) { continue; }

            //none of these can be neg so we keep em unsigned
            float32_t cx = row[0], cy = row[1], w = row[2], h = row[3];
            int32_t x1 = static_cast<int32_t>((cx - w/2.0F) * x_scale);
            int32_t y1 = static_cast<int32_t>((cy - h/2.0F) * y_scale);
            int32_t bw = static_cast<int32_t>(w * x_scale);
            int32_t bh = static_cast<int32_t>(h * y_scale);

            //append in c++ is push_back
            boxes.push_back(cv::Rect(x1, y1, bw, bh));
            confidences.push_back(static_cast<float32_t>(maxScore));
            classIds.push_back(static_cast<uint8_t>(classIdPoint.x));
        }

        //3. Non Maximum Suppression
        std::vector<int32_t> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5F, 0.4F, indices);

        //4. bounding boxes and confidence
        for(int32_t idx : indices)
        {
            if(classIds[static_cast<uint32_t>(idx)] != 0U ){continue ; } // where 0U is 0 unsigned, such that we compare unsigned to unsigned
            cv::Rect  box  = boxes[static_cast<uint32_t>(idx)];
            float32_t conf = confidences[static_cast<uint32_t>(idx)];

            cv::rectangle( 
                frame, 
                cv::Point(box.x, box.y), 
                cv::Point(box.x + box.width, box.y + box.height),
                cv::Scalar(0, 255, 0),
                1 
            );

            //add bounding box to then use later
            char label[32] = {}; //25 chars for human : confidence : <actual_value>; i.e human : confidence : 0.9. if more is needed, modify the size.
            //format the text
            snprintf(label, sizeof(label), "human : confidence : %.2f", conf);
            cv::putText( 
                frame,
                label,
                cv::Point(box.x, box.y - 10), //position relative to box - above here
                cv::FONT_HERSHEY_SIMPLEX, //font
                2, //fontScale
                cv::Scalar(0, 255, 0), //color
                2, //thickness
                cv::LINE_AA, // anti aliased ; 2 other options exist : default which is cv::LINE_8 and cv::LINE_4 which is jagged and fastest
                false //bottom left origin; false for normal text ; true for upside down
            );
        }

        cv::imshow("Capture", frame);
        if((cv::waitKey(1) & 0xFF) == 'q') { 
            break;
        }
    }
}