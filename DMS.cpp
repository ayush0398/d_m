#include <dlib/opencv.h>

#include <opencv2/highgui/highgui.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/image_processing/render_face_detections.h>

#include <dlib/image_processing.h>

#include <dlib/gui_widgets.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/optimization.h>

#include <ctime>

 

#define DLIB_JPEG_SUPPORT

#define DLIB_PNG_SUPPORT

#define DLIB_GIF_SUPPORT

#define DLIB_OPTIMIZATIOn_HEADER

 

using namespace dlib;

using namespace std;

 

// Constants for eye aspect ratio to detect blinking or drowsiness

const double EYE_AR_THRESHOLD = 0.25;

const int EYE_AR_CONSEC_FRAMES = 90;  // Number of frames the eye must be below the threshold

 

// Calculates the Eye Aspect Ratio (EAR) for a set of eye landmarks.

double calculate_ear(const std::vector<point>& landmarks)

{

    // Calculate the distances between the landmarks.

    double a = length(landmarks[1] - landmarks[5]);

    double b = length(landmarks[2] - landmarks[4]);

    double c = length(landmarks[0] - landmarks[3]);

 

    // Calculate EAR.

    double ear = (a + b) / (2.0 * c);

 

    return ear;

}

 

// Function to detect if the driver is looking at the road

bool isLookingAtRoad(const std::vector<point>& face_center, const std::vector<point>& landmarks, const std::vector<point>& face_left, const std::vector<point>& face_right, const std::vector<point>& face_top, const std::vector<point>& face_bottom) {

    // Simple heuristic: eye should be within the central part of the face

 

    /*int faceCenterX = face.x + face.width / 2;

    int eyeCenterX = eye.x + eye.width / 2;

    int faceCenterY = face.y + face.height / 2;

    int eyeCenterY = eye.y + eye.height / 2;*/

 

    int faceCenterX = face_center[0](0);

    int eyeCenterX = (landmarks[0](0) + landmarks[3](0)) / 2;

    int faceCenterY = face_center[0](1);

    int eyeCenterY = (landmarks[0](1) + landmarks[3](1)) / 2;

    double face_width = length(face_left[0] - face_right[0]);

    double face_height = length(face_top[0] - face_bottom[0]);

 

    return (std::abs(faceCenterX - eyeCenterX) < face_width * 0.25) && (std::abs(faceCenterY - eyeCenterY) < face_height * 0.25);

}

 

int main()

{

    try

    {

        //cv::VideoCapture cap("D:/DMS_dummy/Vedio_Veh_Signa_Prima_Ultra 1/Vedio_Veh_Signa_Prima_Ultra/Vehicle_Signa1.mp4");

        cv::VideoCapture cap(0);

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);//Setting the width of the video

        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

        if (!cap.isOpened())

        {

            cerr << "Unable to connect to camera" << endl;

            return 1;

        }

 

        image_window win;

        frontal_face_detector detector = get_frontal_face_detector();

        shape_predictor pose_model;

        deserialize("./shape_predictor_68_face_landmarks.dat") >> pose_model;

 

        // Keep track of the EAR values for the previous and current frame.

        double prev_ear = 0.0;

        double curr_ear = 0.0;

        //shape_predictor threat_model;

        //dlib::deserialize("C:/Users/pdm529913/Downloads/best_full_integer_quant_yolov10.tflite") >> threat_model;

 

        // Keep track of the number of consecutive frames where the eyes have been closed.

        int closed_frames = 0;

        int blink_count = 0;

 

        int drowsinessFrameCount = 0;

        int distractionFrameCount = 0;

        int recordFlag = 0;

        int recordFrameCount = 0;

 

        //--- INITIALIZE VIDEOWRITER

        cv::VideoWriter writer;

        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)

        double fps = 30.0;

 

        while (!win.is_closed())

        {

            cv::Mat temp;

            if (!cap.read(temp))

            {

                break;

            }

            time_t timestamp;

            time(&timestamp);

            char timestamp1[15];

            strftime(timestamp1, 15, "%Y%m%d%H%M%S", localtime(&timestamp));

            cv::Mat gray;

            cv::cvtColor(temp, gray, cv::COLOR_BGR2GRAY);

            dlib::array2d<unsigned char> cimg;

            dlib::assign_image(cimg, dlib::cv_image<unsigned char>(gray));

            //cv_image<bgr_pixel> cimg(temp);

 

            // Detect faces in the image

            std::vector<rectangle> faces = detector(cimg);

 

            // cout << "Number of faces detected: " << faces.size() << endl;

 

            for (const auto& face : faces)

            {

                full_object_detection shape = pose_model(cimg, face);

 

                // Get the landmarks for the left and right eyes.

                std::vector<point> left_eye_landmarks;

                std::vector<point> right_eye_landmarks;

                for (int i = 36; i <= 41; i++)

                {

                    left_eye_landmarks.push_back(shape.part(i));

                }

                for (int i = 42; i <= 47; i++)

                {

                    right_eye_landmarks.push_back(shape.part(i));

                }

 

                std::vector<point> face_l;

                face_l.push_back(shape.part(0));

                //cout << "Face_L: " << face_l[0] << endl;

                std::vector<point> face_r;

                face_r.push_back(shape.part(16));

                //cout << "Face_R: " << face_r[0] << endl;

                std::vector<point> face_c;

                face_c.push_back(shape.part(30));

                //cout << "Face_C: " << face_c[0] << endl;

                //cout << "Face_C Type: " << typeid(face_c[0](1)).name() << endl;

                double width_x = length(face_l[0] - face_r[0]);

                //cout << "Width_X: " << width_x << endl;

                std::vector<point> face_t;

                face_t.push_back(shape.part(19));

                std::vector<point> face_b;

                face_b.push_back(shape.part(8));

 

                // Calculate EAR for the left and right eyes.

                double left_ear = calculate_ear(left_eye_landmarks);

                double right_ear = calculate_ear(right_eye_landmarks);

 

                // Calculate the average EAR for both eyes.

                curr_ear = (left_ear + right_ear) / 2.0;

 

                // Print the current EAR value.

                // cout << "EAR: " << curr_ear << endl;

 

                // Check if the eyes are closed.

                bool closed_eyes = curr_ear < EYE_AR_THRESHOLD;

 

                if (closed_eyes)

                {

                    //closed_frames++;

                    drowsinessFrameCount++;

                    /*if (closed_frames >= 0)

                    {

                        blink_count++;

                        //cout << "Blink detected! Total blinks: " << blink_count << endl;

                        closed_frames = 0;

                    }   */

                }

                else

                {

                    //closed_frames = 0;

                    drowsinessFrameCount = 0;

                }

 

                if (drowsinessFrameCount >= EYE_AR_CONSEC_FRAMES) {

                    // Alert the driver and record the event

                    std::cout << "Drowsiness detected!" << std::endl;

                    //recordEvent(frame, "drowsiness_event.avi");

                    drowsinessFrameCount = 0;

                    cv::Point org1(1, 60);

                    cv::putText(temp, "Drowsiness!", org1, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

                    if (recordFlag == 0) {

                        recordFlag = 1;

                        recordFrameCount = 0;

                        bool isColor = (temp.type() == CV_8UC3);

                        string filename("./Event_");

                        filename = filename + timestamp1 + ".avi";             // name of the output video file

                        writer.open(filename, codec, fps, temp.size(), isColor);

                        // check if we succeeded

                        if (!writer.isOpened()) {

                            cerr << "Could not open the output video file for write\n";

                            return -1;

                        }

                        cout << "Writing videofile: " << filename << endl;

                    }

                }

 

                // Check for distraction

                if (!isLookingAtRoad(face_c, left_eye_landmarks, face_l, face_r, face_t, face_b) || !isLookingAtRoad(face_c, right_eye_landmarks, face_l, face_r, face_t, face_b)) {

                    distractionFrameCount++;

                }

                else {

                    distractionFrameCount = 0;

                }

 

                if (distractionFrameCount >= EYE_AR_CONSEC_FRAMES) {

                    // Alert the driver and record the event

                    std::cout << "Distraction detected!" << std::endl;

                    //recordEvent(frame, "distraction_event.avi");

                    cv::Point org2(1, 60);

                    cv::putText(temp, "Distraction!", org2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

                    distractionFrameCount = 0;

                    if (recordFlag == 0) {

                        recordFlag = 1;

                        recordFrameCount = 0;

                        bool isColor = (temp.type() == CV_8UC3);

                        string filename("./Event_");

                        filename = filename + timestamp1 + ".avi";             // name of the output video file

                        writer.open(filename, codec, fps, temp.size(), isColor);

                        // check if we succeeded

                        if (!writer.isOpened()) {

                            cerr << "Could not open the output video file for write\n";

                            return -1;

                        }

                        cout << "Writing videofile: " << filename << endl;

                    }

                }

 

                if (recordFlag == 1) {

                    // encode the frame into the videofile stream

                    cv::Point org(1, 30);

                    cv::putText(temp, timestamp1, org, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

                    if (drowsinessFrameCount >= 45) {

                        cv::Point org3(1, 60);

                        cv::putText(temp, "Drowsiness!", org3, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                    }

                    if (distractionFrameCount >= 45) {

                        cv::Point org4(1, 60);

                        cv::putText(temp, "Distraction!", org4, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                    }

                    writer.write(temp);

                    recordFrameCount++;

                    if (recordFrameCount == 750) {

                        recordFlag = 0;

                    }

                }                

 

                for (const auto& landmark : left_eye_landmarks)

                {

                    circle(temp, cv::Point(landmark.x(), landmark.y()), 2, cv::Scalar(0, 255, 0), -1);

                }

 

                // Draw circles around the landmarks of the right eye.

                for (const auto& landmark : right_eye_landmarks)

                {

                    circle(temp, cv::Point(landmark.x(), landmark.y()), 2, cv::Scalar(0, 255, 0), -1);

                }

 

                win.clear_overlay();

                win.set_image(cimg);

                win.add_overlay(render_face_detections(shape));

            }

 

        }

    }

    catch (exception& e)

    {

        cout << e.what() << endl;

    }

}
