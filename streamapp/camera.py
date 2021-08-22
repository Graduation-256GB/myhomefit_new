from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import mediapipe as mp
import time
import math
from PIL import ImageFont, ImageDraw, Image

face_detection_videocam = cv2.CascadeClassifier(os.path.join(
            settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
            settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([settings.BASE_DIR, "face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR,"face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))
##
poseEstimationModel = load_model(os.path.join(settings.BASE_DIR,'face_detector/model_django.h5'))

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        frame_flip = cv2.flip(image,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()


class PoseWebCam(object):
    def __init__(self):
        # self.vs = VideoStream(src=0).start()
        self.cap = cv2.VideoCapture(0)
        # self.mpPose = mp.solutions.pose
        self.mpPose = mp.solutions.mediapipe.python.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.pTime = 0

        self.frame_cnt=0
        self.allkeypoints=[]
        self.outputkeypoints=[]

        """
        # mediapipe 키포인트 33개 중에서내 사용될 12개의 키포인트
        self.skeleton = {'Right Shoulder': 12, 'Right Elbow': 14, 'Right Wrist': 16, 'Left Shoulder': 11, 'Left Elbow': 13,
                            'Left Wrist': 15, 'Right Hip': 24, 'Right Knee': 26, 'Right Ankle': 28, 'Left Hip': 23,
                            'Left Knee': 25, 'Left Ankle': 27}
        """

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):

        success, img = self.cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        # print(results.pose_landmarks.landmark[0])

        keypoints=[] # 1프레임의 keypoints를 담은 배열
        # keypoints.add([results.pose_landmarks.landmark[0]])

        if results.pose_landmarks:
            self.frame_cnt+=1

            #print(results.pose_landmarks.landmark[0])

            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w,c = img.shape
                
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

                keypoints.append((cx, cy))  # 1프레임에 33개의 keypoints값 차례로 넣어줌.

            # if self.frame_cnt < 17 : # 나머지 이용 
            ## self.allkeypoints.append(keypoints) # 프레임별 keypoints가 모두 있는 배열
            self.allkeypoints.append(keypoints) ## input을 계산하는데 필요한 12 points만 append 함
            
            if len(self.allkeypoints)==16: # 배열의 길이는 항상 16개를 유지 
                # self.outputkeypoints=[self.allkeypoints]  # 단지, 3차원 배열로 만들어주기 위함(이전까지는 2차원 배열)
                #                                             (수정) => get_input()에서 3차원으로 입력층을 생성
                self.outputkeypoints = self.allkeypoints
                # self.get_keypoints() # 프레임 수가 16개가 되면, 16개의 프레임에 대한 keypoints가 모여있는 배열 반환해주는 함수

                predicted_pose = self.detect_and_predict_pose() ## 예측된 포즈(라벨)

                ## 예측된 포즈(라벨) 출력
                """
                font = ImageFont.truetype("fonts/gulim.ttc", 20)
                img = np.full((200, 300, 3), (255, 255, 255), np.uint8)
                img = Image.fromarray(img)

                draw = ImageDraw(img)

                text = predicted_pose
                draw.text((30, 50), text, font=font, fill=(0,0,0))

                img = np.array(img)
                cv2.imshow("text", img)
                """
                cv2.putText(img, predicted_pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                frame_flip = cv2.flip(img,1)
                ret, jpeg = cv2.imencode('.jpg', frame_flip)

                self.allkeypoints=[] # 배열 초기화  
           
            # 제대로 만들었는지 확인하기 위한 print문 (cmd창 참고)
            # print(self.frame_cnt)
            #print(len(self.allkeypoints))

            # print(len(self.allkeypoints[0]))

            #print(self.allkeypoints)

        cTime = time.time()
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime

        ## cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
        frame_flip = cv2.flip(img,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()

    # 16개의 프레임에서 keypoints를 모두 모아서 반환해주는 함수 (3차원 배열 형태) -> ## 2차원으로 수정
    def get_keypoints(self):
        #print("get_keypoints 호출!")
        #print(self.outputkeypoints)
        return self.outputkeypoints

    ## 예측 값에 해당하는 라벨(한글) 반환하는 함수
    def detect_and_predict_pose(self):
        """
        poses = { 0: "스탠딩 사이드 크런치",
                    1: "스탠딩 니업",
                    2: "버피 테스트",
                    3: "스텝 포워드 다이나믹 런지",
                    4: "스텝 백워드 다이나믹 런지",
                    5: "사이드 런지",
                    6: "크로스 런지",
                    7: "굿모닝"
                  }
        """

        poses = {0: "STANDING SIDE CRUNCH",
                 1: "STANDING KNEE UP",
                 2: "BURPEE TEST",
                 3: "STEP FORWARD DYNAMIC LUNGE",
                 4: "STEP BACKWARD DYNAMIC LUNGE",
                 5: "SIDE LUNGE",
                 6: "CROSS LUNGE",
                 7: "GOODMORNING"
                 }
        inputs = np.array(self.get_input(), dtype="float32")
        preds = poseEstimationModel.predict(inputs, batch_size=32)
        label = poses[np.argmax(preds)]

        return label

    # 1개 프레임의 관절값에 대해 angle, ratio, vertical, ratioavg 계산하는 함수
    def calc_one_frame(self, points):
        # 각도를 재는 부위 : 각도를 재는데 필요한 부위 3개 -> mediapipe 키포인트에 맞춰서 수정
        ANGLE_PAIR = {
            "leftArmpit": [11, 13, 23],
            "rightArmpit": [12, 14, 24],
            'LeftShoulder': [11, 12, 23],
            'RightShoulder': [12, 11, 24],
            'leftElbow': [13, 11, 15],
            'rightElbow': [14, 12, 16],
            'leftHip': [23, 24, 11],
            'RightHip': [24, 23, 12],
            'leftGroin': [23, 25, 27],
            'rightGroin': [24, 26, 28],
            'leftKnee': [25, 27, 23],
            'rightKnee': [26, 28, 24]
        }

        # Angle
        parts = []
        for pair1 in ANGLE_PAIR:  # 각도 계산하는 부위

            slope = 0  # slope 초기화

            partA = ANGLE_PAIR[pair1][0]  # 각도를 계산하는데 필요한 부위 1
            partB = ANGLE_PAIR[pair1][1]  # 각도를 계산하는데 필요한 부위 2
            partC = ANGLE_PAIR[pair1][2]  # 각도를 계산하는데 필요한 부위 3
            if points[partA] and points[partB] and points[partC]:
                line_1_2 = math.sqrt(
                    math.pow(points[partA][0] - points[partB][0], 2) + math.pow(points[partA][1] - points[partB][1], 2))
                line_1_3 = math.sqrt(
                    math.pow(points[partA][0] - points[partC][0], 2) + math.pow(points[partA][1] - points[partC][1], 2))
                line_2_3 = math.sqrt(
                    math.pow(points[partB][0] - points[partC][0], 2) + math.pow(points[partB][1] - points[partC][1], 2))

                try:
                    radians = math.acos(
                        (math.pow(line_1_2, 2) + math.pow(line_1_3, 2) - math.pow(line_2_3, 2)) / (
                                    2 * line_1_2 * line_1_3))
                except ZeroDivisionError as e:
                    radians = 0
                parts.append((radians * 180) / math.pi)
            else:
                parts.append(0)

        # Slope
        hipSlope = 0
        shoulderSlope = 0

        ## openpose skeleton 인덱스 에서 mediapipe skeleton 인덱스로 바꿈
        part_5 = self.change_part(5)
        part_11 = self.change_part(11)
        part_12 = self.change_part(12)
        part_2 = self.change_part(2)
        part_8 = self.change_part(8)
        part_9 = self.change_part(9)

        if points[part_2][0] - points[part_9][0] == 0 or points[part_8][0] - points[part_9][0] == 0:
            slope = -9999
        else:
            shoulderSlope = abs((
                    (points[part_2][1] - points[part_9][1]) / (points[part_2][0] - points[part_9][0])
            ))
            hipSlope = abs((
                    (points[part_8][1] - points[part_9][1]) / (points[part_8][0] - points[part_9][0])
            ))

        slope = min(shoulderSlope, hipSlope)

        # Vertical
        if math.atan(slope) > 0.87:
            verticalPose = 1000
        else:
            verticalPose = -1000

        # ratioAvg
        leftTop = math.sqrt(
            math.pow(points[part_5][0] - points[part_11][0], 2) + math.pow(points[part_5][1] - points[part_11][1], 2))

        leftBottom = math.sqrt(
            math.pow(points[part_11][0] - points[part_12][0], 2) + math.pow(points[part_11][1] - points[part_12][1], 2))

        rightTop = math.sqrt(
            math.pow(points[part_2][0] - points[part_8][0], 2) + math.pow(points[part_2][1] - points[part_8][1], 2))

        rightBottom = math.sqrt(
            math.pow(points[part_8][0] - points[part_9][0], 2) + math.pow(points[part_8][1] - points[part_9][1], 2))

        ratioAvg = (leftTop / leftBottom + rightTop / rightBottom) / 2

        parts.append(slope)
        parts.append(verticalPose)
        parts.append(ratioAvg)

        return parts

    ## 하나의 입력층 반환하는 함수
    def get_input(self):
        inputs = []
        for one_frame in self.get_keypoints():
            inputs.append(self.calc_one_frame(one_frame))
        return [inputs]

    ## openpose skeleton 인덱스 에서 mediapipe skeleton 인덱스로 바꾸는 함수(기존 계산 코드를 편리하게 적용하기 위함)
    def change_part(self, openpose_index):
        skeleton_dic = {2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 8: 24, 9: 26, 10: 28, 11: 23, 12: 25, 13: 27}
        return skeleton_dic[openpose_index]

class IPWebCam(object):
    def __init__(self):
        self.url = "http://192.168.0.100:8080/shot.jpg"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        imgResp = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img= cv2.imdecode(imgNp,-1)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR) 
        frame_flip = cv2.flip(resize,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()


class MaskDetect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        cv2.destroyAllWindows()

    def detect_and_predict_mask(self,frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    def get_frame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=650)
        frame = cv2.flip(frame, 1)
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
class LiveWebCam(object):
    def __init__(self):
        self.url = cv2.VideoCapture("rtsp://admin:Mumbai@123@203.192.228.175:554/")

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        success,imgNp = self.url.read()
        resize = cv2.resize(imgNp, (640, 480), interpolation = cv2.INTER_LINEAR) 
        ret, jpeg = cv2.imencode('.jpg', resize)
        return jpeg.tobytes()
