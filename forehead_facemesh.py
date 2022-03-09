import cv2
import imutils
import mediapipe as mp
import time
import numpy as np


class ForeheadDetector:

    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def draw_text(self, frame, text, col, text_pos=(10, 10), font_scale=0.75, font_thickness=2):
        """
        Creates a background for the text depending on the text size and puts the text on the image.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color_bg = (0, 0, 0)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(frame, text_pos, (text_pos[0] + text_w, text_pos[1] + text_h), text_color_bg, -1)
        cv2.putText(frame, text, (text_pos[0], text_pos[1] + text_h), font, font_scale, col, font_thickness)
        return frame

    def findFaceMesh(self, img, face_draw=False, landmark_draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.landmark_results = self.faceMesh.process(self.imgRGB)
        if self.landmark_results.multi_face_landmarks:
            if face_draw:
                self.face_results = self.faceDetection.process(self.imgRGB)
                for detection in self.face_results.detections:
                    self.mpDraw.draw_detection(img, detection, self.drawSpec)
            if landmark_draw:
                self.mpDraw.draw_landmarks(img, self.landmark_results.multi_face_landmarks[0],
                                           self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
            # ih, iw, ic = img.shape
            # fx, fy = int(faceLms.landmark[151].x * iw) , int(faceLms.landmark[151].y * ih)
            # cv2.circle(img , (fx,fy), 10,(0,0,255),1)
            # for id, lm in enumerate(faceLms.landmark):
            #     # print(lm)
            #     ih, iw, ic = img.shape
            #     x, y = int(lm.x * iw), int(lm.y * ih)
            #     cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
            #               0.7, (0, 255, 0), 1)
            #
            #     print(id,x,y)
        return img

    def get_forehead_coordinates(self, img, draw=True):
        """
        landmark numbers
                109     10      338
                108     151     337
                107     9       336
        """
        if self.landmark_results.multi_face_landmarks:
            faceLms = self.landmark_results.multi_face_landmarks[0].landmark
            top_left = x_109, y_109 = int(faceLms[109].x * self.img_width), int(faceLms[109].y * self.img_height)
            mid_left = x_108, y_108 = int(faceLms[108].x * self.img_width), int(faceLms[108].y * self.img_height)
            x_107, y_107 = int(faceLms[107].x * self.img_width), int(faceLms[107].y * self.img_height)
            top_right = x_338, y_338 = int(faceLms[338].x * self.img_width), int(faceLms[338].y * self.img_height)
            mid_right = x_337, y_337 = int(faceLms[337].x * self.img_width), int(faceLms[337].y * self.img_height)
            x_336, y_336 = int(faceLms[336].x * self.img_width), int(faceLms[336].y * self.img_height)
            self.forecentx, self.forecenty = x_151, y_151 = int(faceLms[151].x * self.img_width), int(
                faceLms[151].y * self.img_height)
            bottom_left = ((x_108 + x_107) // 2), ((y_108 + y_107) // 2)
            bottom_right = ((x_336 + x_337) // 2), ((y_337 + y_336) // 2)

            points = [top_left, top_right, bottom_right, bottom_left]
            self.forehead_points = np.array(points).astype("int").reshape(4, 2)
            cv2.drawContours(img, [self.forehead_points], 0, (0, 0, 255), 2)
            text = "Forehead Center: {}, {}".format(str(self.forecentx), str(self.forecenty))
            img = self.draw_text(img, text, (0, 70, 255), (10, 30), 0.75, 1)
        return img

    def analyze(self, img, face_draw=False, landmark_draw=False, fh_draw=True):
        """
        :param img: video frame from source.
        :param face_draw: True if you want to draw face bounding box on the output video. Default:False.
        :param landmark_draw: True if you want landmarks and connections drawing on the output video. Default:False.
        :param fh_draw: True if you want forehead rectangle drawing on the output video. Default:True.
        :return: output image
        """
        self.img_height, self.img_width, _ = img.shape
        # Find Face Mesh
        img = self.findFaceMesh(img, face_draw, landmark_draw)
        # Extract Forehead
        img = self.get_forehead_coordinates(img, fh_draw)
        return img


def main():
    # Source video
    # source = 0  # For webcam
    source = "yatay2.avi"  # For video file

    cap = cv2.VideoCapture(source)

    # output file
    video_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(video_file, fourcc, fps, (w, h))

    # initialize the tracker
    detector = ForeheadDetector(maxFaces=1)

    pTime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = detector.analyze(frame, False, False, True)  # frame, face, Landmark, forehead
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            detector.draw_text(frame, (f'FPS: {int(fps)}'), (0, 255, 0), (10, 10), 0.75, 1)
            video_out.write(frame)
            frame = imutils.resize(frame, width=640)
            cv2.imshow("Forehead Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video_out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
