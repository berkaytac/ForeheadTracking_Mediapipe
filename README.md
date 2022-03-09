# Face and Forehead Tracking with Mediapipe

The purpose of this side project is to extract the forehead information using Mediapipe.
[MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)  is a face geometry solution that estimates 468 3D face landmarks in real-time even on mobile devices.
Forehead image information is extracted throughtout the video and can be used in RPPG modules or any other applications. 
This program works also in real time for any **webcam-based Forehead Tracking Applications**. 

[![Forehead](https://media.giphy.com/media/czHiLlOv6ooXaMjE3s/giphy.gif)](https://www.youtube.com/watch?v=XBjXK__qQTM)

## Installation

- Install these dependencies (imutils, Numpy, Mediapipe, Opencv-Python):

```
pip install -r requirements.txt
```
- Under the main() function, change source name to;
    - "0" for Real-Time Webcam Forehead Tracking,
    - Path of video file for Forehead Tracking from a video.

```
# Source video
source = 0  # For webcam
# OR
source = "source_vid.avi"

```

Run the Forehead Tracking file:

```
python forehead_facemesh.py
```

"output_video.mp4" output file will be created in the source folder.



## Analyze Frame

```python
def analyze(self, img, face_draw=False, landmark_draw=False, fh_draw=True):
    '''
    :param img: video frame from source. 
    :param face_draw: True if you want to draw face bounding box on the output video. Default:False.
    :param landmark_draw: True if you want landmarks and connections drawing on the output video. Default:False. 
    :param fh_draw: True if you want forehead rectangle drawing on the output video. Default:True.
    :return: output image
    '''
frame = detector.analyze(frame, False, False, True)  # frame, face, Landmark, forehead
```
Analyze frame calls two functions with given parameters explained above; 
- FindFaceMesh which estimates the 468 landmarks on the face and detects the face bounding box.
- Get_forehead_coordinates which derive/draw contours of the forehead area.

### You want to help?
Your suggestions, bugs reports and pull requests are welcome and appreciated. You can also star ⭐️ the project!

