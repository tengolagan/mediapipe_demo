import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image 
import mediapipe as mp

import os 
import csv

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

)

@st.cache()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is not None and height is None:
        return image 
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    # resize the image 
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

st.sidebar.subheader('Solutions')
app_mode = st.sidebar.selectbox("Choose a mode", 
['Face Detection',
'Face Mesh', 
'Hands',
'Pose',
'Holistic'])

if app_mode == 'Face Detection':

    st.header('Face Detection')
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    st.sidebar.markdown('---')

    # Configuration Options 
    st.sidebar.subheader('Configuration Options :   ' + app_mode)

    model_selection = st.sidebar.number_input('Model Selection', value = 0, min_value = 0, max_value = 1)
    
    detection_confidence = st.sidebar.slider('Minimum Detection Confidence', min_value = 0.0, max_value = 1.0, value = 0.5)

    if record:
        st.checkbox('Recording', value = True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

    )

    st.markdown("## Output")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4', 'mov', 'avi', 'm4v', 'asf'])
    tffile = tempfile.NamedTemporaryFile(delete=False) #temporary file 

    ## get input video 
    if video_file_buffer is not None:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    
    else:
        #if use_webcam: #if button pressed
        vid = cv2.VideoCapture(0)
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second 

    #Recording Part 
    codec = cv2.VideoWriter_fourcc('m', 'p', '4',  'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input video')
    st.sidebar.video(tffile.name)

    fps = 0 # frames per seconds
    i = 0 # iterations 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")
    
    # Face Detection Predictor 

    with mp_face_detection.FaceDetection(
        model_selection = model_selection,
        min_detection_confidence = detection_confidence) as face_detection:
        
        prevTime = 0

        while vid.isOpened:
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue 
            results = face_detection.process(frame)
            frame.flags.writeable = True

            face_count = 0 

            if results.detections:
                for detection in results.detections:
                    face_count +=1 
                    mp_drawing.draw_detection(frame, detection)

            # FPS Counter Logic 
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record: 
                out.write(frame)
            
            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html = True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
    
    
    st.text("Video Processed")
    output_video = open("output1.mp4")
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()


#--------------------------------------------------

elif app_mode == 'Face Mesh':

    st.header('Face Mesh')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    st.sidebar.markdown('---')

    # Configuration Options 
    st.sidebar.subheader('Configuration Options :   ' + app_mode)

    MAX_NUM_FACES = st.sidebar.number_input('Maximum number of faces to detect', value = 2, min_value = 1)
    
    MIN_DETECTION_CONFIDENCE = st.sidebar.slider('Minimum confidence value', value = 0.5, min_value = 0.0, max_value = 1.0)

    MIN_TRACKING_CONFIDENCE = st.sidebar.slider('Minimum tracking value', value = 0.5, min_value = 0.0, max_value = 1.0)

    if record:
        st.checkbox('Recording', value = True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

    )
    st.markdown("## Output")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4', 'mov', 'avi', 'm4v', 'asf'])
    tffile = tempfile.NamedTemporaryFile(delete=False) #temporary file 

    ## get input video 
    if video_file_buffer is not None:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    
    else:
        #if use_webcam: #if button pressed
        vid = cv2.VideoCapture(0)
        
        #else:
        #    vid = cv2.VideoCapture(DEMO_FACE_VIDEO)
        #    tffile.name = DEMO_FACE_VIDEO
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second 

    #Recording Part 
    codec = cv2.VideoWriter_fourcc('m', 'p', '4',  'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input video')
    st.sidebar.video(tffile.name)

    fps = 0 # frames per seconds
    i = 0 # iterations 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")
    
    # Face Detection Predictor 

    with mp_face_mesh.FaceMesh(
        max_num_faces= MAX_NUM_FACES,
        refine_landmarks= True,
        min_detection_confidence= MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as face_mesh:
        
        prevTime = 0

        while vid.isOpened:
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue 
            results = face_mesh.process(frame)
            frame.flags.writeable = True

            face_count = 0 

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count +=1 
                    
                    mp_drawing.draw_landmarks(
                        image = frame, 
                        landmark_list = face_landmarks, 
                        connections = mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # FPS Counter Logic 
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record: 
                out.write(frame)
            
            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html = True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
    
    
    st.text("Video Processed")
    output_video = open("output1.mp4")
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()
#--------------------------------------------------

elif app_mode == 'Hands':

    st.header('Hands')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    st.sidebar.markdown('---')

    # Configuration Options 
    st.sidebar.subheader('Configuration Options :   ' + app_mode)

    MAX_NUM_HANDS = st.sidebar.number_input('Maximum number of hands to detect', value = 2, min_value = 1)
    
    MIN_DETECTION_CONFIDENCE = st.sidebar.slider('Minimum confidence value', value = 0.5, min_value = 0.0, max_value = 1.0)

    MIN_TRACKING_CONFIDENCE = st.sidebar.slider('Minimum tracking value', value = 0.5, min_value = 0.0, max_value = 1.0)

    if record:
        st.checkbox('Recording', value = True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

    )
    st.markdown("## Output")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4', 'mov', 'avi', 'm4v', 'asf'])
    tffile = tempfile.NamedTemporaryFile(delete=False) #temporary file 

    ## get input video 
    if video_file_buffer is not None:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    
    else:
        #if use_webcam: #if button pressed
        vid = cv2.VideoCapture(0)
        
        #else:
        #    vid = cv2.VideoCapture(DEMO_HAND_VIDEO)
        #    tffile.name = DEMO_HAND_VIDEO
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second 

    #Recording Part 
    codec = cv2.VideoWriter_fourcc('m', 'p', '4',  'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input video')
    st.sidebar.video(tffile.name)

    fps = 0 # frames per seconds
    i = 0 # iterations 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Detected Hands**")
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")
    
    # Face Detection Predictor 

    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:
        
        prevTime = 0

        while vid.isOpened:
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue 
            results = hands.process(frame)
            frame.flags.writeable = True

            hand_count = 0 

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    hand_count +=1 
                    
                    mp_drawing.draw_landmarks(
                        image = frame, 
                        landmark_list = hand_landmarks, 
                        connections = mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style())
            
            # FPS Counter Logic 
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record: 
                out.write(frame)
            
            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{hand_count}</h1>", unsafe_allow_html = True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html = True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
    
    
    st.text("Video Processed")
    output_video = open("output1.mp4")
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()


#--------------------------------------------------

elif app_mode == 'Pose':

    st.header('Pose')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose


    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    st.sidebar.markdown('---')

    # Configuration Options 
    st.sidebar.subheader('Configuration Options :   ' + app_mode)
    
    MIN_DETECTION_CONFIDENCE = st.sidebar.slider('Minimum confidence value', value = 0.5, min_value = 0.0, max_value = 1.0)

    MIN_TRACKING_CONFIDENCE = st.sidebar.slider('Minimum tracking value', value = 0.5, min_value = 0.0, max_value = 1.0)

    if record:
        st.checkbox('Recording', value = True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

    )
    st.markdown("## Output")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4', 'mov', 'avi', 'm4v', 'asf'])
    tffile = tempfile.NamedTemporaryFile(delete=False) #temporary file 

    ## get input video 
    if video_file_buffer is not None:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    
    else:
        #if use_webcam: #if button pressed
        vid = cv2.VideoCapture(0)
        
        #else:
        #    vid = cv2.VideoCapture(DEMO_BODY_VIDEO)
        #   tffile.name = DEMO_BODY_VIDEO
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second 

    #Recording Part 
    codec = cv2.VideoWriter_fourcc('m', 'p', '4',  'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input video')
    st.sidebar.video(tffile.name)

    fps = 0 # frames per seconds
    i = 0 # iterations 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Image Width**")
        kpi2_text = st.markdown("0")
    
    # Face Detection Predictor 

    with mp_pose.Pose(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:
        
        prevTime = 0

        while vid.isOpened:
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue 
            results = pose.process(frame)
            frame.flags.writeable = True

            if results.pose_landmarks:
                
                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = results.pose_landmarks, 
                    connections = mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            
            # FPS Counter Logic 
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record: 
                out.write(frame)
            
            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html = True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
    
    
    st.text("Video Processed")
    output_video = open("output1.mp4")
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

#--------------------------------------------------


elif app_mode == 'Holistic':

    st.header('Holistic')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic


    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    st.sidebar.markdown('---')

    # Configuration Options 
    st.sidebar.subheader('Configuration Options :   ' + app_mode)
    
    MIN_DETECTION_CONFIDENCE = st.sidebar.slider('Minimum confidence value', value = 0.5, min_value = 0.0, max_value = 1.0)

    MIN_TRACKING_CONFIDENCE = st.sidebar.slider('Minimum tracking value', value = 0.5, min_value = 0.0, max_value = 1.0)

    SMOOTH_LANDMARKS = True 

    if record:
        st.checkbox('Recording', value = True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,

    )
    st.markdown("## Output")
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4', 'mov', 'avi', 'm4v', 'asf'])
    tffile = tempfile.NamedTemporaryFile(delete=False) #temporary file 

    ## get input video 
    if video_file_buffer is not None:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    
    else:
        #if use_webcam: #if button pressed
        vid = cv2.VideoCapture(0)
        
        #else:
        #    vid = cv2.VideoCapture(DEMO_BODY_VIDEO)
        #    tffile.name = DEMO_BODY_VIDEO
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second 

    #Recording Part 
    codec = cv2.VideoWriter_fourcc('m', 'p', '4',  'v')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input video')
    st.sidebar.video(tffile.name)

    fps = 0 # frames per seconds
    i = 0 # iterations 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Image Width**")
        kpi2_text = st.markdown("0")
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        prevTime = 0

        while vid.isOpened:
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue 
            results = holistic.process(frame)
            frame.flags.writeable = True

            if results.face_landmarks:
                
                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = results.face_landmarks,
                    connections = mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            if results.pose_landmarks:

                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = results.pose_landmarks,
                    connections = mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # FPS Counter Logic 
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            if record: 
                out.write(frame)
            
            ## Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html = True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
    
    
    

    st.text("Video Processed")
    output_video = open("output1.mp4")
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

#--------------------------------------------------
st.sidebar.markdown('---')
button = st.sidebar.button('About MediaPipe')
if button :

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">Media Pipe</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.subheader('**MediaPipe** is an open-source framework from Google for building multimodal (eg. video, audio, any time series data), cross platform (i.e Android, iOS, web, edge devices) applied ML pipelines. It is performance optimized with end-to-end on device inference in mind.')
    st.video('https://www.youtube.com/watch?v=V9CiJhHQKkc')
