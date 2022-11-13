import streamlit as st 
import logging
import queue
import threading
import av
import cv2

from pytorch_infer import *

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)


## Config WEBRTC 
logger = logging.getLogger(__name__)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
        
#st.title("FaceMask Detection SYNAPSE")
#webrtc_streamer(key='sample')
#st.write("FaceMask Detection")

def main():
    st.header("FaceMask Detection Demo V1")

    pages = {
                # "Simple": simple,
                "FaceMask Detection": facemask_detection
            }

    page_titles = pages.keys()
    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def simple():
    webrtc_streamer(key='loopback')

def facemask_detection():

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        conf_thresh = 0.7

        inference(image,
            conf_thresh,
            iou_thresh=0.6,
            target_shape=(360, 360),
            draw_result=True,
            show_result=False)

        return av.VideoFrame.from_ndarray(image, format="rgb24")

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == '__main__':
    main()
