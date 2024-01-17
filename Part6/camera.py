#!/usr/bin/python3
import cv2
import numpy as np
import open3d as o3d
from openni import openni2
from openni import _openni2 as c_api


def main():

    # Initialize the depth device
    openni2.initialize()
    dev = openni2.Device.open_any()

    # Start the depth stream
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

    # Initial OpenCV Window Functions
    cv2.namedWindow("Color Image")
    cv2.namedWindow("Depth Image")


    cap = cv2.VideoCapture(2)


    # Loop
    while True:

        # Depth image
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()

        img_depth = np.frombuffer(frame_data, dtype=np.uint16)
        img_depth.shape = (1, 480, 640)
        img_depth = np.concatenate((img_depth, img_depth, img_depth), axis=0)
        img_depth = np.swapaxes(img_depth, 0, 2)
        img_depth = np.swapaxes(img_depth, 0, 1)


        # Color image
        _, img_color = cap.read()
        img_color = cv2.flip(img_color, 1)


        # Show Images
        cv2.imshow("Depth Image", img_depth)
        cv2.imshow("Color Image", img_color)


        # Close with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Close all windows and unload/release devices
    cap.release()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()