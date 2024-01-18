#!/usr/bin/python3
import cv2
import numpy as np
import os

import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering

from openni import openni2
from openni import _openni2 as c_api


def pcdFromCamera(camera_id):

    # Initialize the depth stream
    try:
        openni2.initialize()
        dev = openni2.Device.open_any()
    except Exception:
        print('\nError: RGBD camera not found!')
        raise SystemExit
    
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))


    # Initialize OpenCV
    cap = cv2.VideoCapture(camera_id)
    cv2.namedWindow('Color & Depth Images')
    cv2.moveWindow('Color & Depth Images', 500, 20)


    # Main loop
    print('Opening camera... Press ESC to close windows and confirm scene')
    while True:

        # Depth image
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        img_depth = np.frombuffer(frame_data, dtype=np.uint16)
        img_depth.shape = (480, 640)
        img_depth = cv2.flip(img_depth, 1)


        # Color image
        _, img_color = cap.read()
        if img_color is None:
            print('\nError: RGBD camera disconnected!')
            cap.release()
            openni2.unload()
            cv2.destroyAllWindows()

            raise SystemExit


        # Draw a crossair in the color image
        center_x, center_y = img_color.shape[1] // 2, img_color.shape[0] // 2
        cv2.line(img_color, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
        cv2.line(img_color, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)


        # Concatenate images to use just 1 window
        img_depth_bgr = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR)
        cv2.normalize(img_depth_bgr, img_depth_bgr, 0, 255, cv2.NORM_MINMAX)
        img_depth_bgr = img_depth_bgr.astype(np.uint8)
        img_concat = np.vstack([img_color, img_depth_bgr])


        # Show image
        cv2.imshow("Color & Depth Images", img_concat)


        # Exit loop with enter/space/esc
        if cv2.waitKey(1) & 0xFF in [13, 32, 27]:
            break


    # Close all windows and unload/release devices
    cap.release()
    openni2.unload()
    cv2.destroyAllWindows()


    # Convert to Open3D images
    color_raw = o3d.geometry.Image(img_color)
    depth_raw = o3d.geometry.Image((img_depth * 0.1).astype(np.uint16))     # Scaled down to 10%


    # Create RGBDImage
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)


    # Create PointCloud from RGBDImage
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


    # Prepare visualization elements
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-0.5, -0.5, 0], max_bound=[0.5, 0.5, 1])
    bbox.color = (1, 0, 0)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    pcd = pcd.crop(bbox)


    # Show the final PCD
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Open3D", 1024, 768)   # 4x3
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry(f'pcd', pcd, rendering.MaterialRecord())
    scene.scene.add_geometry(f'bbox', bbox, rendering.MaterialRecord())
    scene.scene.add_geometry(f'origin', origin, rendering.MaterialRecord())

    scene.setup_camera(60, bbox, [0, 0, 0])
    scene.look_at(np.array([0, 0, 0.5], dtype=np.float32),  # lookat
                  np.array([0, 0, -1], dtype=np.float32),   # front
                  np.array([0, -1, 0], dtype=np.float32))   # up
    gui.Application.instance.run()

    # o3d.visualization.draw_geometries([pcd, origin, bbox],
    #     zoom=0.75,
    #     front=[0, 0, -1],
    #     lookat=[0, 0, 0.5],
    #     up=[0, -1, 0])

    return pcd


def initCamera(pcd_path):

    pcd = pcdFromCamera(2)

    print(f'\nSaving a point cloud with {len(pcd.points)} points...')

    if os.path.exists(pcd_path):
        os.remove(pcd_path)
    o3d.io.write_point_cloud(pcd_path, pcd)




if __name__ == "__main__":
    initCamera('Camera/temp.pcd')