#!/usr/bin/python3
import cv2
import numpy as np
import open3d as o3d
from pcd_processing_P6 import PointCloudProcessing
from openni import openni2
from openni import _openni2 as c_api


def main():

    # Initialize the depth stream
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))


    # Initialize OpenCV camera
    cap = cv2.VideoCapture(2)


    # Initialize Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=1000, left=900)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=np.array([0., 0., 0.]))
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-0.5, -0.5, -1], max_bound=[0.5, 0.5, 1])
    bbox.color = (1, 0, 0)


    # Main loop
    while vis.poll_events():

        # Depth image
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        img_depth = np.frombuffer(frame_data, dtype=np.uint16)
        img_depth.shape = (480, 640)
        img_depth = cv2.flip(img_depth, 1)


        # Color image
        _, img_color = cap.read()


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
        

        # Small transformation to help visualize and point camera
        pcd_vis = pcd.crop(bbox)
        pcd_vis.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])   # Invert up/down


        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(pcd_vis)
        vis.add_geometry(origin)
        vis.add_geometry(bbox)
        vis.update_renderer()


        # OpenCV Images
        center_x, center_y = img_color.shape[1] // 2, img_color.shape[0] // 2
        cv2.line(img_color, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
        cv2.line(img_color, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)

        img_depth_bgr = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR)
        cv2.normalize(img_depth_bgr, img_depth_bgr, 0, 255, cv2.NORM_MINMAX)
        img_depth_bgr = img_depth_bgr.astype(np.uint8)

        img_concat = np.vstack([img_color, img_depth_bgr])
        cv2.imshow("Color & Depth Images", img_concat)


        # Exit loop with enter/space/esc
        if cv2.waitKey(1) & 0xFF in [13, 32, 27]:
            break


    # Close all windows and unload/release devices
    cap.release()
    openni2.unload()
    cv2.destroyAllWindows()
    vis.destroy_window()


    # Show the final PCD
    o3d.visualization.draw_geometries([pcd, origin],
        zoom=0.4,
        front=[0, 0, -1],
        lookat=[0, 0, 1.5],
        up=[0, -1, 0])


if __name__ == "__main__":
    main()