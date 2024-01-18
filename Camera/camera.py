#!/usr/bin/python3
import cv2
import numpy as np
import os

import open3d as o3d
# from open3d.visualization import gui
# from open3d.visualization import rendering

from openni import openni2
from openni import _openni2 as c_api


def convert_pcd_to_off(pcd_file, off_file):
    # Read PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Downsample the point cloud (optional, but can be useful for large point clouds)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # Estimate normals for the point cloud
    pcd.estimate_normals()

    # Create a surface mesh using Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    # Save as OFF file
    o3d.io.write_triangle_mesh(off_file, mesh)

class PlaneDetection:
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):
        self.inlier_cloud.paint_uniform_color([r, g, b])

    def segment(self, distance_threshold=0.03, ransac_n=4, num_iterations=200):    # Find plane

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=False)

        return outlier_cloud

    def __str__(self):

        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) + ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0'
        return text




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


    # Initialize Open3D visualization window
    # gui.Application.instance.initialize()
    # window = gui.Application.instance.create_window("Camera Point Cloud", 1000, 900)
    # scene = gui.SceneWidget()
    # scene.scene = rendering.Open3DScene(window.renderer)
    # window.add_child(scene)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-1, -1, 0], max_bound=[1, 1, 2])
    bbox.color = (1, 0, 0)
    # scene.scene.add_geometry('bbox', bbox, rendering.MaterialRecord())

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    # scene.scene.add_geometry('origin', origin, rendering.MaterialRecord())

    # scene_bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    # scene.setup_camera(60, scene_bounds, [0, 0, 0])


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


        # Convert to Open3D images
        color_raw = o3d.geometry.Image(cv2.flip(img_color, -1))
        depth_raw = o3d.geometry.Image((cv2.flip(img_depth, -1) * 0.1).astype(np.uint16))     # Scaled down to 10%


        # Create RGBDImage
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)


        # Create PointCloud from RGBDImage
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        

        # Small transformation to help visualize and point camera
        pcd = pcd.crop(bbox)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])   # Invert up/down


        # # Update visualization
        # scene.scene.remove_geometry('pcd')
        # scene.scene.add_geometry('pcd', pcd, rendering.MaterialRecord())
        # gui.Application.instance.run_one_tick()
        # scene.force_redraw()


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


    # Show the final PCD
    o3d.visualization.draw_geometries([pcd, origin, bbox],
        zoom=0.4,
        front=[0, 0, -1],
        lookat=[0, 0, 1],
        up=[0, 1, 0])
    return pcd


def initCamera(pcd_path):

    pcd = pcdFromCamera(2)

    print(f'\nSaving a point cloud with {len(pcd.points)} points...')

    if os.path.exists(pcd_path):
        os.remove(pcd_path)
    o3d.io.write_point_cloud(pcd_path, pcd)




if __name__ == "__main__":
    initCamera()