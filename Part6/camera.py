#!/usr/bin/python3
import cv2
import numpy as np
import open3d as o3d
from pcd_processing_P6 import PointCloudProcessing
from openni import openni2
from openni import _openni2 as c_api


view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.41590951312155949, 0.97330287524632042, -0.60100001096725464 ],
			"boundingbox_min" : [ -1.815977123124259, -0.56555715061369394, -2.9839999675750732 ],
			"field_of_view" : 60.0,
			"front" : [ -0.78974422652934717, -0.027044285339143485, 0.61283983494389316 ],
			"lookat" : [ -0.022561790241817495, -0.25108904807479676, -0.22964330450237716 ],
			"up" : [ 0.18644403931156128, 0.94118476024943876, 0.28179756436029674 ],
			"zoom" : 0.35999999999999854
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


# ----------------------------------
# Functions and Classes to Pointcloud processing
# ----------------------------------

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
        
        # Creating Pointcloud from camera RGB-D image

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_color, img_depth)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Flip the point cloud to handle orientation issues
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # ---------------------------------------------------
        # PointCloud Processing
        # ---------------------------------------------------



        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        entities = [pcd]

        o3d.visualization.draw_geometries(entities,
                                            zoom=view['trajectory'][0]['zoom'],
                                            front=view['trajectory'][0]['front'],
                                            lookat=view['trajectory'][0]['lookat'],
                                            up=view['trajectory'][0]['up'])


    # Close all windows and unload/release devices
    cap.release()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()