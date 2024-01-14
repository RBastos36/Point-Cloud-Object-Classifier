#!/usr/bin/env python3

import math
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pcd_processing_testing import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate
import os
import glob


# Default view
view = {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 0.69999999999999996, 0.69999999999999996, 0.5 ],
                    "boundingbox_min" : [ -0.69999999999999996, -0.69999999999999996, -0.25 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.48357815199197129, 0.61548483867363935, 0.62235888704100151 ],
                    "lookat" : [ 0.25470084286906458, 0.23151583259577294, 0.25384666908559167 ],
                    "up" : [ -0.40379961065115821, -0.47397267466848536, 0.78249330866504885 ],
                    "zoom" : 0.87999999999999901
                }
            ],
            "version_major" : 1,
            "version_minor" : 0
        }

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

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    p = PointCloudProcessing()


    # Get scene datapath and load scene

    datapath = 'pcd/05.pcd'
    # datapath = 'data/scenes/pcd_new/05.pcd'
    # datapath = 'data/scenes/ply_original/05.ply'
    
    p.loadPointCloud(datapath)
    p.preProcess(voxel_size=0.009)

    # ------------------------------------------
    # Find table
    # ------------------------------------------

    original_pcd = o3d.io.read_point_cloud(datapath)

    # Estimate normals

    original_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=25))
    original_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    # Angle tolerance verification

    angle_tolerance = 0.05
    vx, vy, vz = 1, 0, 0
    norm_b = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    horizontal_idxs = []
    for idx, normal in enumerate(original_pcd.normals):

        nx, ny, nz = normal
        ab = nx*vx + ny*vy + nz*vz
        norm_a = math.sqrt(nx**2 + ny**2 + nz**2)
        angle = math.acos(ab/(norm_a * norm_b)) * 180/math.pi

        if abs(angle - 90) < angle_tolerance:
            horizontal_idxs.append(idx)

    # Get horizontal pointcloud
            
    horizontal_cloud = original_pcd.select_by_index(horizontal_idxs)
    _ = original_pcd.select_by_index(horizontal_idxs, invert=True)      # Non horizontal points

    # Remove unwanted points

    (table_point_cloud, _) = horizontal_cloud.remove_radius_outlier(150, 0.3)

    # Get table plane

    table_plane = PlaneDetection(table_point_cloud)
    table_plane_point_cloud = table_plane.segment()

    # Get table center

    table_center = table_plane_point_cloud.get_center()

    # Positioning coordinate axis in the middle of table

    p.transform(0,0,0,-table_center[0],-table_center[1],-table_center[2])
    p.transform(-120,0,0,0,0,0)
    p.transform(0,0,-120,0,0,0)
    p.transform(0,-7,0,0,0,0)

    # Cropping point cloud

    p.crop(-0.6, -0.5, -0.025, 0.6, 0.5, 0.5)

    # Find plane

    all_objects = p.findPlane()
    
    # ------------------------------------------------------
    # Clustering
    # ------------------------------------------------------

    cluster_idxs = list(all_objects.cluster_dbscan(eps=0.031, min_points=70, print_progress=True))
    obj_idxs = list(set(cluster_idxs))
    obj_idxs.remove(-1)

    number_of_objects = len(obj_idxs)
    print ('Number of objects: ' + str(number_of_objects))

    colormap = cm.Pastel1(list(range(0,number_of_objects)))

    # Deleting existent .pcd in the folder

    for file in glob.glob('Part2_Test/Objects_pcd/*.pcd'):

        os.remove(file)
    print('All files removed')
 
    # Objects on the table

    classes = {"bowl": 0,
           "cap": 1,
           "cereal": 2,
           "coffee": 3,
           "soda": 4}

    objects = []
    for obj_idx in obj_idxs:

        obj_point_idxs = list(locate(cluster_idxs, lambda x: x == obj_idx))

        obj_points = all_objects.select_by_index(obj_point_idxs)

        # entities = []
        # entities.append(obj_points)

        # o3d.visualization.draw_geometries(entities,
        #                             zoom=view['trajectory'][0]['zoom'],
        #                             front=view['trajectory'][0]['front'],
        #                             lookat=view['trajectory'][0]['lookat'],
        #                             up=view['trajectory'][0]['up'])

        # print('\nBowl = 0\nCap = 1\nCereal Box = 2\nCoffee mug = 3\nSoda can = 4\n')

        # # while True:

        # #     gt_object = input('Insert object class: ')   # TODO make a pop-up or something because like this is not good 

        # #     if gt_object == '0' or gt_object == '1' or gt_object == '2' or gt_object == '3' or gt_object == '4':
        # #         break

        # #     else:
        # #         print('Invalid Input! Please try again.')

    

        # o3d.io.write_point_cloud('Part2_Test/Objects_pcd/' + list(classes.keys())[int(obj_idx)] + '_' + str(obj_idx) + '.pcd', obj_points)

        # Create a dictionary to represent the objects

        d = {}
        d['idx'] = str(obj_idx)
        d['points'] = obj_points
        d['color'] = colormap[obj_idx, 0:3]
        d['points'].paint_uniform_color(d['color'])
        d['center'] = d['points'].get_center()

        objects.append(d)

    # Deleting existent .off in the folder

    # for file in glob.glob('Part2_Test/Objects_off/*.off'):

    #     os.remove(file)
    # print('All files removed')


    # ----------------------------------------------------
    # Converting .pcd to .off
    # ----------------------------------------------------

    # Get filenames of all images (including sub-folders)
    # object_files = glob.glob('Part2_Test/Objects_pcd/*.pcd')

    # # Check if dataset data exists
    # if len(object_files) < 1:
    #     raise FileNotFoundError('Dataset files not found')
    
    

    # for pcd_file_path in object_files:

    #     off_file_name = ((os.path.basename(pcd_file_path)).split("."))[0]
    #     off_file_path = "Part2_Test/Objects_off/" + off_file_name + '.off'

    #     convert_pcd_to_off(pcd_file_path, off_file_path)


    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw --------

    entities = []

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox

    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    # Draw objects -----------------------------

    for obj_idx, object in enumerate(objects):
        entities.append(object['points'])
    

    # original_pcd = original_pcd.voxel_down_sample(voxel_size=0.02) 
    # entities = [original_pcd]

    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    # entities.append(frame)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
