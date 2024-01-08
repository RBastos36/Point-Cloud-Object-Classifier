#!/usr/bin/env python3

# Necessário para as scenes tal como a 5, onde o eixo dos z's passa a mesa.


import csv
import os
import pickle
import random
import glob
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from more_itertools import locate


view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.648259162902832, 0.17168067395687103, 1.7319790124893188 ],
			"boundingbox_min" : [ 0.76354414224624634, -0.17333842813968658, 0.89897191524505615 ],
			"field_of_view" : 60.0,
			"front" : [ 0.76868665380378454, -0.62384756489317472, 0.14119151547029002 ],
			"lookat" : [ 1.2059016525745392, -0.00082887709140777588, 1.3154754638671875 ],
			"up" : [ -0.54858848092307122, -0.75652228416597067, -0.35598414593745553 ],
			"zoom" : 0.90000000000000013
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.16, ransac_n=3, num_iterations=1250):    #0.09 without cereal box

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Scene 05, 06 (meio mal captado), 07 (meio mal captadp), 

    point_cloud_original = o3d.io.read_point_cloud('data/scenes/pcd/13.pcd')
    if len(point_cloud_original.points) < 1:
        exit('File not found')
    print('loaded a point cloud with ' + str(len(point_cloud_original.points)))

    point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.01) 
    print('After downsampling point cloud has ' + str(len(point_cloud_downsampled.points)) + ' points')


    number_of_planes = 2
    minimum_number_points = 50
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    point_cloud = deepcopy(point_cloud_downsampled) 
    planes = []
    while True: # run consecutive plane detections

        plane = PlaneDetection(point_cloud) # create a new plane instance
        point_cloud = plane.segment() # new point cloud are the outliers of this plane detection
        print(plane)

        # colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        # plane.colorizeInliers(r=color[0], g=color[1], b=color[2])

        planes.append(plane)

        if len(planes) >= number_of_planes: # stop detection planes
            print('Detected planes >= ' + str(number_of_planes))
            break
        elif len(point_cloud.points) < minimum_number_points:
            print('Number of remaining points < ' + str(minimum_number_points))
            break
            

    # Cluster extraction from both planes point clouds
    clusters = []
    for plane in planes:
        p = deepcopy(plane)
        cluster_idxs = list(p.inlier_cloud.cluster_dbscan(eps=0.04, min_points=10, print_progress=True))
        #print(cluster_idxs)

        possible_values = list(set(cluster_idxs))
        if -1 in possible_values:
            possible_values.remove(-1)

        for value in possible_values:

            point_idxs = list(locate(cluster_idxs, lambda x: x == value))
            cluster_cloud = p.inlier_cloud.select_by_index(point_idxs)
            clusters.append(deepcopy(cluster_cloud))


    #colormap = cm.Pastel1(list(range(0,len(clusters))))
    # colormap = cm.hsv(list(range(0,len(clusters))))
    # for cluster_idx, cluster in enumerate(clusters):
    #     #color = colormap[cluster_idx, 0:3]
    #     cluster.paint_uniform_color(color) # paints the table green

    # Table plane detector
    table_plane = None
    table_plane_mean_y = 1000
    for plane_idx, plane in enumerate(planes):
        center = plane.inlier_cloud.get_center()
        print('Cloud ' + str(plane_idx) + ' has center ' + str(center))
        mean_y = center[1]
        if mean_y < table_plane_mean_y:
            table_plane = plane
            table_plane_mean_y = mean_y

    #table_plane.colorizeInliers(r=1, g=0, b=0)


    # Cluster extraction
    cluster_idxs = list(table_plane.inlier_cloud.cluster_dbscan(eps=0.15, min_points=25, print_progress=True))

    # print(cluster_idxs)
    # print(type(cluster_idxs))

    possible_values = list(set(cluster_idxs))
    if -1 in possible_values:
        possible_values.remove(-1)
    print(possible_values)

    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))
    table_cloud = table_plane.inlier_cloud.select_by_index(largest_idxs)

    #table_cloud.paint_uniform_color([0,1,0]) # paints the table green

    # Auto define table reference frame

    # origin is the center of the table cloud
    center = table_cloud.get_center()
    tx,ty,tz = center[0], center[1], center[2]
    # nx, ny, nz 

    # ---------------------------------------------------

    # plane_model, inlier_idxs = table_cloud.segment_plane(distance_threshold=0.2, 
    #                                                 ransac_n=3,
    #                                                 num_iterations=100)
    

    # table_pcd = point_cloud.select_by_index(inlier_idxs)

    # objects_pcd = point_cloud.select_by_index(inlier_idxs, invert=True)

    # ----------------------------------------------------

    table_plane = PlaneDetection(table_cloud) # create a new plane instance
    pcd_objects = table_plane.segment(distance_threshold = 0.0048, ransac_n = 3 , num_iterations = 100) #has objects and table boundaries

    # print(type(table_plane))
    # pcd_objects = table_cloud.inlier_cloud


    # --------------------------------------
    # Clustering
    # --------------------------------------

    #labels = pcd_objects.cluster_dbscan(eps=0.065, min_points=60, print_progress=True)    # valores com chapeu todo, mas 2 juntos
    #labels = pcd_objects.cluster_dbscan(eps=0.04, min_points=30, print_progress=True)      # valores com tudo separado, mas chapeu a meio 

    labels = pcd_objects.cluster_dbscan(eps = 0.034, min_points = 80, print_progress = True)      # valores para voxel size = 0.015 - chapeu cortado a meio

    #print("Max label:", max(labels))

    group_idxs = list(set(labels))
    group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    colormap = cm.Pastel1(range(0, num_groups))

    print("Max label:", str(num_groups))


    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        color = colormap[group_idx, 0:3]
        pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)


    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw --------
    entities = []
    #entities = [x.inlier_cloud for x in planes]
    #entities = [cluster for cluster in clusters]
    # entities = [point_cloud_original]
    # entities.append(table_cloud)
    # entities.append(pcd_objects)
    # entities.append()
    entities.extend(pcd_separate_objects)



    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    # entities.append(frame)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()