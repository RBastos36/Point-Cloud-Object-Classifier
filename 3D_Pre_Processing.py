#!/usr/bin/env python3

import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

# Scene 01 , 02, 03, 04

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.8232353472953937, 2.3837740017242588, 0.70045684766887073 ],
			"boundingbox_min" : [ -2.4489022765124426, -2.3000045737073003, -1.0783387351096425 ],
			"field_of_view" : 60.0,
			"front" : [ -0.72929795678377729, -0.49373062485540781, 0.47366080723540599 ],
			"lookat" : [ 0.30511249801364193, -0.046654415241437798, -0.43627551452718877 ],
			"up" : [ 0.40372115223153893, 0.24838365926685493, 0.88051961309788296 ],
			"zoom" : 0.44120000000000009
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():


    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    pcd_original = o3d.io.read_point_cloud('datasets/01.pcd')

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.015)
    # pcd_downsampled.paint_uniform_color([1,0,0])
    #pcd_downsampled = pcd_original

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = pcd_downsampled.get_rotation_matrix_from_xyz((113*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R
    # T[0:3, 0] = [1, 0, 0]  # add n vector
    # T[0:3, 1] = [0, 1, 0]  # add s vector
    # T[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T1[0:3, 3] = [0, 0, 0]
    print('T1=\n' + str(T1))

    # Create transformation T2 only with translation
    T2 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation
    T2[0:3, 0] = [1, 0, 0]  # add n vector
    T2[0:3, 1] = [0, 1, 0]  # add s vector
    T2[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.38]
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))

    #Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T)

    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    #Crop the points in the table

    # Create a vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = sy = 0.6
    sz_top = 0.18
    sz_bottom = -0.1
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    #print('np_vertices =\n' + str(np_vertices))

    vertices = o3d.utility.Vector3dVector(np_vertices)

    # Create a bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    print(bbox)

    # Crop the original point cloud using the bounding box
    pcd_cropped = pcd_downsampled.crop(bbox)


    # --------------------------------------
    # Plane segmentation
    # --------------------------------------
    plane_model, inlier_idxs = pcd_cropped.segment_plane(distance_threshold=0.005,
                                                         ransac_n=3, num_iterations=1000)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idxs, invert=False)
    pcd_table.paint_uniform_color([0, 0, 1])

    pcd_objects = pcd_cropped.select_by_index(inlier_idxs, invert=True)

    # --------------------------------------
    # Clustering
    # --------------------------------------

    #labels = pcd_objects.cluster_dbscan(eps=0.065, min_points=60, print_progress=True)    # valores com chapeu todo, mas 2 juntos
    #labels = pcd_objects.cluster_dbscan(eps=0.04, min_points=30, print_progress=True)      # valores com tudo separado, mas chapeu a meio 

    labels = pcd_objects.cluster_dbscan(eps=0.0435, min_points=65, print_progress=True)      # valores para voxel size = 0.015 - chapeu cortado a meio



    #print("Max label:", max(labels))

    group_idxs = list(set(labels))
    group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    colormap = cm.Pastel1(range(0, num_groups))

    print("Max label:", str(num_groups))


    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        color = colormap[group_idx, 0:3]
        pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)
        # pcd_separate_object.paint_uniform_color([0.5, 0.5, 0.5])
        # pcd_separate_objects.append(pcd_separate_object)


    # --------------------------------------
    # Visualization 
    # --------------------------------------
    pcd_downsampled.paint_uniform_color([0.4, 0.3, 0.3])
    pcd_cropped.paint_uniform_color([0.9, 0.0, 0.0])

    #pcds_to_draw = [pcd_downsampled, pcd_cropped, pcd_table]
    pcds_to_draw = []
    #pcds_to_draw = [pcd_downsampled]
    #pcds_to_draw = [pcd_cropped, pcd_table]
    pcds_to_draw.extend(pcd_separate_objects)
    #pcds_to_draw = [pcd_separate_objects]


    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = []
    entities.append(frame_world)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.44120000000000009,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------




if __name__ == '__main__':
    main()




