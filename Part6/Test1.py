
import open3d as o3d
from matplotlib import pyplot as plt

# Link úteis -----------------------

# https://www.open3d.org/docs/release/
# https://www.open3d.org/docs/release/python_api/open3d.camera.html
# https://medium.com/@ramazanilkera2/rgbd-to-3d-point-cloud-223b0a6f46db
# https://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
# https://github.com/PHANTOM0122/3D_Object_Reconstruction

# ----------------------------------
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


# Specify the correct file paths
color_raw = o3d.io.read_image("Part6/RGB-D images/cap_1_1_1.png")
depth_raw = o3d.io.read_image("Part6/RGB-D images/cap_1_1_1_depth.png")

# Create RGBDImage
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

# Visualize color and depth images
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth Image')
plt.imshow(rgbd_image.depth)
plt.show()

# Create PointCloud from RGBDImage
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flip the point cloud to handle orientation issues
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Visualize the point cloud
# o3d.visualization.draw_geometries([pcd], zoom=0.5)
entities = [pcd]

o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])




