import glob
import os
import open3d as o3d


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


# Get filenames of all images (including sub-folders)
dataset_filenames = glob.glob('data/objects_pcd/**/*.pcd', recursive=True)

# Check if dataset data exists
if len(dataset_filenames) < 1:
    raise FileNotFoundError('Dataset files not found')

i = 0
for pcd_file_path in dataset_filenames:

    off_file_name = ((os.path.basename(pcd_file_path)).split("."))[0]
    off_file_path = "data/objects_off/" + off_file_name + '.off'

    if os.path.exists(off_file_path):
        i += 1
        print(str(i) + "/" + str(len(dataset_filenames)))
        print(str((i / len(dataset_filenames)) * 100) + "%")

    else:
        convert_pcd_to_off(pcd_file_path, off_file_path)

        i += 1
        print(str(i) + "/" + str(len(dataset_filenames)))
        print(str((i / len(dataset_filenames)) * 100) + "%")

