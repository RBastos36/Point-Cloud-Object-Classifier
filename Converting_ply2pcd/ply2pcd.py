

import open3d as o3d
import os

def convert_ply_to_pcd(input_ply, output_pcd):
    # Loading .ply file
    ply_cloud = o3d.io.read_point_cloud(input_ply)

    # Saving .pcd file
    o3d.io.write_point_cloud(output_pcd, ply_cloud)

if __name__ == "__main__":

    input_directory = 'data/scenes/ply_original/'
    output_directory = 'data/scenes/pcd_new/'

    # Garantir que o diretório de saída existe
    os.makedirs(output_directory, exist_ok=True)

    # Iterar sobre os ficheiros .ply no diretório de entrada
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".ply"):
            input_ply = os.path.join(input_directory, file_name)
            output_pcd = os.path.join(output_directory, file_name.replace(".ply", ".pcd"))
            
            convert_ply_to_pcd(input_ply, output_pcd)

    print("Conversão concluída.")

