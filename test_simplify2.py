import numpy as np
import trimesh
import mcubes
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import open3d as o3d
import os


def add_random_noise(vertices, noise_level=0.01):
    # 添加随机噪声
    noise = np.random.normal(0, noise_level, vertices.shape)
    vertices += noise
    return vertices

def random_vertex_deletion(mesh, deletion_ratio=0.1):
    # 随机删除顶点
    num_vertices = mesh.vertices.shape[0]
    num_to_delete = int(num_vertices * deletion_ratio)
    indices_to_delete = np.random.choice(num_vertices, num_to_delete, replace=False)
    mesh.vertices = np.delete(mesh.vertices, indices_to_delete, axis=0)
    mesh.faces = np.array([f for f in mesh.faces if not any(v in indices_to_delete for v in f)])
    return mesh

def simplify_mesh(mesh, simplify_ratio=0.6):
    # 网格简化
    # print(mesh)
    # print(mesh)
    num_points = int(mesh.vertices.shape[0] * simplify_ratio)
    # print(num_points)
    simplified_mesh = mesh.simplify_quadric_decimation(num_points)
    return simplified_mesh

def deform_mesh(vertices, deformation_strength=0.05):
    # 形状变形
    deformation = np.random.uniform(-deformation_strength, deformation_strength, vertices.shape)
    vertices += deformation
    return vertices

def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    size = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    normalized_vertices = (vertices - center) / size.max() * 2  # 标准化到[-1, 1]
    return normalized_vertices

def normalize_mesh(mesh):
    bounds = mesh.bounds

    max_dimension = np.max(bounds[1] - bounds[0])
    scale_factor = 2.0 / max_dimension
    center = (bounds[1] + bounds[0]) / 2.0
    mesh.apply_translation(-center)
    mesh.apply_scale(scale_factor)
    return mesh

def create_voxel_grid(vertices, grid_size=32, bounds=None):
    if bounds is None:
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    
    # 创建体素网格的边界
    x = np.linspace(bounds[0, 0], bounds[1, 0], grid_size)
    y = np.linspace(bounds[0, 1], bounds[1, 1], grid_size)
    z = np.linspace(bounds[0, 2], bounds[1, 2], grid_size)
    
    # 创建网格
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # 初始化体素网格
    voxel_grid = np.zeros(grid_x.shape, dtype=np.float32)
    
    # 计算每个体素内的点到最近顶点的距离
    for v in vertices:
        distances = np.sqrt((grid_x - v[0])**2 + (grid_y - v[1])**2 + (grid_z - v[2])**2)
        # 使用距离来更新体素值，设置一个阈值
        voxel_grid = np.maximum(voxel_grid, 1.0 - distances / np.max(distances))

    return voxel_grid, (grid_x, grid_y, grid_z)

def process_single_mesh(mesh):
    mesh = simplify_mesh(mesh, simplify_ratio=0.4)
    num_samples = mesh.vertices.shape[0]

    sampled_points = mesh.sample(1000)

    sampled_points = add_random_noise(sampled_points, noise_level=0.01)
    # mesh = random_vertex_deletion(mesh, deletion_ratio=0.05)
    # print(mesh)
    sampled_points = deform_mesh(sampled_points, deformation_strength=0.05)
    # print(mesh)
    # 将处理后的网格添加到列表中
    return sampled_points

def process_mesh(input_file, file_path):

    scene = trimesh.load(input_file)

    # scene.export('test_ori.obj')
    
    # 存储处理后的网格
    # original_meshes = []
    processed_points = []

    # 如果是Scene对象，处理其中的每个网格
    if isinstance(scene, trimesh.Scene):
        combined_mesh = trimesh.util.concatenate(scene.dump())
        for i, (name, mesh) in enumerate(scene.geometry.items()):
            sampled_points = process_single_mesh(mesh)
            processed_points.append(sampled_points)
    else:
        combined_mesh = scene.copy()
        sampled_points = process_single_mesh(scene)
        processed_points.append(sampled_points)
    
    
    # 将处理后的网格重新组合成一个场景
    # new_scene = trimesh.Scene()
    # for mesh in original_meshes:
    #     new_scene.add_geometry(mesh)

    # # 导出处理后的场景
    # new_scene.export(output_file)
    
    all_vertices = np.concatenate([points for points in processed_points])
    all_vertices = normalize_vertices(all_vertices)
    print(len(all_vertices))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_vertices)

    alpha = 0.15

    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    print(len(alpha_shape.vertices))
    # alpha_shape.compute_vertex_normals()
    # o3d.visualization.draw_geometries([alpha_shape])
    # o3d.io.write_triangle_mesh(output_file, alpha_shape)

    vertices_o3d = np.asarray(alpha_shape.vertices)
    faces_o3d = np.asarray(alpha_shape.triangles)
    alpha_mesh = trimesh.Trimesh(vertices=vertices_o3d, faces=faces_o3d)
    alpha_mesh.export(file_path+'_32.obj')
    
    coarse_surface_points = alpha_mesh.sample(500000)
    print(coarse_surface_points.min(), coarse_surface_points.max())
    combined_mesh = normalize_mesh(combined_mesh)
    surface_points = combined_mesh.sample(500000)
    print(surface_points.min(), surface_points.max())
    np.savez(file_path + "_points.npz", surface_points=surface_points, coarse_surface_points=coarse_surface_points)


    # mesh_rc4 = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.1)
    # mesh_rc4.compute_vertex_normals()
    # o3d.io.write_triangle_mesh("test_voxel_shape_mesh.obj", mesh_rc4)
    # point_cloud.estimate_normals(   # 法向量计算
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    # )
    # mesh_rc3, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    # mesh_rc3 = o3d.geometry.TriangleMesh(mesh_rc3)
    # o3d.io.write_triangle_mesh("test_possi_shape_mesh.obj", mesh_rc3)
    
    # 创建点云（可以使用 trimesh 的 PointCloud）
    point_cloud = trimesh.PointCloud(vertices=all_vertices)
    point_cloud.export('./simplified_point_cloud.ply')  # PLY 格式

    # print("vert")
    # voxel_grid, _ = create_voxel_grid(all_vertices, 64)
    # print("Voxel grid min:", voxel_grid.min())
    # print("Voxel grid max:", voxel_grid.max())
    # print("voxel")
    # verts, faces = mcubes.marching_cubes(voxel_grid, 0.5)

    # new_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # new_mesh.export(output_file)


if __name__ == "__main__":
    for mesh_id in os.listdir('/data/ljf/shapenet_test'):
        input_mesh_file = f'/data/ljf/shapenet_test/{mesh_id}/{mesh_id}.gltf'
    # input_mesh_file = '/data/ljf/1006be65e7bc937e9141f9b58470d646.gltf'
        output_mesh_file = f'/data/ljf/shapenet_test/{mesh_id}/{mesh_id}_32.obj' 
        file_path = f'/data/ljf/shapenet_test/{mesh_id}/{mesh_id}' 
        process_mesh(input_mesh_file, file_path)
        print(f"Processed mesh saved to {output_mesh_file}")

        # if mesh_id == "1006be65e7bc937e9141f9b58470d646":
        #     break
