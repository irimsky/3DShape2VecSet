import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("/data/ljf/10155655850468db78d106ce0a280f87.gltf")
# reduction = 0.01
# preserve_q = 0.5

# o3d.io.write_triangle_mesh("./test_ori.obj", mesh)
# num_vertices_original = len(mesh.vertices)

# print(len(mesh.triangles))

# target_number_of_triangles = int(num_vertices_original * reduction)

# # Simplify the mesh
# mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
# print(len(mesh.triangles))
# 转换为非 CUDA 的 TriangleMesh 对象
# 打印原始网格信息
print("Original mesh has {} points and {} triangles.".format(len(mesh.vertices), len(mesh.triangles)))
mesh.compute_vertex_normals()

# 获取顶点坐标和法线
vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)

# 随机膨胀
np.random.seed(42)  # 设置随机种子以保证可重复性
noise = np.random.normal(0, 0.01, size=vertices.shape)  # 生成随机噪声
expanded_vertices = vertices + normals * noise  # 沿法线方向膨胀

# 更新网格顶点
mesh.vertices = o3d.utility.Vector3dVector(expanded_vertices)

# Save the simplified mesh
o3d.io.write_triangle_mesh("./test_expand.obj", mesh)

# 设置体素大小
voxel_size = 0.01  # 你可以根据需要调整这个值

# 使用顶点聚类进行简化
mesh_simplified = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.cuda.pybind.geometry.SimplificationContraction.Average
)

# 打印简化后的网格信息
print("Simplified mesh has {} points and {} triangles.".format(len(mesh_simplified.vertices), len(mesh_simplified.triangles)))

# Save the simplified mesh
o3d.io.write_triangle_mesh("./test_simp.obj", mesh_simplified)