"""Render a prepared sequence (mttrain.render prepare) with Blender, one PNG per frame.

    blender -b -t 8 --python tools/render/render_blender.py -- <seq.npz> <frames dir>

Scene: the skinned MHR mesh (vertices per frame from the npz), a plain skin material, a wall behind
the performer, soft key light near the camera plus ambient light; the camera reproduces the sampled
desk camera (pose and horizontal field of view) at 640x480, EEVEE.
Coordinates: MHR world (metres, y up) -> Blender (z up): (x, y, z) -> (x, -z, y).
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
seq, out = Path(argv[0]), Path(argv[1])
out.mkdir(parents=True, exist_ok=True)
d = np.load(seq)
V, F = d["verts"], d["faces"]
R, t, fx = d["R"], d["t"], float(d["fx"])
skin, wall = d["skin"], d["wall"]


def to_bl(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# performer
mesh = bpy.data.meshes.new("mhr")
mesh.from_pydata(to_bl(V[0]).tolist(), [], F.tolist())
mesh.shade_smooth()
body = bpy.data.objects.new("mhr", mesh)
scene.collection.objects.link(body)
mat = bpy.data.materials.new("skin")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (*skin, 1.0)
bsdf.inputs["Roughness"].default_value = 0.55
mesh.materials.append(mat)

# camera: rows of R are image-right, image-down, forward in MHR world
right, down, fwd = (to_bl(R[k]) for k in range(3))
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = fx * 36.0
cam_data.clip_start, cam_data.clip_end = 0.05, 20.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
rot = Matrix([list(right), list(-down), list(-fwd)]).transposed()  # columns: camera x, y, z in world
cam.matrix_world = Matrix.Translation(Vector(to_bl(t))) @ rot.to_4x4()
scene.camera = cam

# wall behind the performer, facing the camera
centre = to_bl(V[0].mean(0))
dist = float(np.dot(centre - to_bl(t), fwd)) + 1.2
bpy.ops.mesh.primitive_plane_add(size=12.0)
wall_obj = bpy.context.active_object
wall_obj.matrix_world = Matrix.Translation(Vector(to_bl(t) + dist * fwd)) @ rot.to_4x4()
wmat = bpy.data.materials.new("wall")
wmat.use_nodes = True
wmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*wall, 1.0)
wall_obj.data.materials.append(wmat)

# lights: key near the camera (slightly above and to the side), ambient world
key = bpy.data.lights.new("key", "AREA")
key.energy, key.size = 30.0, 1.5
key_obj = bpy.data.objects.new("key", key)
key_obj.location = Vector(to_bl(t) + 0.4 * right - 0.5 * down)
key_obj.rotation_euler = cam.rotation_euler
scene.collection.objects.link(key_obj)
world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
scene.world = world

r = scene.render
r.engine = "BLENDER_EEVEE"
r.resolution_x, r.resolution_y, r.resolution_percentage = 640, 480, 100
r.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

co = np.empty(len(V[0]) * 3, dtype=np.float32)
for i in range(len(V)):
    co[:] = to_bl(V[i]).astype(np.float32).ravel()
    mesh.vertices.foreach_set("co", co)
    mesh.update()
    r.filepath = str(out / f"{i:05d}.png")
    bpy.ops.render.render(write_still=True)
