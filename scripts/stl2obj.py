import trimesh

mesh = trimesh.load("input.stl", process=False)
mesh.export("output.obj")
