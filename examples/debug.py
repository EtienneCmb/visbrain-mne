import numpy as np
from vispy import scene, app

from vismne.visuals import BrainMesh

file = '/home/etienne/Toolbox/visbrain/visbrain/data/templates/inflated.npz'  # noqa
mat = np.load(file)
vert, faces, norms = mat['vertices'], mat['faces'], mat['normals']  # noqa
# SceneCanvas :
cam = scene.cameras.TurntableCamera()
sc = scene.SceneCanvas(bgcolor='white', show=True)
wc = sc.central_widget.add_view(camera=cam)
# Mesh creation :
sul = np.load('/home/etienne/Toolbox/visbrain/visbrain/data/templates/sulcus.npy')  # noqa
mesh = BrainMesh(vertices=vert, faces=faces, normals=norms, parent=wc.scene, sulcus=sul, camera=cam)  # noqa
# mesh.translucent = True
# mesh.set_camera(cam)
# mesh.hemisphere = 'left'
# Camera update :
dico = mesh._opt_cam_state
dico['scale_factor'] = dico['scale_factor'][-1]
cam.set_state(mesh._opt_cam_state)
distance = dico['scale_factor']
cam.distance = distance

# mesh.clean()

right = mesh._vertices[:, 0] >= 0
data_1 = mesh._vertices[right, 0]
vert = np.arange(len(mesh))[right]
mesh.add_overlay(data_1, vert, cmap='Reds', vmin=10, under='green')

# left = mesh._vertices[:, 1] < 0
# data_1 = mesh._vertices[left, 1]
# vert = np.arange(len(mesh))[left]
# mesh.add_overlay(data_1, vert, cmap='Blues')
# mesh.update_colormap(cmap='autumn', to_overlay=1)

# left = mesh._vertices[:, 0] < 0
# data_1 = mesh._vertices[left, 0]
# vert = np.arange(len(mesh))[left]
# mesh.add_overlay(data_1, vert, cmap='Greens')

# mesh.add_overlay(mesh._vertices[:, 0], cmap='gray')

app.run()