"""
===============================
Basic mesh plot with an overlay
===============================
This script plot a mesh and add an overlay to it.
"""
# Authors: Etienne Combrisson <e.combrisson@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from vispy import scene, app

from mne.utils import _fetch_file

from vismne.visuals import BrainMesh

inflated_url = "https://www.dropbox.com/s/nl2hh0thoy7xbnd/inflated.npz?dl=1"
_fetch_file(inflated_url, 'inflated.npz')
file = './inflated.npz'

sulcus_url = "https://www.dropbox.com/s/jfihlb7pna7ws2i/sulcus.npy?dl=1"
_fetch_file(sulcus_url, 'sulcus.npy')

mat = np.load(file)
vert, faces, norms = mat['vertices'], mat['faces'], mat['normals']

# SceneCanvas :
cam = scene.cameras.TurntableCamera()
sc = scene.SceneCanvas(bgcolor='white', show=True)
wc = sc.central_widget.add_view(camera=cam)

# Mesh creation :
sul = np.load('./sulcus.npy')
mesh = BrainMesh(vertices=vert, faces=faces, normals=norms, parent=wc.scene,
                 sulcus=sul, camera=cam)  # noqa

# Camera update :
dico = mesh._opt_cam_state
dico['scale_factor'] = dico['scale_factor'][-1]
cam.set_state(mesh._opt_cam_state)
distance = dico['scale_factor']
cam.distance = distance

# Add the overlay :
right = mesh._vertices[:, 0] >= 0
data_1 = mesh._vertices[right, 0]
vert = np.arange(len(mesh))[right]
mesh.add_overlay(data_1, vert, cmap='Reds', vmin=10, under='green')

app.run()
