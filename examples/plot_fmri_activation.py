"""
Display fMRI Activation
=======================

The most straightforward way to plot activations is when you already have a map
of them defined on the Freesurfer surface. This map can be stored in any file
format that Nibabel can understand.
"""
import os

from vismne import Brain
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

"""
Bring up the visualization window.
"""
views = ['lateral', 'ventral', 'dorsal']
brain = Brain("fsaverage", "split", "inflated", subjects_dir=subjects_dir, views=views)

"""
Get a path to the overlay file.
"""
overlay_file = "/home/etienne/Toolbox/PySurfer/examples/example_data/lh.sig.nii.gz"

"""
Display the overlay on the surface using the defaults to control thresholding
and colorbar saturation.  These can be set through your config file.
"""
brain.add_overlay(overlay_file, hemi='lh')
brain.show()
0/0

"""
You can then turn the overlay off.
"""
brain.overlays["sig"].remove()

"""
Now add the overlay again, but this time with set threshold and showing only
the positive activations.
"""
brain.add_overlay(overlay_file, min=5, max=20, sign="pos")
