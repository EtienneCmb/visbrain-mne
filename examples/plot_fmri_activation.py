"""
Display fMRI Activation
=======================

The most straightforward way to plot activations is when you already have a map
of them defined on the Freesurfer surface. This map can be stored in any file
format that Nibabel can understand.
"""
# Author: Etienne Combrisson <e.combrisson@gmail.com>
#
# License: BSD (3-clause)

import os

from vismne import Brain
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

# Bring up the visualization window.
brain = Brain("fsaverage", "lh", "inflated", subjects_dir=subjects_dir)

# Get a path to the overlay file.
overlay_file = os.path.join(*(".", "example_data", "lh.sig.nii.gz"))

# Display the overlay on the surface using the defaults to control thresholding
# and colorbar saturation.  These can be set through your config file. Set
# threshold and showing only the positive activations.
brain.add_overlay(overlay_file, min=5., max=20., sign="pos",
                  colorbar=True, colorbar_title='Positive activation')

# Display the brain
brain.show()
