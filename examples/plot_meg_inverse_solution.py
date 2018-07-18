"""
Display MEG inverse solution
============================

This example shows how to use visbrain-mne to plot
inverse solutions
"""
# Author: Etienne Combrisson <e.combrisson@gmail.com>
#
# License: BSD (3-clause)

import os
import numpy as np

import mne
from mne.datasets import sample

from vismne import Brain

print(__doc__)

# define subject, surface and hemisphere(s) to plot
data_path = sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')
subject_id, surf = 'fsaverage', 'inflated'
hemi = 'lh'

# create Brain object for visualization
brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)


# label for time annotation in milliseconds

def time_label(t):
    """Time label format."""
    return 'time=%0.2f ms' % (t * 1e3)


# read MNE dSPM inverse solution
for hemi in ['lh']:  # , 'rh']:
    stc_fname = os.path.join(*(".", "example_data", 'meg_source_estimate-' +
                               hemi + '.stc'))
    stc = mne.read_source_estimate(stc_fname)

    # data and vertices for which the data is defined
    data = stc.data
    vertices = stc.vertices[0]
    data = data[vertices]

    # time points (in seconds)
    time = np.linspace(stc.tmin, stc.tmin + data.shape[1] * stc.tstep,
                       data.shape[1], endpoint=False)

    # colormap to use
    colormap = 'hot'

    # add data and set the initial time displayed to 100 ms
    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=5, time=time, time_label=time_label,
                   hemi=hemi, initial_time=0.10)

# scale colormap
brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)
brain.show()
