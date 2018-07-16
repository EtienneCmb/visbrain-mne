"""Utility mesh functions."""
from os import path as op

import numpy as np
import nibabel as nib

from ..io import _get_subjects_dir


class Surface(object):
    """Container for surface object.

    Attributes
    ----------
    subject_id : string
        Name of subject
    hemi : {'lh', 'rh'}
        Which hemisphere to load
    surf : string
        Name of the surface to load (eg. inflated, orig ...)
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    offset : float | None
        If float, align inside edge of each hemisphere to center + offset.
        If None, do not change coordinates (default).
    units : str
        Can be 'm' or 'mm' (default).
    """

    def __init__(self, subject_id, hemi, surf, subjects_dir=None,
                 offset=None, units='mm'):
        """Surface.

        Parameters
        ----------
        subject_id : string
            Name of subject
        hemi : {'lh', 'rh'}
            Which hemisphere to load
        surf : string
            Name of the surface to load (eg. inflated, orig ...)
        offset : float | None
            If 0.0, the surface will be offset such that the medial
            wall is aligned with the origin. If None, no offset will
            be applied. If != 0.0, an additional offset will be used.
        """
        if hemi not in ['lh', 'rh']:
            raise ValueError('hemi must be "lh" or "rh')
        self.subject_id = subject_id
        self.hemi = hemi
        self.surf = surf
        self.offset = offset
        self.coords = None
        self.faces = None
        self.nn = None
        self.units = _check_units(units)

        subjects_dir = _get_subjects_dir(subjects_dir)
        self.data_path = op.join(subjects_dir, subject_id)

    def __len__(self):
        """Get the number of vertices."""
        return self.coords.shape[0]

    def load_geometry(self):
        """Load vertices, faces and nrmals."""
        surf_path = op.join(self.data_path, "surf",
                            "%s.%s" % (self.hemi, self.surf))
        coords, faces = nib.freesurfer.read_geometry(surf_path)
        if self.units == 'm':
            coords /= 1000.
        if self.offset is not None:
            if self.hemi == 'lh':
                coords[:, 0] -= (np.max(coords[:, 0]) + self.offset)
            else:
                coords[:, 0] -= (np.min(coords[:, 0]) + self.offset)
        nn = _compute_normals(coords, faces)

        if self.coords is None:
            self.coords = coords.astype(np.float32)
            self.faces = faces.astype(np.uint32)
            self.nn = nn.astype(np.float32)
        else:
            self.coords[:] = coords.astype(np.float32)
            self.faces[:] = faces.astype(np.uint32)
            self.nn[:] = nn.astype(np.float32)

    @property
    def x(self):
        """Get x coordinate of vertices."""
        return self.coords[:, 0]

    @property
    def y(self):
        """Get y coordinate of vertices."""
        return self.coords[:, 1]

    @property
    def z(self):
        """Get z coordinate of vertices."""
        return self.coords[:, 2]

    def load_curvature(self):
        """Load in curvature values from the ?h.curv file."""
        curv_path = op.join(self.data_path, "surf", "%s.curv" % self.hemi)
        self.curv = nib.freesurfer.read_morph_data(curv_path)
        self.bin_curv = self.curv > 0

    def load_label(self, name):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.
        """
        label = nib.freesurfer.read_label(op.join(self.data_path, 'label',
                                          '%s.%s.label' % (self.hemi, name)))
        label_array = np.zeros(len(self.x), np.int)
        label_array[label] = 1
        try:
            self.labels[name] = label_array
        except AttributeError:
            self.labels = {name: label_array}

    def apply_xfm(self, mtx):
        """Apply an affine transformation matrix to the x,y,z vectors."""
        self.coords = np.dot(np.c_[self.coords, np.ones(len(self.coords))],
                             mtx.T)[:, :3]


###############################################################################
# USEFUL FUNCTIONS

def _check_units(units):
    if units not in ('m', 'mm'):
        raise ValueError('Units must be "m" or "mm", got %r' % (units,))
    return units


def _fast_cross_3d(x, y):
    """Compute cross product between list of 3D vectors.

    Much faster than np.cross() when the number of cross products
    becomes large (>500). This is because np.cross() methods become
    less memory efficient at this stage.

    Parameters
    ----------
    x : array
        Input array 1.
    y : array
        Input array 2.

    Returns
    -------
    z : array
        Cross product of x and y.

    Notes
    -----
    x and y must both be 2D row vectors. One must have length 1, or both
    lengths must match.
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == 3
    assert y.shape[1] == 3
    assert (x.shape[0] == 1 or y.shape[0] == 1) or x.shape[0] == y.shape[0]
    if max([x.shape[0], y.shape[0]]) >= 500:
        return np.c_[x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
                     x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
                     x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]]
    else:
        return np.cross(x, y)


def _compute_normals(rr, tris):
    """Efficiently compute vertex normals for triangulated surface.

    Parameters
    ----------
    rr : array_like
        Vertices
    tris : array_like
        Faces
    """
    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = _fast_cross_3d((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    zidx = np.where(size == 0)[0]
    size[zidx] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    npts = len(rr)

    # the following code replaces this, but is faster (vectorized):
    #
    # for p, verts in enumerate(tris):
    #     nn[verts, :] += tri_nn[p, :]
    #
    nn = np.zeros((npts, 3))
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(verts, tri_nn[:, idx], minlength=npts)
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]
    return nn
