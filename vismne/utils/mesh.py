"""Utility mesh functions."""
from os import path as op

import logging
import numpy as np
import nibabel as nib

from ..io import _get_subjects_dir

logger = logging.getLogger('visbrain-mne')


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


def smoothing_matrix(vertices, adj_mat, smoothing_steps=20, verbose=None):
    """Create a smoothing matrix which can be used to interpolate data defined
       for a subset of vertices onto mesh with an adjancency matrix given by
       adj_mat.

       If smoothing_steps is None, as many smoothing steps are applied until
       the whole mesh is filled with with non-zeros. Only use this option if
       the vertices correspond to a subsampled version of the mesh.

    Parameters
    ----------
    vertices : 1d array
        vertex indices
    adj_mat : sparse matrix
        N x N adjacency matrix of the full mesh
    smoothing_steps : int or None
        number of smoothing steps (Default: 20)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).

    Returns
    -------
    smooth_mat : sparse matrix
        smoothing matrix with size N x len(vertices)
    """
    from scipy import sparse

    logger.info("Updating smoothing matrix, be patient..")

    e = adj_mat.copy()
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    e = e + sparse.eye(n_vertices, n_vertices)
    idx_use = vertices
    smooth_mat = 1.0
    n_iter = smoothing_steps if smoothing_steps is not None else 1000
    for k in range(n_iter):
        e_use = e[:, idx_use]

        data1 = e_use * np.ones(len(idx_use))
        idx_use = np.where(data1)[0]
        scale_mat = sparse.dia_matrix((1 / data1[idx_use], 0),
                                      shape=(len(idx_use), len(idx_use)))

        smooth_mat = scale_mat * e_use[idx_use, :] * smooth_mat

        logger.info("Smoothing matrix creation, step %d" % (k + 1))
        if smoothing_steps is None and len(idx_use) >= n_vertices:
            break

    # Make sure the smoothing matrix has the right number of rows
    # and is in COO format
    smooth_mat = smooth_mat.tocoo()
    smooth_mat = sparse.coo_matrix((smooth_mat.data,
                                    (idx_use[smooth_mat.row],
                                     smooth_mat.col)),
                                   shape=(n_vertices,
                                          len(vertices)))

    return smooth_mat


def mesh_edges(faces):
    """Returns sparse matrix with edges as an adjacency matrix

    Parameters
    ----------
    faces : array of shape [n_triangles x 3]
        The mesh faces

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    from scipy import sparse
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                              shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                      shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                      shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges
