"""Need doc."""
import os

import numpy as np

from vispy import scene

from .io import _get_subjects_dir, read_scalar_data
from .utils import Surface, _check_units, string_types
from .visuals import BrainMesh


lh_viewdict = {'lateral': {'v': (-90., 0.), 'r': 90., 'xyz': [1, 2]},
               'medial': {'v': (90., 0.), 'r': -90., 'xyz': [1, 2]},
               'rostral': {'v': (180., 0.), 'r': -180., 'xyz': [0, 2]},
               'caudal': {'v': (0., 0.), 'r': 0., 'xyz': [0, 2]},
               'dorsal': {'v': (0., 90.), 'r': 90., 'xyz': [0, 1]},
               'ventral': {'v': (0., -90.), 'r': 90., 'xyz': [0, 1]},
               'frontal': {'v': (120., 80.), 'r': 106.739, 'xyz': [0, 1]},
               'parietal': {'v': (-120., 60.), 'r': 49.106, 'xyz': [0, 1]}}
rh_viewdict = {'lateral': {'v': (90., 0.), 'r': -90., 'xyz': [1, 2]},
               'medial': {'v': (-90., 0.), 'r': 90., 'xyz': [1, 2]},
               'rostral': {'v': (180., 0.), 'r': 180., 'xyz': [0, 2]},
               'caudal': {'v': (0., 0.), 'r': 0., 'xyz': [0, 2]},
               'dorsal': {'v': (0., 90.), 'r': 90., 'xyz': [0, 1]},
               'ventral': {'v': (0., -90.), 'r': 90., 'xyz': [0, 1]},
               'frontal': {'v': (60., 80.), 'r': -106.739, 'xyz': [0, 1]},
               'parietal': {'v': (-60., 60.), 'r': -49.106, 'xyz': [0, 1]}}
viewdicts = dict(lh=lh_viewdict, rh=rh_viewdict)


class Brain(object):
    """Class for visualizing a brain using multiple views in vispy.

    Parameters
    ----------
    subject_id : str
        subject name in Freesurfer subjects dir
    hemi : str
        hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        freesurfer surface mesh name (ie 'white', 'inflated', etc.)
    title : str
        title for the window
    views : list | str
        views to use (lateral, medial, rostral, caudal, dorsal, ventral,
        frontal parietal)
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True)
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    units : str
        Can be 'm' or 'mm' (default).
    size : float or pair of floats
        the size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : matplotlib color
        Color of the background.
    foreground : matplotlib color
        Color of the foreground (will be used for colorbars and text).
        None (default) will use black or white depending on the value
        of ``background``.
    """

    def __init__(self, subject_id, hemi, surf, title=None, views=['lateral'],
                 offset=True, subjects_dir=None, units='mm', size=800,
                 foreground="white", background="black",):  # noqa: D102
        self._units = _check_units(units)
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        if hemi not in col_dict.keys():
            raise ValueError('hemi must be one of [%s], not %s'
                             % (', '.join(col_dict.keys()), hemi))
        # Get the subjects directory from parameter or env. var
        subjects_dir = _get_subjects_dir(subjects_dir=subjects_dir)

        self._hemi = hemi
        if title is None:
            title = subject_id
        self.subject_id = subject_id

        if not isinstance(views, list):
            views = [views]

        # _______________________ GEOMETRY _______________________
        offset = None if (not offset or hemi != 'both') else 0.0
        self.geo = dict()
        if hemi in ['split', 'both']:
            geo_hemis = ['lh', 'rh']
        elif hemi == 'lh':
            geo_hemis = ['lh']
        elif hemi == 'rh':
            geo_hemis = ['rh']
        else:
            raise ValueError('bad hemi value')
        # geo_kwargs, geo_reverse, geo_curv = self._get_geo_params(cortex, alpha)
        for h in geo_hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and (maybe) curvature
            geo.load_geometry()
            if True:  #geo_curv:
                geo.load_curvature()
            self.geo[h] = geo

        # _______________________ PARENT _______________________
        # Deal with making figures
        self._set_window_properties(size, background, foreground)
        del background, foreground
        # Camera, canvas and grid : :
        self._sc = scene.SceneCanvas(bgcolor=self._bg_color, show=True,
                                     size=self._scene_size, title=title)
        self._grid = self._sc.central_widget.add_grid(margin=10)
        self._parents = dict()

        # _______________________ BRAINS _______________________
        # fill figures with brains
        brains = []
        for ri, view in enumerate(views):
            for hi, h in enumerate(['lh', 'rh']):
                if not (hemi in ['lh', 'rh'] and h != hemi):
                    ci = hi if hemi == 'split' else 0
                    # Switch if the subplot exist :
                    sub_exist = (ri, ci) in self._parents.keys()
                    if not sub_exist:
                        camera = scene.cameras.TurntableCamera()
                        parent = self._grid.add_view(row=ri, col=ci,
                                                     camera=camera)
                        kwrci = dict(parent=parent, camera=camera)
                        self._parents[(ri, ci)] = kwrci
                        parent.camera = camera
                    else:
                        parent = self._parents[(ri, ci)]['parent']
                        camera = self._parents[(ri, ci)]['camera']
                    # Mesh creation :
                    geo = self.geo[h]
                    brain = BrainMesh(vertices=geo.coords, faces=geo.faces,
                                      normals=geo.nn, sulcus=geo.bin_curv,
                                      parent=parent.scene, camera=camera)
                    brains += [dict(row=ri, col=ci, brain=brain, hemi=h)]
                    # If 'both', center must be the mean of lh and rh :
                    cam_state = dict(center=brain._cam_center)
                    if sub_exist:
                        center = (camera.center + brain._cam_center) / 2.
                        cam_state['center'] = center
                    brain.show_view(viewdicts[h][view], self._scene_size,
                                    cam_state)
        self.brains = brains

    def _set_window_properties(self, size, background, foreground):
        """Set window properties that are used elsewhere."""
        # old option "size" sets both width and height
        from matplotlib.colors import colorConverter
        if isinstance(size, (tuple, list)):
            width, height = size
        elif isinstance(size, int):
            width, height = size, size
        self._scene_size = height, width
        self._bg_color = colorConverter.to_rgb(background)
        if foreground is None:
            foreground = 'w' if sum(self._bg_color) < 2 else 'k'
        self._fg_color = colorConverter.to_rgb(foreground)

    def _check_hemi(self, hemi):
        """Check for safe single-hemi input, returns str."""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        return hemi

    def _read_scalar_data(self, source, hemi, name=None):
        """Load in scalar data from an image stored in a file or an array.

        Parameters
        ----------
        source : str or numpy array
            path to scalar data file or a numpy array
        name : str or None, optional
            name for the overlay in the internal dictionary

        Returns
        -------
        scalar_data : numpy array
            flat numpy array of scalar data
        name : str
            if no name was provided, deduces the name if filename was given
            as a source
        """
        # If source is a string, try to load a file
        if isinstance(source, string_types):
            if name is None:
                basename = os.path.basename(source)
                if basename.endswith(".gz"):
                    basename = basename[:-3]
                if basename.startswith("%s." % hemi):
                    basename = basename[3:]
                name = os.path.splitext(basename)[0]
            scalar_data = read_scalar_data(source)
        else:
            # Can't think of a good way to check that this will work nicely
            scalar_data = source

        return scalar_data, name

    def _get_display_range(self, scalar_data, min, max, sign):
        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"

        # Get data with a range that will make sense for automatic thresholding
        if sign == "neg":
            range_data = np.abs(scalar_data[np.where(scalar_data < 0)])
        elif sign == "pos":
            range_data = scalar_data[np.where(scalar_data > 0)]
        else:
            range_data = np.abs(scalar_data)

        # Get a numeric value for the scalar minimum
        if min is None:
            min = "robust_min"
        if min == "robust_min":
            min = np.percentile(range_data, 2)
        elif min == "actual_min":
            min = range_data.min()

        # Get a numeric value for the scalar maximum
        if max is None:
            max = "robust_max"
        if max == "robust_max":
            max = np.percentile(scalar_data, 98)
        elif max == "actual_max":
            max = range_data.max()

        return min, max

    def _get_overlay_limits(self, scalar_data, min, max, sign):
        """Get the limits of the overlay."""
        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"

        if sign in ["abs", "pos"]:
            pos_max = np.max((0.0, np.max(scalar_data)))
            if pos_max < min:
                thresh_low = pos_max
            else:
                thresh_low = min
            self.pos_lims = [thresh_low, min, max]
        else:
            self.pos_lims = None

        if sign in ["abs", "neg"]:
            neg_min = np.min((0.0, np.min(scalar_data)))
            if neg_min > -min:
                thresh_up = neg_min
            else:
                thresh_up = -min
            self.neg_lims = [thresh_up, -max, -min]
        else:
            self.neg_lims = None

        colormap = dict()
        if self.neg_lims is None:
            cmap, clim = 'Reds_r', (self.pos_lims[1], self.pos_lims[2])
            translucent = (None, self.pos_lims[1])
        if self.pos_lims is None:
            cmap, clim = 'PuBu', (self.neg_lims[1], self.neg_lims[2])
            translucent = (self.neg_lims[2], None)
        if self.neg_lims and self.pos_lims:
            cmap, clim = 'bwr', (self.neg_lims[1], self.pos_lims[2])
            translucent = None
        colormap['cmap'] = cmap
        colormap['clim'] = clim
        colormap['translucent'] = translucent

        return colormap

    ###########################################################################
    # ADDING DATA PLOTS
    def add_overlay(self, source, min=2, max="robust_max", sign="abs",
                    name=None, hemi=None):
        """Add an overlay to the overlay dict from a file or array.

        Parameters
        ----------
        source : str or numpy array
            path to the overlay file or numpy array with data
        min : float
            threshold for overlay display
        max : float
            saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)
        # load data here
        scalar_data, name = self._read_scalar_data(source, hemi, name=name)
        min, max = self._get_display_range(scalar_data, min, max, sign)
        if sign not in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
        old = OverlayData(scalar_data, min, max, sign)
        for brain in self.brains:
            if brain['hemi'] == hemi:
                brain['brain'].add_overlay(scalar_data.copy(), clim=(5., 20.))

    def add_data(self):
        """Doc."""
        raise NotImplementedError

    def add_annotation(self):
        """Doc."""
        raise NotImplementedError

    def add_label(self):
        """Doc."""
        raise NotImplementedError

    def remove_data(self):
        """Doc."""
        raise NotImplementedError

    def remove_labels(self):
        """Doc."""
        raise NotImplementedError

    def add_morphometry(self):
        """Doc."""
        raise NotImplementedError

    def add_foci(self):
        """Doc."""
        raise NotImplementedError

    def add_contour_overlay(self):
        """Doc."""
        raise NotImplementedError

    def add_text(self):
        """Doc."""
        raise NotImplementedError

    def update_text(self):
        """Doc."""
        raise NotImplementedError

    def reset_view(self):
        """Doc."""
        raise NotImplementedError

    def show_view(self):
        """Doc."""
        raise NotImplementedError

    def set_distance(self):
        """Doc."""
        raise NotImplementedError

    def set_surf(self):
        """Doc."""
        raise NotImplementedError

    def scale_data_colormap(self):
        """Doc."""
        raise NotImplementedError

    def set_data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def set_data_smoothing_steps(self):
        """Doc."""
        raise NotImplementedError

    def index_for_time(self):
        """Doc."""
        raise NotImplementedError

    def set_time(self):
        """Doc."""
        raise NotImplementedError

    def show_colorbar(self):
        """Doc."""
        raise NotImplementedError

    def hide_colorbar(self):
        """Doc."""
        raise NotImplementedError

    def close(self):
        """Doc."""
        raise NotImplementedError

    def show(self):
        """Display the figure."""
        from vispy import app
        app.run()

    ###########################################################################
    # SAVING
    def save_single_image(self):
        """Doc."""
        raise NotImplementedError

    def save_image(self):
        """Doc."""
        raise NotImplementedError

    def screenshot(self):
        """Doc."""
        raise NotImplementedError

    def screenshot_single(self):
        """Doc."""
        raise NotImplementedError

    def save_imageset(self):
        """Doc."""
        raise NotImplementedError

    def save_image_sequence(self):
        """Doc."""
        raise NotImplementedError

    def save_montage(self):
        """Doc."""
        raise NotImplementedError

    def save_movie(self):
        """Doc."""
        raise NotImplementedError

    def animate(self):
        """Doc."""
        raise NotImplementedError
