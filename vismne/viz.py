"""Need doc."""
import os

import numpy as np

from vispy import scene

from .io import _get_subjects_dir, read_scalar_data
from .utils import (Surface, _check_units, string_types, Colormap,
                    smoothing_matrix, mesh_edges)
from .visuals import BrainMesh


lh_viewdict = {'lat': {'v': (-90., 0.), 'r': 90., 'xyz': [1, 2]},
               'med': {'v': (90., 0.), 'r': -90., 'xyz': [1, 2]},
               'ros': {'v': (180., 0.), 'r': -180., 'xyz': [0, 2]},
               'cau': {'v': (0., 0.), 'r': 0., 'xyz': [0, 2]},
               'dor': {'v': (0., 90.), 'r': 90., 'xyz': [0, 1]},
               'ven': {'v': (0., -90.), 'r': 90., 'xyz': [0, 1]},
               'fro': {'v': (120., 80.), 'r': 106.739, 'xyz': [0, 1]},
               'par': {'v': (-120., 60.), 'r': 49.106, 'xyz': [0, 1]}}
rh_viewdict = {'lat': {'v': (90., 0.), 'r': -90., 'xyz': [1, 2]},
               'med': {'v': (-90., 0.), 'r': 90., 'xyz': [1, 2]},
               'ros': {'v': (180., 0.), 'r': 180., 'xyz': [0, 2]},
               'cau': {'v': (0., 0.), 'r': 0., 'xyz': [0, 2]},
               'dor': {'v': (0., 90.), 'r': 90., 'xyz': [0, 1]},
               'ven': {'v': (0., -90.), 'r': 90., 'xyz': [0, 1]},
               'fro': {'v': (60., 80.), 'r': -106.739, 'xyz': [0, 1]},
               'par': {'v': (-60., 60.), 'r': -49.106, 'xyz': [0, 1]}}
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
        views to use (lat, med, ros, cau, dor, ven, fro, par). By default,
        only lateral view is shown.
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

    def __init__(self, subject_id, hemi, surf, title=None, views=None,
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

        if views is None:
            views = ['lat']
        if not isinstance(views, list):
            views = [views]
        self._n_rows = len(views)

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
        for h in geo_hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and (maybe) curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo

        # _______________________ PARENT _______________________
        # Deal with making figures
        self._set_window_properties(size, background, foreground)
        del background, foreground
        # Camera, canvas and grid : :
        self._sc = scene.SceneCanvas(bgcolor=self._bg_color, show=False,
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
                        parent.height_max = self._scene_size[0] / self._n_rows
                        kwrci = dict(parent=parent, camera=camera)
                        self._parents[(ri, ci)] = kwrci
                        parent.camera = camera
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
        self._times = None
        self.n_times = None
        self._cbar_is_displayed = False

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

        # _____________________ ABS / POS / NEG _____________________
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

        # _____________________ COLORMAP _____________________
        colormap = dict()
        if self.neg_lims is None:
            cmap, clim = 'YlOrRd_r', (self.pos_lims[1], self.pos_lims[2])
            translucent = (self.pos_lims[0], None)
        if self.pos_lims is None:
            cmap, clim = 'PuBu', (self.neg_lims[1], self.neg_lims[2])
            translucent = (None, self.neg_lims[0])
        if self.neg_lims and self.pos_lims:
            cmap = ['PuBu', 'YlOrRd_r']
            clim = [(self.neg_lims[1], self.neg_lims[2]),
                    (self.pos_lims[1], self.pos_lims[2])]
            translucent = (self.neg_lims[0], self.pos_lims[0])
        colormap['cmap'] = cmap
        colormap['clim'] = clim
        colormap['translucent'] = translucent

        # _____________________ COLORBAR _____________________
        limits = clim
        if isinstance(clim, list):
            limits = (self.neg_lims[1], self.pos_lims[2])
        cbar = Colormap(cmap=cmap)

        return colormap, cbar, limits

    def _add_colorbar(self, colormap, clim=None, size=(60, 4), height_max=100,
                      orientation='bottom', title=None):
        """Add a colorbar to the scene.

        Parameters
        ----------
        colormap : vispy.color.Colormap
            Visbrain-mne colormap instance.
        clim : tuple | None
            Colorbar limits.
        size : tuple | (60, 4)
            Size of the colorbar.
        height_max : float | 100.
            Height max of the colorbar subplot.
        orientation : {'left', 'right', 'top', 'bottom'}
            The orientation of the colorbar;
        title : string | None
            Colorbar title.
        """
        assert isinstance(colormap, Colormap)
        # Create the colorbar :
        if not hasattr(self, '_cbar'):
            # Create the rectangular camera :
            w, h = size
            rect = [-3 * w / 2, -h, w * 3, 5 * h]
            camera = scene.cameras.PanZoomCamera(rect=rect)
            # Create the subplot :
            r = 1 if self._hemi in ['lh', 'rh', 'both'] else 2
            parent = self._grid.add_view(row=self._n_rows, col=0, col_span=r,
                                         camera=camera)
            parent.height_max = height_max
            self._cbar = scene.ColorBar(colormap.vispy, orientation, size,
                                        clim=clim, label_str=title,
                                        parent=parent.scene,
                                        label_color=self._fg_color)
        else:
            self._cbar.cmap = colormap.vispy
            self._cbar._ticks[0].text = '%.2e' % clim[0]
            self._cbar._ticks[1].text = '%.2e' % clim[1]
        self._cbar.update()
        self._cbar_is_displayed = True

    ###########################################################################
    # ADDING DATA PLOTS
    def add_overlay(self, source, min=2, max="robust_max", sign="abs",
                    name=None, hemi=None, colorbar=True, colorbar_title=None):
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
        colorbar : bool | True
            Add a colorbar to the figure.
        colorbar_title : string | None
            Colorbar title.
        """
        hemi = self._check_hemi(hemi)
        # load data here
        scalar_data, name = self._read_scalar_data(source, hemi, name=name)
        min, max = self._get_display_range(scalar_data, min, max, sign)
        if sign not in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
        cmap, cbar, lim = self._get_overlay_limits(scalar_data, min, max, sign)
        for brain in self.brains:
            if brain['hemi'] == hemi:
                brain['brain'].add_overlay(scalar_data.copy(), **cmap)
        # Colorbar :
        if colorbar:
            self._add_colorbar(cbar, clim=lim, title=colorbar_title)

    def add_data(self, array, min=None, max=None, thresh=None, colormap="auto",
                 alpha=1, vertices=None, smoothing_steps=20, time=None,
                 time_label="time index=%d", colorbar=True, hemi=None,
                 remove_existing=False, time_label_size=14, initial_time=None,
                 mid=None, center=None, transparent=False, verbose=None):
        """Display data from a numpy array on the surface.

        Parameters
        ----------
        array : numpy array, shape (n_vertices[, n_times])
            Data array.
        min : float
            min value in colormap (uses real min if None)
        max : float
            max value in colormap (uses real max if None)
        thresh : None or float
            if not None, values below thresh will not be visible
        colormap : string, list of colors, or array
            name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            alpha level to control opacity of the overlay.
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int or None
            number of smoothing steps (smoothing is used if len(data) < nvtx)
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D or 3D)
        time_label : str | callable | None
            format of the time label (a format string, a function that maps
            floating point time values to strings, or None for no label)
        colorbar : bool
            whether to add a colorbar to the figure
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14)
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        mid : float
            intermediate value in colormap (middle between min and max if None)
        center : float or None
            if not None, center of a divergent colormap, changes the meaning of
            min, max and mid, see :meth:`scale_data_colormap` for further info.
        transparent : bool
            if True: use a linear transparency between fmin and fmid and make
            values below fmin fully transparent (symmetrically for divergent
            colormaps)
        verbose : bool, str, int, or None
            If not None, override default verbose level.
        """
        hemi = self._check_hemi(hemi)
        array = np.asarray(array)

        # ____________________ MIN / MAX / CENTER ____________________
        if center is None:
            if min is None:
                min = array.min() if array.size > 0 else 0
            if max is None:
                max = array.max() if array.size > 0 else 1
        else:
            if min is None:
                min = 0
            if max is None:
                max = np.abs(center - array).max() if array.size > 0 else 1
        if mid is None:
            mid = (min + max) / 2.
        _check_limits(min, mid, max, extra='')

        # ____________________ SMOOTHING ____________________
        if len(array) < self.geo[hemi].x.shape[0]:
            if vertices is None:
                raise ValueError("len(data) < nvtx (%s < %s): the vertices "
                                 "parameter must not be None"
                                 % (len(array), self.geo[hemi].x.shape[0]))
            adj_mat = mesh_edges(self.geo[hemi].faces)
            smooth_mat = smoothing_matrix(vertices, adj_mat, smoothing_steps)
        else:
            smooth_mat = None

        # ____________________ TIME INDEX ____________________
        if array.ndim <= 1:
            initial_time_index = None
        else:
            # check time array
            if time is None:
                time = np.arange(array.shape[-1])
            else:
                time = np.asarray(time)
                if time.shape != (array.shape[-1],):
                    raise ValueError('time has shape %s, but need shape %s '
                                     '(array.shape[-1])' %
                                     (time.shape, (array.shape[-1],)))

            if self.n_times is None:
                self.n_times = len(time)
                self._times = time
            elif len(time) != self.n_times:
                raise ValueError("New n_times is different from previous "
                                 "n_times")
            elif not np.array_equal(time, self._times):
                raise ValueError("Not all time values are consistent with "
                                 "previously set times.")

            # initial time
            if initial_time is not None:
                initial_time_index = self.index_for_time(initial_time)
            else:
                initial_time_index = 0

        # ____________________ OVERLAY ____________________
        for brain in self.brains:
            if brain['hemi'] == hemi:
                brain['brain'].add_data(array[:, initial_time_index],
                                        smooth_mat, cmap=colormap)

        self.scale_data_colormap(min, mid, max, transparent, alpha)

        # ____________________ COLORBAR ____________________
        # time label
        if isinstance(time_label, string_types):
            time_label_fmt = time_label

            def time_label(x):
                return time_label_fmt % x
        # Colorbar :
        if colorbar:
            title = time_label(time[initial_time_index])
            cbar = self.brains[0]['brain'].get_colormap(0)
            self._add_colorbar(cbar, clim=cbar['clim'], title=title)

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

    def scale_data_colormap(self, fmin, fmid, fmax, transparent, alpha=1.):
        """Doc."""
        kw = dict(clim=(fmin, fmax), alpha=alpha, translucent=(fmid, None),
                  smooth=transparent)
        for brain in self.brains:
            brain['brain'].update_colormap(0, **kw)

        if self._cbar_is_displayed:
            cbar = self.brains[0]['brain'].get_colormap(0)
            self._add_colorbar(cbar, clim=cbar['clim'])

    def set_data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def set_data_smoothing_steps(self, smoothing_steps, verbose=None):
        """Set the number of smoothing steps.

        Parameters
        ----------
        smoothing_steps : int
            Number of smoothing steps
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).
        """
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                adj_mat = mesh_edges(self.geo[hemi].faces)
                smooth_mat = smoothing_matrix(data["vertices"],
                                              adj_mat, smoothing_steps)
                data["smooth_mat"] = smooth_mat

                # Redraw
                if data["array"].ndim == 1:
                    plot_data = data["array"]
                elif data["array"].ndim == 2:
                    plot_data = data["array"][:, data["time_idx"]]
                else:  # vector-valued
                    plot_data = data["magnitude"][:, data["time_idx"]]

                plot_data = data["smooth_mat"] * plot_data
                for brain in self.brains:
                    if brain.hemi == hemi:
                        brain.set_data(data['layer_id'], plot_data)

                # Update data properties
                data["smoothing_steps"] = smoothing_steps

    def index_for_time(self, time, rounding='closest'):
        """Find the data time index closest to a specific time point.

        Parameters
        ----------
        time : scalar
            Time.
        rounding : 'closest' | 'up' | 'down'
            How to round if the exact time point is not an index.

        Returns
        -------
        index : int
            Data time index closest to time.
        """
        if self.n_times is None:
            raise RuntimeError("Brain has no time axis")
        times = self._times

        # Check that time is in range
        tmin = np.min(times)
        tmax = np.max(times)
        max_diff = (tmax - tmin) / (len(times) - 1) / 2
        if time < tmin - max_diff or time > tmax + max_diff:
            err = ("time = %s lies outside of the time axis "
                   "[%s, %s]" % (time, tmin, tmax))
            raise ValueError(err)

        if rounding == 'closest':
            idx = np.argmin(np.abs(times - time))
        elif rounding == 'up':
            idx = np.nonzero(times >= time)[0][0]
        elif rounding == 'down':
            idx = np.nonzero(times <= time)[0][-1]
        else:
            err = "Invalid rounding parameter: %s" % repr(rounding)
            raise ValueError(err)

        return idx

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
        self._sc.show(visible=True)
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


def _check_limits(fmin, fmid, fmax, extra='f'):
    """Check for monotonicity."""
    if fmin >= fmid:
        raise ValueError('%smin must be < %smid, got %0.4g >= %0.4g'
                         % (extra, extra, fmin, fmid))
    if fmid >= fmax:
        raise ValueError('%smid must be < %smax, got %0.4g >= %0.4g'
                         % (extra, extra, fmid, fmax))
