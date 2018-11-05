""" coding: utf-8 """

from typing import Callable, Generic, Iterable, Iterator, List, NamedTuple,\
    NewType, Optional, Text, Tuple, TypeVar, Union

import os
import numpy
import pyFAI

from fabio.edfimage import edfimage
from functools import partial
from h5py import Dataset, File
from itertools import cycle, repeat
from numpy import arange, arctan, array, ndarray, logical_or, ma, rad2deg, where, zeros_like
from numpy.ma import MaskedArray
try:
    from pyFAI.detectors import Detector as PyFAIDetector
except ImportError:
    from pyFAI import Detector as PyFAIDetector
from pyFAI.goniometer import GeometryTransformation, GoniometerRefinement
from pyFAI.gui import jupyter

# TODO the label should be part of the calibration frame

# NewTypes

Angle = NewType("Angle", float)
Calibrant = NewType("Calibrant", Text)
Detector = NewType("Detector", Text)
Length = NewType("Length", float)
NumExpr = NewType("NumExpr", Text)
Threshold = NewType("Threshold", int)
Wavelength = NewType("Wavelength", float)

# Typevar

T = TypeVar('T', int, float, Angle, Length)
C = TypeVar('C', float, ndarray)

class Parameter(Generic[T]):
    def __init__(self, name: Text, value: T, bounds: Tuple[T, T]) -> None:
        self.name = name
        self.value = value
        self.bounds = bounds

# Generic hdf5 access types.


DatasetPathContains = NamedTuple("DatasetPathContains", [("path", Text)])
DatasetPathContainsDefault = NamedTuple("DatasetPathContains", [("path", Text),
                                                                ("default", float)])

DatasetPathWithAttribute = NamedTuple("DatasetPathWithAttribute",
                                      [('attribute', Text),
                                       ('value', bytes)])

DatasetPath = Union[DatasetPathContains,
                    DatasetPathWithAttribute]


def _v_attrs(attribute: Text, value: Text, _name: Text, obj) -> Dataset:
    """extract all the images and accumulate them in the acc variable"""
    if isinstance(obj, Dataset):
        if attribute in obj.attrs and obj.attrs[attribute] == value:
            return obj


def _v_item(key: Text, name: Text, obj: Dataset) -> Dataset:
    if key in name:
        return obj


def get_dataset(h5file: File, path: DatasetPath) -> Optional[Dataset]:
    res = None
    if isinstance(path, DatasetPathContains):
        res = h5file.visititems(partial(_v_item, path.path))
    elif isinstance(path, DatasetPathContainsDefault):
        res = h5file.visititems(partial(_v_item, path.path))
    elif isinstance(path, DatasetPathWithAttribute):
        res = h5file.visititems(partial(_v_attrs,
                                        path.attribute, path.value))
    return res


def gen_dataset_value(h5file: File, path: DatasetPath, extend: bool=False) -> Iterator[C]:
    dataset = get_dataset(h5file, path)
    if isinstance(path, DatasetPathContains):
        if extend is True:
            yield from cycle(dataset)
        else:
            yield from dataset
    elif isinstance(path, DatasetPathContainsDefault):
        if dataset is None:
            yield from repeat(path.default)
        else:
            if extend is True:
                yield from cycle(dataset)
            else:
                yield from dataset
    elif isinstance(path, DatasetPathWithAttribute):
        if extend is True:
            yield from cycle(dataset)
        else:
            yield from dataset


# PyFAI calibration functions

CalibrationFunctions = NamedTuple("CalibrationFunctions",
                                  [("distance", NumExpr),
                                   ("poni1", NumExpr),
                                   ("poni2", NumExpr),
                                   ("rot1", NumExpr),
                                   ("rot2", NumExpr),
                                   ("rot3", NumExpr)])

Functions = NewType("Functions",
                    Tuple['CalibrationFunctions', List['Parameter']])


# Calibration Tx Tz (Mars)

CalibrationMarsTxTzFrame = NamedTuple("CalibrationMarsTxTzFrame", [("idx", int),
                                                                   ("label", Text),
                                                                   ("image", ndarray),
                                                                   ("tx", Length),
                                                                   ("tz", Length)])

CalibrationMarsTxTz = NamedTuple("CalibrationMarsTxTz",
                                 [("basedir", Text),
                                  ("filename", Text),
                                  ("images_path", DatasetPath),
                                  ("tx_path", DatasetPath),
                                  ("tz_path", DatasetPath),
                                  ("idxs", List[int]),
                                  ("to_use", Callable[[CalibrationMarsTxTzFrame], bool]),
                                  ("calibrant", Calibrant),
                                  ("detector", Detector),
                                  ("wavelength", Wavelength),
                                  ("functions", Functions),
                                  ("max_rings", int)])  # noqa

def gen_metadata_idx_mars_tx_tz(h5file: File,
                                calibration: CalibrationMarsTxTz,
                                indexes: Optional[Iterable[int]]=None,
                                to_use: Optional[bool]=True) -> Iterator[CalibrationMarsTxTzFrame]:  # noqa
    base = os.path.basename(calibration.filename)

    for idx, (image, tx, tz) in enumerate(zip(gen_dataset_value(h5file,
                                                                calibration.images_path),
                                              gen_dataset_value(h5file,
                                                                calibration.tx_path, True),
                                              gen_dataset_value(h5file,
                                                                calibration.tz_path, True))):
        if indexes is None or idx in indexes:
            label = base + "_{:d}".format(idx)
            frame = CalibrationMarsTxTzFrame(idx, label, image, tx, tz)
            if calibration.to_use(frame) == True or to_use == False:
                yield frame


def save_as_edf_mars_tx_tz(calibration: CalibrationMarsTxTz) -> None:
    """Save the multi calib images into edf files in order to do the first
    calibration and print the command line in order to do the
    calibration with pyFAI-calib
    """
    cmds = []
    with File(calibration.filename, mode='r') as h5file:
        for frame in gen_metadata_idx_mars_tx_tz(h5file, calibration, calibration.idxs):
            base = os.path.basename(calibration.filename)
            output = base + "_{:02d}.edf".format(frame.idx)
            edfimage(frame.image).write(os.path.join(calibration.basedir, output))  # noqa
            # temporary until pyFAI-calib2 works
            wavelength = calibration.wavelength * 1e10
            cmd = "cd {directory} && pyFAI-calib2 -w {wavelength} --calibrant {calibrant} -D {detector} {filename}".format(directory=calibration.basedir,  # noqa
                                                                                                                           wavelength=wavelength,  # noqa
                                                                                                                           calibrant=calibration.calibrant,  # noqa
                                                                                                                           detector=calibration.detector,  # noqa
                                                                                                                           filename=output)  # noqa
            cmds.append(cmd)
    return cmds


def optimize_with_new_images_mars_tx_tz(h5file: File,
                                        calibration: CalibrationMarsTxTz,
                                        gonioref,
                                        calibrant: pyFAI.calibrant.Calibrant,
                                        indexes: Iterable[int],
                                        pts_per_deg: float=1) -> None:
    """This function adds new images to the pool of data used for the
    refinement.  A set of new control points are extractred and a
    refinement step is performed at each iteration The last image of
    the serie is displayed

    """
    sg = None
    for frame in gen_metadata_idx_mars_tx_tz(h5file, calibration, indexes):
        print()
        if frame.label in gonioref.single_geometries:
            continue
        print(frame.label)
        sg = gonioref.new_geometry(frame.label, image=frame.image,
                                   metadata=frame,
                                   calibrant=calibrant)
        print(sg.extract_cp(max_rings=calibration.max_rings, pts_per_deg=pts_per_deg))

        # filter the cp to remove unwanted values.
        cp = sg.control_points
        cp.pop(0)

        sg.geometry_refinement.data = numpy.asarray(cp.getList(), dtype=numpy.float64)
        sg.control_points = cp

    print("*"*50)
    gonioref.refine2()
    if sg:
        sg.geometry_refinement.set_param(gonioref.get_ai(sg.get_position()).param)  # noqa
        jupyter.display(sg=sg)


def _calibration_mars_tx_tz(params: CalibrationMarsTxTz,
                            gonioref: Optional[GoniometerRefinement]=None,
                            indexes: Optional[Iterable[int]]=None) -> GoniometerRefinement:
    """Do a calibration with a bunch of images"""

    # Definition of the geometry refinement: the parameter order is
    # the same as the param_names
    calibrant = get_calibrant(params.calibrant,
                              params.wavelength)
    detector = get_detector(params.detector)

    (functions, initial_parameters) = params.functions
    parameters = {p.name: p.value for p in initial_parameters}
    bounds = {p.name: p.bounds for p in initial_parameters}
    param_names = [p.name for p in initial_parameters]

    # Let's refine poni1 and poni2 also as function of the distance:

    trans_function = GeometryTransformation(param_names=param_names,
                                            pos_names=["tx", "tz"],
                                            dist_expr=functions.distance,
                                            poni1_expr=functions.poni1,
                                            poni2_expr=functions.poni2,
                                            rot1_expr=functions.rot1,
                                            rot2_expr=functions.rot2,
                                            rot3_expr=functions.rot3)

    def pos_function(frame: CalibrationMarsTxTzFrame) -> Tuple[float, float]:
        """Definition of the function reading the detector position from the
        header of the image."""
        return (frame.tx, frame.tz)

    if gonioref is None:
        print("Empty refinement object:")
        gonioref = GoniometerRefinement(parameters,  # initial guess
                                        bounds=bounds,
                                        pos_function=pos_function,
                                        trans_function=trans_function,
                                        detector=detector,
                                        wavelength=params.wavelength)

    print(gonioref)

    # Let's populate the goniometer refinement object with the know poni

    with File(params.filename, mode='r') as h5file:
        for frame in gen_metadata_idx_mars_tx_tz(h5file, params, params.idxs):
            base = os.path.basename(params.filename)
            control_points = os.path.join(params.basedir, base + "_{:02d}.npt".format(frame.idx))  # noqa
            ai = pyFAI.load(os.path.join(params.basedir, base + "_{:02d}.poni".format(frame.idx)))  # noqa
            print(ai)

            gonioref.new_geometry(frame.label, frame.image, frame,
                                  control_points, calibrant, ai)

        print("Filled refinement object:")
        print(gonioref)
        print(os.linesep + "\tlabel \t tx")
        for k, v in gonioref.single_geometries.items():
            print(k, v.get_position())

        for g in gonioref.single_geometries.values():
            ai = gonioref.get_ai(g.get_position())
            print(ai)

        for sg in gonioref.single_geometries.values():
            jupyter.display(sg=sg)

        gonioref.refine2()

    for multi in [params]:
        with File(multi.filename, mode='r') as h5file:
            optimize_with_new_images_mars_tx_tz(h5file, multi, gonioref,
                                                calibrant, indexes, pts_per_deg=1)

    for idx, sg in enumerate(gonioref.single_geometries.values()):
        sg.geometry_refinement.set_param(gonioref.get_ai(sg.get_position()).param)  # noqa
        jupyter.display(sg=sg)

    return gonioref

def calibration_mars_tx_tz(paramss: List[CalibrationMarsTxTz],
                           json: Optional[str]=None) -> GoniometerRefinement:
    gonioref = None
    for params in paramss:
        gonioref = _calibration_mars_tx_tz(params, gonioref)

    if json is not None:
        gonioref.save(json)

    return gonioref

def integrate_mars_tx_tz(json: str,
                         paramss: List[CalibrationMarsTxTz],
                         f: Callable[[ndarray], ndarray],
                         plot_calibrant: bool=False,
                         save: bool=False,
                         n: int=10000,
                         lst_mask: ndarray=None,
                         lst_flat: ndarray=None,
                         to_use: bool=False,
                         no_plot: bool=False) -> None:
    """Integrate a file with a json calibration file"""
    gonio = pyFAI.goniometer.Goniometer.sload(json)
    images = []
    positions = []

    for params in paramss:
        with File(params.filename, mode='r') as h5file:
            for frame in gen_metadata_idx_mars_tx_tz(h5file, params, to_use=to_use):
                images.append(f(frame.image))
                positions.append((frame.tx, frame.tz))


    mai = gonio.get_mg(positions)
    res = mai.integrate1d(images, n,
                              lst_mask=lst_mask, lst_flat=lst_flat)
    if save is True:
        try:
            os.makedirs(params.basedir)
        except os.error:
            pass
        ofilename = os.path.join(params.basedir,
                                 os.path.basename(params.filename) + '.txt')
        numpy.savetxt(ofilename, numpy.vstack([res.radial, res.intensity]).T)
        print("Saves as: {}".format(ofilename))
    if no_plot is False:
        if plot_calibrant:
            calibrant = get_calibrant(params.calibrant, params.wavelength)
            jupyter.plot1d(res, calibrant)
        else:
            jupyter.plot1d(res)
    return res

# Calibration K6C (Diffabs)


CalibrationFrame = NamedTuple("CalibrationFrame", [("idx", int),
                                                   ("label", Text),
                                                   ("image", ndarray),
                                                   ("delta", Angle)])

Calibration = NamedTuple("Calibration",
                         [("basedir", Text),
                          ("filename", Text),
                          ("images_path", DatasetPath),
                          ("deltas_path", DatasetPath),
                          ("idxs", List[int]),
                          ("to_use", Callable[[CalibrationFrame], bool]),
                          ("calibrant", Calibrant),
                          ("detector", Detector),
                          ("wavelength", Wavelength),
                          ("functions", Functions)])  # noqa


def gen_metadata_idx(h5file: File,
                     calibration: Calibration,
                     indexes: Optional[Iterable[int]]=None,
                     to_use: Optional[bool]=True) -> Iterator[CalibrationFrame]:  # noqa
    images = get_dataset(h5file, calibration.images_path)
    if indexes is None:
        indexes = range(images.shape[0])
    deltas = get_dataset(h5file, calibration.deltas_path)
    base = os.path.basename(calibration.filename)
    for idx in indexes:
        label = base + "_{:d}".format(idx)
        frame = CalibrationFrame(idx, label, images[idx], deltas[idx])
        if calibration.to_use(frame) == True or to_use == False:
            yield CalibrationFrame(idx, label, images[idx], deltas[idx])


# Mythen Calibration

Mythen = NamedTuple("Mythen", [("dataset", DatasetPath),
                               ("functions", Functions)])  # noqa

MythenFrame = NamedTuple("Mythen", [("data", MaskedArray),
                                    ("tth", ndarray)])

MythenCalibrationFrame = NamedTuple("MythenCalibrationFrame",
                                    [("idx", int),
                                     ("label", Text),
                                     ("mythens", List[MythenFrame]),
                                     ("tth2C", Angle)])

MythenCalibration = NamedTuple("MythenCalibration",
                               [("basedir", Text),
                                ("filename", Text),
                                ("calibrant", Calibrant),
                                ("wavelength", Wavelength),
                                ("threshold", Threshold),
                                ("mythens", List[Mythen]),
                                ("tth2C", DatasetPath),
                                ("rings", List[int])])

Pixel = NewType("Pixel", int)

Peak = NamedTuple("Peak", [("position", Pixel),
                           ("intensity", int),
                           ("tth", Angle),
                           ("index", int)])


def mythenTth(tth2C: float, module: int, positions: Optional[List[int]]=None) -> ndarray:
    """Compute the real tth for each module"""
    size = 1280  # pixels
    center = size / 2  # pixel
    module_central = 4
    distance = 720  # mm
    decalage = 60 + (module_central - module) * 5.7
    positions = array(positions) if positions else arange(1280)
    return tth2C - decalage - rad2deg(arctan((positions - center) * 0.05 / distance))  # noqa


def mkMythenFrame(data: ndarray,
                  tth2C: Angle,
                  module: int) -> MythenFrame:
    # compute the masks
    mask = zeros_like(data, dtype=bool)
    mask[:20] = mask[-20:] = True
    mask = logical_or(mask,
                      where(data == -2, True, False))
    data = ma.masked_array(data, mask)

    tth = mythenTth(tth2C, module)

    return MythenFrame(data, tth)


def gen_metadata_idx_mythen(h5file: File,
                            calibration: MythenCalibration,
                            indexes: Optional[Iterable[int]]=None) -> Iterator[MythenCalibrationFrame]:  # noqa
    h5nodes = [get_dataset(h5file, m.dataset) for m in calibration.mythens]
    tth2C = get_dataset(h5file, calibration.tth2C)

    base = os.path.basename(calibration.filename)

    if indexes is None:
        indexes = range(h5nodes[0].shape[0])

    for idx in indexes:
        label = base + "_{:d}".format(idx)
        tth = tth2C[idx]
        yield MythenCalibrationFrame(idx, label,
                                     [mkMythenFrame(node[idx], tth, m)
                                      for m, node in enumerate(h5nodes)],
                                     tth)


def save_as_edf(calibration: Calibration) -> None:
    """Save the multi calib images into edf files in order to do the first
    calibration and print the command line in order to do the
    calibration with pyFAI-calib
    """
    cmds = []
    with File(calibration.filename, mode='r') as h5file:
        for frame in gen_metadata_idx(h5file, calibration, calibration.idxs):
            base = os.path.basename(calibration.filename)
            output = base + "_{:02d}.edf".format(frame.idx)
            edfimage(frame.image).write(os.path.join(calibration.basedir, output))  # noqa
            # temporary until pyFAI-calib2 works
            wavelength = calibration.wavelength * 1e10
            cmd = "cd {directory} && pyFAI-calib2 -w {wavelength} --calibrant {calibrant} -D {detector} {filename}".format(directory=calibration.basedir,  # noqa
                                                                                                                          wavelength=wavelength,  # noqa
                                                                                                                          calibrant=calibration.calibrant,  # noqa
                                                                                                                          detector=calibration.detector,  # noqa
                                                                                                                          filename=output)  # noqa
            cmds.append(cmd)
    return cmds


def get_total_length(calibration: Calibration) -> int:
    """Return the total number of frame of the calib file"""
    with File(calibration.filename, mode='r') as h5file:
        images = get_dataset(h5file, calibration.images_path)
        return images.shape[0]


def optimize_with_new_images(h5file: File,
                             calibration: Calibration,
                             gonioref,
                             calibrant: pyFAI.calibrant.Calibrant,
                             indexes: Iterable[int],
                             pts_per_deg: float=1) -> None:
    """This function adds new images to the pool of data used for the
    refinement.  A set of new control points are extractred and a
    refinement step is performed at each iteration The last image of
    the serie is displayed

    """
    sg = None
    for frame in gen_metadata_idx(h5file, calibration, indexes):
        print()
        if frame.label in gonioref.single_geometries:
            continue
        print(frame.label)
        sg = gonioref.new_geometry(frame.label, image=frame.image,
                                   metadata=frame,
                                   calibrant=calibrant)
        print(sg.extract_cp(pts_per_deg=pts_per_deg))
    print("*"*50)
    gonioref.refine2()
    if sg:
        sg.geometry_refinement.set_param(gonioref.get_ai(sg.get_position()).param)  # noqa
        jupyter.display(sg=sg)


def get_calibrant(calibrant: Calibrant,
                  wavelength: Wavelength) -> pyFAI.calibrant.Calibrant:
    """Return the calibrant with the right wavelength"""
    pyFAI_calibrant = pyFAI.calibrant.get_calibrant(calibrant)
    pyFAI_calibrant.wavelength = wavelength
    return pyFAI_calibrant


def get_detector(detector: Detector) -> PyFAIDetector:
    return pyFAI.detector_factory(detector)


def calibration(json: str,
                params: Calibration,
                indexes: Optional[Iterable[int]]=None) -> None:
    """Do a calibration with a bunch of images"""

    # Definition of the geometry refinement: the parameter order is
    # the same as the param_names
    calibrant = get_calibrant(params.calibrant,
                              params.wavelength)
    detector = get_detector(params.detector)

    (functions, initial_parameters) = params.functions
    parameters = {p.name: p.value for p in initial_parameters}
    bounds = {p.name: p.bounds for p in initial_parameters}
    param_names = [p.name for p in initial_parameters]

    # Let's refine poni1 and poni2 also as function of the distance:

    trans_function = GeometryTransformation(param_names=param_names,
                                            pos_names=["delta"],
                                            dist_expr=functions.distance,
                                            poni1_expr=functions.poni1,
                                            poni2_expr=functions.poni2,
                                            rot1_expr=functions.rot1,
                                            rot2_expr=functions.rot2,
                                            rot3_expr=functions.rot3)

    def pos_function(frame: CalibrationFrame) -> Tuple[float]:
        """Definition of the function reading the detector position from the
        header of the image."""
        return (frame.delta,)

    gonioref = GoniometerRefinement(parameters,  # initial guess
                                    bounds=bounds,
                                    pos_function=pos_function,
                                    trans_function=trans_function,
                                    detector=detector,
                                    wavelength=params.wavelength)

    print("Empty refinement object:")
    print(gonioref)

    # Let's populate the goniometer refinement object with the know poni

    with File(params.filename, mode='r') as h5file:
        for frame in gen_metadata_idx(h5file, params, params.idxs):
            base = os.path.basename(params.filename)
            control_points = os.path.join(params.basedir, base + "_{:02d}.npt".format(frame.idx))  # noqa
            ai = pyFAI.load(os.path.join(params.basedir, base + "_{:02d}.poni".format(frame.idx)))  # noqa
            print(ai)

            gonioref.new_geometry(frame.label, frame.image, frame,
                                  control_points, calibrant, ai)

        print("Filled refinement object:")
        print(gonioref)
        print(os.linesep + "\tlabel \t tx")
        for k, v in gonioref.single_geometries.items():
            print(k, v.get_position())

        for g in gonioref.single_geometries.values():
            ai = gonioref.get_ai(g.get_position())
            print(ai)

        for sg in gonioref.single_geometries.values():
            jupyter.display(sg=sg)

        gonioref.refine2()

    for multi in [params]:
        with File(multi.filename, mode='r') as h5file:
            optimize_with_new_images(h5file, multi, gonioref,
                                     calibrant, indexes, pts_per_deg=10)

    for idx, sg in enumerate(gonioref.single_geometries.values()):
        sg.geometry_refinement.set_param(gonioref.get_ai(sg.get_position()).param)  # noqa
        jupyter.display(sg=sg)

    gonioref.save(json)

# Integrate


def integrate(json: str,
              params: Calibration,
              f: Callable[[ndarray], ndarray],
              plot_calibrant: bool=False,
              save: bool=False,
              n: int=10000,
              lst_mask: ndarray=None,
              lst_flat: ndarray=None,
              to_use: bool=False) -> None:
    """Integrate a file with a json calibration file"""
    gonio = pyFAI.goniometer.Goniometer.sload(json)
    with File(params.filename, mode='r') as h5file:
        images = []
        deltas = []
        for frame in gen_metadata_idx(h5file, params, to_use=to_use):
            images.append(f(frame.image))
            deltas.append((frame.delta,))
        mai = gonio.get_mg(deltas)
        res = mai.integrate1d(images, n,
                              lst_mask=lst_mask, lst_flat=lst_flat)
        if save is True:
            try:
                os.makedirs(params.basedir)
            except os.error:
                pass
            numpy.savetxt(os.path.join(params.basedir,
                                       os.path.basename(params.filename) + '.txt'),
                          numpy.vstack([res.radial, res.intensity]).T)
        if plot_calibrant:
            calibrant = get_calibrant(params.calibrant, params.wavelength)
            jupyter.plot1d(res, calibrant)
        else:
            jupyter.plot1d(res)
        return res
