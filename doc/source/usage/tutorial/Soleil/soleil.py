""" coding: utf-8 """

from typing import Generic, Iterator, List, NamedTuple, NewType, Text, Tuple,\
    TypeVar, Union

import os
import pyFAI

from fabio.edfimage import edfimage
from functools import partial
from h5py import Dataset, File
from numpy import ndarray
from pyFAI.goniometer import GeometryTransformation, GoniometerRefinement
from pyFAI.gui import jupyter

# TODO the label should be part of the calibration frame

# NewTypes

Angle = NewType("Angle", float)
Calibrant = NewType("Calibrant", Text)
Detector = NewType("Detector", Text)
Length = NewType("Length", float)
NumExpr = NewType("NumExpr", Text)
Wavelength = NewType("Wavelength", float)

# Typevar

T = TypeVar('T', int, float, Angle, Length)


class Parameter(Generic[T]):
    def __init__(self, name: Text, value: T, bounds: Tuple[T, T]) -> None:
        self.name = name
        self.value = value
        self.bounds = bounds

# Generic hdf5 access types.


DatasetPathContains = NamedTuple("DatasetPathContains", [("path", Text)])

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


def get_dataset(h5file: File, path: DatasetPath) -> Dataset:
    res = None
    if isinstance(path, DatasetPathContains):
        res = h5file.visititems(partial(_v_item, path.path))
    elif isinstance(path, DatasetPathWithAttribute):
        res = h5file.visititems(partial(_v_attrs,
                                        path.attribute, path.value))
    return res

# Calibration


CalibrationFrame = NamedTuple("CalibrationFrame", [("idx", int),
                                                   ("image", ndarray),
                                                   ("delta", Angle)])

CalibrationFunctions = NamedTuple("CalibrationFunctions",
                                  [("distance", NumExpr),
                                   ("poni1", NumExpr),
                                   ("poni2", NumExpr),
                                   ("rot1", NumExpr),
                                   ("rot2", NumExpr),
                                   ("rot3", NumExpr)])

CalibrationParameters = NamedTuple("CalibrationParameters",
                                   [("distance", Parameter[Length]),
                                    ("poni1", Parameter[Length]),
                                    ("poni2", Parameter[Length]),
                                    ("rot1_scale", float),
                                    ("rot1_offset", Parameter[Angle]),
                                    ("rot2", Parameter[Angle]),
                                    ("rot3", Parameter[Angle])])

Calibration = NamedTuple("Calibration",
                         [("basedir", Text),
                          ("filename", Text),
                          ("images_path", DatasetPath),
                          ("deltas_path", DatasetPath),
                          ("idxs", List[int]),
                          ("calibrant", Calibrant),
                          ("detector", Detector),
                          ("wavelength", Wavelength),
                          ("functions", Tuple[CalibrationFunctions, List[Parameter]])])  # noqa


def gen_metadata_idx(h5file: File,
                     calibration: Calibration) -> Iterator[CalibrationFrame]:
    images = get_dataset(h5file, calibration.images_path)
    deltas = get_dataset(h5file, calibration.deltas_path)
    for idx in calibration.idxs:
        yield CalibrationFrame(idx, images[idx], deltas[idx])


def gen_metadata_all(h5file: File,
                     calibration: Calibration) -> Iterator[CalibrationFrame]:
    images = get_dataset(h5file, calibration.images_path)
    deltas = get_dataset(h5file, calibration.deltas_path)
    for idx in range(images.shape[0]):
        yield CalibrationFrame(idx, images[idx], deltas[idx])


def save_as_edf(calibration: Calibration) -> None:
    """Save the multi calib images into edf files in order to do the first
    calibration and print the command line in order to do the
    calibration with pyFAI-calib
    """
    with File(calibration.filename, mode='r') as h5file:
        for frame in gen_metadata_idx(h5file, calibration):
            base = os.path.basename(calibration.filename)
            output = base + "_{:02d}.edf".format(frame.idx)
            edfimage(frame.image).write(os.path.join(calibration.basedir, output))  # noqa
            # temporary until pyFAI-calib2 works
            wavelength = calibration.wavelength * 1e10
            cmd = "cd {directory} && pyFAI-calib -w {wavelength} --calibrant {calibrant} -D {detector} {filename}".format(directory=calibration.basedir,  # noqa
                                                                                                                          wavelength=wavelength,  # noqa
                                                                                                                          calibrant=calibration.calibrant,  # noqa
                                                                                                                          detector=calibration.detector,  # noqa
                                                                                                                          filename=output)  # noqa
            print(cmd)


def get_total_length(calibration: Calibration) -> int:
    """Return the total number of frame of the calib file"""
    with File(calibration.filename, mode='r') as h5file:
        images = get_dataset(h5file, calibration.images_path)
        return images.shape[0]


def optimize_with_new_images(h5file: File,
                             calibration: Calibration,
                             gonioref,
                             calibrant: pyFAI.calibrant.Calibrant,
                             pts_per_deg: float=1) -> None:
    """This function adds new images to the pool of data used for the
    refinement.  A set of new control points are extractred and a
    refinement step is performed at each iteration The last image of
    the serie is displayed

    """
    sg = None
    for n, frame in enumerate(gen_metadata_all(h5file, calibration)):
        print()
        base = os.path.basename(calibration.filename)

        label = base + "_%d" % (frame.idx,)
        if label in gonioref.single_geometries:
            continue
        print(label)
        sg = gonioref.new_geometry(label, image=frame.image, metadata=frame,
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


def get_detector(detector: Detector) -> pyFAI.Detector:
    return pyFAI.detector_factory(detector)


def calibration(json: str, params: Calibration) -> None:
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
        for frame in gen_metadata_idx(h5file, params):
            base = os.path.basename(params.filename)

            label = base + "_%d" % (frame.idx,)
            control_points = os.path.join(params.basedir, base + "_{:02d}.npt".format(frame.idx))  # noqa
            ai = pyFAI.load(os.path.join(params.basedir, base + "_{:02d}.poni".format(frame.idx)))  # noqa
            print(ai)

            gonioref.new_geometry(label, frame.image, frame,
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
            optimize_with_new_images(h5file, multi, gonioref, calibrant)

    for idx, sg in enumerate(gonioref.single_geometries.values()):
        sg.geometry_refinement.set_param(gonioref.get_ai(sg.get_position()).param)  # noqa
        jupyter.display(sg=sg)

    gonioref.save(json)

    # pylab.show()


# Integrate

'''
def integrate(json: str) -> None:
    """Integrate a file with a json calibration file"""
    filename = os.path.join(ROOT, "scan_77_01.nxs")
    gonio = pyFAI.goniometer.Goniometer.sload(json)
    wavelength = 4.85945727522e-11
    multicalib = MultiCalib(os.path.join(ROOT, "scan_4_01.nxs"),
                            MetaDataSource("",
                                           H5PathContains("scan_data/actuator_1_1"),
                                           H5PathOptionalItemValue("MARS/D03-1-CX0__DT__DTC_2D-MT_Tz__#1/raw_value", -1.0)),
                            [], "LaB6", "xpad_flat", wavelength)

    with h5py.File(filename, mode='r') as h5file:
        images = []
        positions = []
        for metadata in gen_metadata(h5file, multicalib):
            images.append(metadata.img)
            positions.append((metadata.tx, metadata.tz))
        mai = gonio.get_mg(positions)
        res = mai.integrate1d(images, 10000)
        jupyter.plot1d(res)
        pylab.show()
'''
