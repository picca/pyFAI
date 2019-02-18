
# coding: utf-8

# # Calibration of the 9-Mythen detector at the Cristal beamline at Soleil
#
# Mythen detectors are 1D-strip detector sold by Dectris.
# On the Cristal beamline at Soleil, 9 of them are mounted on the goniometer.
#
# This notebook explains how to calibrate precisely their position (including the wavelength used) as function of the goniometer position.
#
# All input data are provided in a Nexus file wich contrains both the (approximate) energy, the goniometer positions (500 points have been measured) and the measured signal.
#
# As pyFAI is not made for 1D data, the Mythen detector will be considered as a 1x1280 image.
#
# We start by importing a whole bunch of modules:

# In[1]:


# In[2]:

import os

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy

import h5py

# from pyFAI import goniometer
from pyFAI.detectors import Detector
from pyFAI.goniometer import ExtendedTransformation, GoniometerRefinement
from pyFAI.control_points import ControlPoints
# from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.gui import jupyter
from pyFAI.units import hc
from pyFAI.calibrant import get_calibrant
from pyFAI.containers import Integrate1dResult

# import ipywidgets as widgets
# from IPython.display import display

from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
from scipy.optimize import bisect, minimize
from scipy.spatial import distance_matrix
import time

start_time = time.time()


# The data file can be downoaded from:
# http://www.silx.org/pub/pyFAI/gonio/LaB6_17keV_att3_tth2C_24_01_2018_19-43-20_1555.nxs

# In[3]:

# Open the Nexus file and retrieve the actual positions:

ROOT = "/nfs/ruche-cristal/cristal-soleil/com-cristal/2018/Run1/99180029_Mythen"  # NOQA
filename = os.path.join(ROOT, "LaB6_17keV_att3_tth2C_24_01_2018_19-43-20_1555.nxs")  # NOQA

h5 = h5py.File(filename, mode="r")
position = h5["/LaB6_17keV_att3_1555/scan_data/actuator_1_1"][:]
print("Positions: ", position[:5], "...")


# In[4]:

# Read all data

data = {}  # type: Dict[str, ndarray]
ds_names = []
for idx in range(1, 13):
    name = "data_%02i" % idx
    ds = h5["/LaB6_17keV_att3_1555/scan_data/"+name][:]
    print(name, ds.shape)
    if ds.shape[1] < 2000:
        # Keep only the single modules
        data[name] = ds
        ds_names.append(name)

print(ds_names)


# In[5]:

# Define a Mythen-detector mounted vertically:

class MythenV(Detector):
    "Verical Mythen dtrip detector from Dectris"
    aliases = ["MythenV 1280"]
    force_pixel = True
    MAX_SHAPE = (1280, 1)

    def __init__(self, pixel1=50e-6, pixel2=8e-3):
        super(MythenV, self).__init__(pixel1=pixel1, pixel2=pixel2)

# In[6]:

# Define all modules as single detectors of class MythenV.
# Each one has a mask defined from dummy-values in the dataset


modules = {}
for name, ds in data.items():
    one_module = MythenV()
    mask = ds[0] < 0
    # discard the first 20 and last 20 pixels as their intensities are
    # less reliable
    mask[:20] = True
    mask[-20:] = True
    one_module.mask = mask.reshape(-1, 1)
    modules[name] = one_module

for k, v in modules.items():
    print(k, v.name)


# In[7]:

# Define a peak-picking function based on the dataset-name and the frame_id:

def peak_picking(module_name, frame_id,
                 threshold=500):
    """Peak-picking base on find_peaks_cwt from scipy plus
    second-order tailor exapention refinement for sub-pixel resolution.

    The half-pixel offset is accounted here, i.e pixel #0 has its center at 0.5

    """
    module = modules[module_name]
    msk = module.mask.ravel()

    spectrum = data[module_name][frame_id]
    guess = find_peaks_cwt(spectrum, numpy.array([20]))

    valid = numpy.logical_and(numpy.logical_not(msk[guess]),
                              spectrum[guess] > threshold)
    guess = numpy.extract(valid, guess)

    # Based on maximum is f'(x) = 0 ~ f'(x0) + (x-x0)*(f''(x0))
    df = numpy.gradient(spectrum)
    d2f = numpy.gradient(df)
    bad = d2f == 0
    d2f[bad] = 1e-10  # prevent devision by zero. Discared later on
    cor = df / d2f
    cor[abs(cor) > 1] = 0
    cor[bad] = 0
    ref = guess - cor[guess] + 0.5  # half a pixel offset
    x = numpy.zeros_like(ref) + 0.5  # half a pixel offset
    return numpy.vstack((ref, x)).T


print(peak_picking(ds_names[0], 93))


# In[8]:

nrj = h5["/LaB6_17keV_att3_1555/CRISTAL/Monochromator/energy"][0]
wl = hc / nrj * 1e-10
print("Energy (keV): ", nrj, "\nWavelength (A): ", wl)

LaB6 = get_calibrant("LaB6")
LaB6.wavelength = wl
print(LaB6)


# In[9]:

# This cell defines the transformation of coordinates for a simple
# goniometer mounted vertically.

trans = ExtendedTransformation(dist_expr="dist",
                               poni1_expr="poni1",
                               poni2_expr="poni2",
                               rot1_expr="rot1",
                               rot2_expr="pi*(offset+scale*angle)/180.",
                               rot3_expr="0.0",
                               wavelength_expr="hc/nrj*1e-10",
                               param_names=["dist", "poni1", "poni2", "rot1", "offset", "scale", "nrj"],  # NOQA
                               pos_names=["angle"],
                               constants={"hc": hc})


# In[10]:

def get_position(idx):
    "Returns the postion of the goniometer for the given frame_id"
    return position[idx]

# Approximate offset for the module #0 at 0°


print("Approximated offset for the first module: ",get_position(36))


# In[11]:

# This interactive plot lets one visualize any spectra acquired by any module

fig, ax = plt.subplots()
line = ax.plot(data[ds_names[0]][250])[0]
ligne = plt.Line2D(xdata=[640, 640], ydata=[-500, 1000],
                   figure=fig, linestyle="--", color='red', axes=ax)
ax.add_line(ligne)
ax.set_title("spectrum")
fig.show()


def update(module_id, frame_id):
    spectrum = data[ds_names[module_id]][frame_id]
    line.set_data(numpy.arange(spectrum.size), spectrum)
    ax.set_title("Module %i, Frame %i" % (module_id, frame_id))

    fig.canvas.draw()


# interactive_plot = widgets.interactive(update,
#                                       module_id=(0, len(data)-1),
#                                       frame_id=(0, data[ds_names[0]].shape[0] - 1))  # NOQA
# display(interactive_plot)


# In[ ]:


# In[12]:

# Work with the first module corresponding to:

name = ds_names[0]
print(name)
ds = data[name]
module = modules[name]

# Use the previous widget to select:
# # the index where the beam-center is in the middle of the module
zero_pos = 36

# # The frame index where the first LaB6 peak enters the right-hand
# # side of the spectrum
peak_zero_start = 74

# # The frame index where this first LaB6 leaves the spectrum or the
# # second LaB6 peak appears:
peak_zero_end = 94

# The frames between peak_zero_start and peak_zero_end will be used to
# calibrate roughly the goniometer and used later for finer peak
# extraction


# In[13]:

param0 = {"dist": 0.72,
          "poni1": 640*50e-6,
          "poni2": 4e-3,
          "rot1": 0,
          "offset": -get_position(zero_pos),
          "scale": 1,
          "nrj": nrj}

# Lock enegy for now and a couple of other parameters
bounds0 = {"nrj": (nrj, nrj),
           "dist": (0.71, 0.73),
           "poni2": (4e-3, 4e-3),
           "rot1": (0, 0),
           "scale": (1, 1)}

gonioref0 = GoniometerRefinement(param0,
                                 get_position,
                                 trans,
                                 detector=module,
                                 wavelength=wl,
                                 bounds=bounds0
                                 )
goniometers = {name:  gonioref0}
print(gonioref0)


# In[14]:

# Extract the frames where only the peak zero from LaB6 is present.

for i in range(peak_zero_start, peak_zero_end):
    cp = ControlPoints(calibrant=LaB6, wavelength=wl)
    peak = peak_picking(name, i)
    if len(peak) != 1:
        continue
    cp.append([peak[0]], ring=0)
    img = ds[i].reshape((-1, 1))  # Images are vertical ... transpose the spectrum  # NOQA
    sg = gonioref0.new_geometry("%s_%04i" % (name, i),
                                image=img,
                                metadata=i,
                                control_points=cp,
                                calibrant=LaB6)
    sg.geometry_refinement.data = numpy.array(cp.getList())

print(gonioref0)
print("Residual error before fit:")
print(gonioref0.chi2())


# In[15]:

# First refinement:
gonioref0.refine2()


# In[16]:

# Here we extract all spectra for peaks, If there are as many peaks as
# expected from the theoritical LaB6. perform the assignment.

# Peaks from LaB6:
tths = LaB6.get_2th()

for i in range(peak_zero_end, ds.shape[0]):
    peak = peak_picking(name, i)
    ai = gonioref0.get_ai(get_position(i))
    tth = ai.array_from_unit(unit="2th_rad", scale=False)
    tth_low = tth[20]
    tth_hi = tth[-20]
    ttmin, ttmax = min(tth_low, tth_hi), max(tth_low, tth_hi)
    valid_peaks = numpy.logical_and(ttmin <= tths, tths < ttmax)
    cnt = valid_peaks.sum()
    if (len(peak) == cnt):
        cp = ControlPoints(calibrant=LaB6, wavelength=wl)
        # revert the order of assignment if needed !!
        if tth_hi < tth_low:
            peak = peak[-1::-1]
        for p, r in zip(peak, numpy.where(valid_peaks)[0]):
            # print(p,r)
            cp.append([p], ring=r)
        img = ds[i].reshape((-1, 1))
        sg = gonioref0.new_geometry("%s_%04i" % (name, i),
                                    image=img,
                                    metadata=i,
                                    control_points=cp,
                                    calibrant=LaB6)
        sg.geometry_refinement.data = numpy.array(cp.getList())
        # print(sg.label, len(sg.geometry_refinement.data))

# print(gonioref0)
print(" Number of peaks found and used for refinement")
print(sum([len(sg.geometry_refinement.data)
           for sg in gonioref0.single_geometries.values()]))
print("Residual error before fitting: ", gonioref0.chi2())


# In[17]:

gonioref0.refine2()


# In[18]:

gonioref0.set_bounds("poni1", -1, 1)
gonioref0.set_bounds("poni2", -1, 1)
gonioref0.set_bounds("rot1", -1, 1)
gonioref0.set_bounds("scale", 0.9, 1.1)
gonioref0.refine2()


# In[19]:

# Perform the azimuthal intgration of all data for the first module:

mg = gonioref0.get_mg(position)
mg.radial_range = (0, 95)
images = [i.reshape(-1, 1) for i in ds]
res_mg = mg.integrate1d(images, 50000)
results = {name: res_mg}
print(results)


# In[20]:

# Plot the integrated pattern vs expected peak positions:

LaB6_new = get_calibrant("LaB6")
LaB6_new.wavelength = hc/gonioref0.param[-1]*1e-10
p = jupyter.plot1d(res_mg, calibrant=LaB6_new)
p.figure.show()


# In[21]:

# Peak profile function based on a bilinear interpolations:

def calc_fwhm(integrate_result, calibrant, tth_min=None, tth_max=None):
    "calculate the tth position and FWHM for each peak"
    delta = integrate_result.intensity[1:] - integrate_result.intensity[:-1]
    maxima = numpy.where(numpy.logical_and(delta[:-1] > 0, delta[1:] < 0))[0]
    minima = numpy.where(numpy.logical_and(delta[:-1] < 0, delta[1:] > 0))[0]
    maxima += 1
    minima += 1
    tth = []
    FWHM = []
    if tth_min is None:
        tth_min = integrate_result.radial[0]
    if tth_max is None:
        tth_max = integrate_result.radial[-1]
    for tth_rad in calibrant.get_2th():
        tth_deg = tth_rad*integrate_result.unit.scale
        if (tth_deg <= tth_min) or (tth_deg >= tth_max):
            continue
        idx_theo = abs(integrate_result.radial-tth_deg).argmin()
        id0_max = abs(maxima-idx_theo).argmin()
        id0_min = abs(minima-idx_theo).argmin()
        I_max = integrate_result.intensity[maxima[id0_max]]
        I_min = integrate_result.intensity[minima[id0_min]]
        tth_maxi = integrate_result.radial[maxima[id0_max]]
        I_thres = (I_max + I_min)/2.0
        if minima[id0_min] > maxima[id0_max]:
            if id0_min == 0:
                min_lo = integrate_result.radial[0]
            else:
                min_lo = integrate_result.radial[minima[id0_min-1]]
            min_hi = integrate_result.radial[minima[id0_min]]
        else:
            if id0_min == len(minima) - 1:
                min_hi = integrate_result.radial[-1]
            else:
                min_hi = integrate_result.radial[minima[id0_min+1]]
            min_lo = integrate_result.radial[minima[id0_min]]

        f = interp1d(integrate_result.radial,
                     integrate_result.intensity-I_thres)
        try:
            tth_lo = bisect(f, min_lo, tth_maxi)
            tth_hi = bisect(f, tth_maxi, min_hi)
        except:
            pass
        else:
            FWHM.append(tth_hi-tth_lo)
            tth.append(tth_deg)
    return tth, FWHM


# In[22]:

# Peak error:


def calc_peak_error(integrate_result, calibrant, tth_min=10, tth_max=95):
    "calculate the tth position and FWHM for each peak"
    peaks = find_peaks_cwt(integrate_result.intensity, numpy.array([10]))
    df = numpy.gradient(integrate_result.intensity)
    d2f = numpy.gradient(df)
    bad = d2f == 0
    d2f[bad] = 1e-10
    cor = df / d2f
    print((abs(cor) > 1).sum())
    cor[abs(cor) > 1] = 0
    cor[bad] = 0
    got = numpy.interp(peaks-cor[peaks],
                       numpy.arange(len(integrate_result.radial)),
                       integrate_result.radial)
    mask = numpy.logical_and(got >= tth_min,
                             got <= tth_max)
    got = got[mask]
    target = numpy.array(calibrant.get_2th())*integrate_result.unit.scale
    mask = numpy.logical_and(target >= tth_min,
                             target <= tth_max)
    target = target[mask]
    print(len(got), len(target))
    d2 = distance_matrix(target.reshape(-1, 1),
                         got.reshape(-1, 1), p=1)

    return target, target-got[d2.argmin(axis=-1)]


# In[23]:

# fig, ax = plt.subplots()
# ax.plot(*calc_fwhm(res_mg, LaB6_new), "o", label="FWHM")
# ax.plot(*calc_peak_error(res_mg, LaB6_new), "o", label="offset")
# ax.set_title("Peak shape & error as function of the angle")
# ax.set_xlabel(res_mg.unit.label)
# ax.legend()
# fig.show()


# ## Module 1
#
# We can apply the same procdure for the second module ... and try to
# rationalize the procedure.

# In[24]:

module_id = 1
name = ds_names[module_id]
ds = data[name]
zero_pos = 64
frame_start = 103
frame_stop = 123


# In[25]:

param1 = {"dist": 0.72,
          "poni1": 640*50e-6,
          "poni2": 4e-3,
          "rot1": 0,
          "offset": -get_position(zero_pos),
          "scale": 1,
          "nrj": nrj}

# Lock enegy for now and a couple of other parameters
bounds1 = {"nrj": (nrj, nrj),
           "dist": (0.7, 0.8),
           "poni2": (4e-3, 4e-3),
           "rot1": (0, 0),
           "scale": (1, 1)}

gonioref1 = GoniometerRefinement(param1,
                                 get_position,
                                 trans,
                                 detector=modules[name],
                                 wavelength=wl,
                                 bounds=bounds1
                                 )
print(gonioref1)
goniometers[name] = gonioref1


# In[26]:

# Exctract frames with peak#0
for i in range(frame_start, frame_stop):
    cp = ControlPoints(calibrant=LaB6, wavelength=wl)
    peak = peak_picking(name, i)
    if len(peak) != 1:
        continue
    cp.append([peak[0]], ring=0)
    img = (ds[i]).reshape((-1,1))
    sg = gonioref1.new_geometry("%s_%04i" % (name, i),
                                image=img,
                                metadata=i,
                                control_points=cp,
                                calibrant=LaB6)
    sg.geometry_refinement.data = numpy.array(cp.getList())

print(gonioref1)
print(gonioref1.chi2())


# In[27]:

gonioref1.refine2()


# In[28]:

# Exctract all frames with peak>0
tths = LaB6.get_2th()
# print(tths)
for i in range(frame_stop, ds.shape[0]):
    frame_name = "%s_%04i" % (name, i)
    if frame_name in gonioref1.single_geometries:
        continue
    peak = peak_picking(name, i)
    ai = gonioref1.get_ai(get_position(i))
    tth = ai.array_from_unit(unit="2th_rad", scale=False)
    tth_low = tth[20]
    tth_hi = tth[-20]
    ttmin, ttmax = min(tth_low, tth_hi), max(tth_low, tth_hi)
    valid_peaks = numpy.logical_and(ttmin <= tths, tths < ttmax)
    cnt = valid_peaks.sum()
    if (len(peak) == cnt) and cnt > 0:
        cp = ControlPoints(calibrant=LaB6, wavelength=wl)
        # revert the order of assignment if needed !!
        if tth_hi < tth_low:
            peak = peak[-1::-1]
        for p, r in zip(peak, numpy.where(valid_peaks)[0]):
            cp.append([p], ring=r)
        img = ds[i].reshape((-1, 1))
        sg = gonioref1.new_geometry(frame_name,
                                    image=img,
                                    metadata=i,
                                    control_points=cp,
                                    calibrant=LaB6)
        sg.geometry_refinement.data = numpy.array(cp.getList())
        # print(frame_name, len(sg.geometry_refinement.data))

print(" Number of peaks found and used for refinement")
print(sum([len(sg.geometry_refinement.data)
           for sg in gonioref1.single_geometries.values()]))
print("Residual error before fitting: ", gonioref1.chi2())


# In[29]:

gonioref1.refine2()
gonioref1.set_bounds("poni1", -1, 1)
gonioref1.set_bounds("poni2", -1, 1)
gonioref1.set_bounds("rot1", -1, 1)
gonioref1.set_bounds("scale", 0.9, 1.1)
gonioref1.refine2()


# In[30]:

mg1 = gonioref1.get_mg(position)
mg1.radial_range = (0, 95)
images = [i.reshape(-1, 1) for i in data[name]]
res_mg1 = mg1.integrate1d(images, 50000)
results[name] = res_mg1


# In[31]:

LaB6_new = get_calibrant("LaB6")
LaB6_new.wavelength = hc/gonioref1.param[-1]*1e-10
p = jupyter.plot1d(res_mg1, calibrant=LaB6_new)
p.figure.show()


# In[32]:

# fig, ax = plt.subplots()
# ax.plot(*calc_fwhm(res_mg1, LaB6_new, 10, 88), "o", label="FWHM")
# ax.plot(*calc_peak_error(res_mg1, LaB6_new, 10, 88), "o", label="error")
# ax.set_title("Peak shape & error as function of the angle")
# ax.set_xlabel(res_mg.unit.label)
# ax.legend()
# fig.show()


# ## All other Modules
#
# We define now an automatic procedure for any module.
# The detection used 3 parameter visually extracted from the Figure1:
#
# * zero_pos: the frame where the beam-stop is in the center of the module
# * frame_start: the frame where the first peak of LaB6 appears (positive)
# * frame_stop: the frame where the second peak of LaB6 appears (positive)
#
# This is enough for boot-strapping the goniometer configuration.

# In[33]:

def add_module(name,
               zero_pos,
               frame_start,
               frame_stop):
    ds = data[name]
    param = {"dist": 0.72,
             "poni1": 640*50e-6,
             "poni2": 4e-3,
             "rot1": 0,
             "offset": -get_position(zero_pos),
             "scale": 1,
             "nrj": nrj}

    # Lock enegy for now and a couple of other parameters
    bounds = {"nrj": (nrj, nrj),
              "dist": (0.7, 0.8),
              "poni2": (4e-3, 4e-3),
              "rot1": (0, 0),
              "scale": (1, 1)}

    gonioref = GoniometerRefinement(param,
                                    get_position,
                                    trans,
                                    detector=modules[name],
                                    wavelength=wl,
                                    bounds=bounds)
    goniometers[name] = gonioref

    for i in range(frame_start, frame_stop):
        cp = ControlPoints(calibrant=LaB6, wavelength=wl)
        peak = peak_picking(name, i)
        if len(peak) != 1:
            continue
        cp.append([peak[0]], ring=0)
        img = (ds[i]).reshape((-1, 1))
        sg = gonioref.new_geometry("%s_%04i" % (name, i),
                                   image=img,
                                   metadata=i,
                                   control_points=cp,
                                   calibrant=LaB6)
        sg.geometry_refinement.data = numpy.array(cp.getList())

    print(gonioref.chi2())
    gonioref.refine2()

    tths = LaB6.get_2th()
    # print(tths)
    for i in range(frame_stop, ds.shape[0]):
        frame_name = "%s_%04i" % (name, i)
        if frame_name in gonioref.single_geometries:
            continue
        peak = peak_picking(name, i)
        ai = gonioref.get_ai(get_position(i))
        tth = ai.array_from_unit(unit="2th_rad", scale=False)
        tth_low = tth[20]
        tth_hi = tth[-20]
        ttmin, ttmax = min(tth_low, tth_hi), max(tth_low, tth_hi)
        valid_peaks = numpy.logical_and(ttmin <= tths, tths < ttmax)
        cnt = valid_peaks.sum()
        if (len(peak) == cnt) and cnt > 0:
            cp = ControlPoints(calibrant=LaB6, wavelength=wl)
            # revert the order of assignment if needed !!
            if tth_hi < tth_low:
                peak = peak[-1::-1]

            for p, r in zip(peak, numpy.where(valid_peaks)[0]):
                cp.append([p], ring=r)
            img = (ds[i]).reshape((-1, 1))
            sg = gonioref.new_geometry(frame_name,
                                       image=img,
                                       metadata=i,
                                       control_points=cp,
                                       calibrant=LaB6)
            sg.geometry_refinement.data = numpy.array(cp.getList())
            # print(frame_name, len(sg.geometry_refinement.data))

    print(" Number of peaks found and used for refinement")
    print(sum([len(sg.geometry_refinement.data)
               for sg in gonioref.single_geometries.values()]))

    gonioref.refine2()
    gonioref.set_bounds("poni1", -1, 1)
    gonioref.set_bounds("poni2", -1, 1)
    gonioref.set_bounds("rot1", -1, 1)
    gonioref.set_bounds("scale", 0.9, 1.1)
    gonioref.refine2()

    mg = gonioref.get_mg(position)
    mg.radial_range = (0, 95)
    images = [i.reshape(-1, 1) for i in ds]
    res_mg = mg.integrate1d(images, 50000)
    results[name] = res_mg

    LaB6_new = get_calibrant("LaB6")
    LaB6_new.wavelength = hc/gonioref.param[-1]*1e-10
    p = jupyter.plot1d(res_mg, calibrant=LaB6_new)
    p.figure.show()

    # fig, ax = plt.subplots()
    # ax.plot(*calc_fwhm(res_mg, LaB6_new), "o", label="FWHM")
    # ax.plot(*calc_peak_error(res_mg, LaB6_new, 10, 89), "o", label="error")
    # ax.set_title("Peak shape & error as function of the angle")
    # ax.set_xlabel(res_mg.unit.label)
    # ax.legend()
    # fig.show()


# In[34]:

add_module(ds_names[2],
           92,
           131,
           151)


# In[35]:

add_module(ds_names[3],
           121,
           159,
           179)


# In[36]:

add_module(ds_names[4],
           150,
           188,
           208)


# In[37]:

add_module(ds_names[5],
           178,
           216,
           236)


# In[38]:

add_module(ds_names[6],
           207,
           245,
           266)


# In[39]:

add_module(ds_names[7],
           236,
           273,
           293)


# In[40]:

add_module(ds_names[8],
           264,
           302,
           322)


# In[41]:

len(goniometers)


# In[42]:

# print all the parameters to be able to compare them visually
goniometers["data_12"] = goniometers["data_11"]
for name in ds_names:
    print(name, *["%8.4e" % i for i in goniometers[name].param])


# ## Use the negative part of the spectum ...
#
# Until now, we used only the data where 2th >0
# For the last modules, this thows away half of the data.
#
# We setup here a way to assign the peaks for the negative part of the
# spectrum.

# In[43]:

def complete_gonio(module_id=None, name=None):
    "Scan missing frames for un-indexed peaks"
    if name is None:
        name = ds_names[module_id]
    gonioref = goniometers[name]
    ds = data[name]
    print("Number of peaks previously found:",
          sum([len(sg.geometry_refinement.data)
               for sg in gonioref.single_geometries.values()]))

    tths = LaB6.get_2th()

    for i in range(ds.shape[0]):
        frame_name = "%s_%04i" % (name, i)
        if frame_name in gonioref.single_geometries:
                continue
        peak = peak_picking(name, i)
        ai = gonioref.get_ai(get_position(i))
        tth = ai.array_from_unit(unit="2th_rad", scale=False)
        tth_low = tth[20]
        tth_hi = tth[-20]
        ttmin, ttmax = min(tth_low, tth_hi), max(tth_low, tth_hi)
        valid_peaks = numpy.logical_and(ttmin <= tths, tths < ttmax)
        cnt = valid_peaks.sum()
        if (len(peak) == cnt) and cnt > 0:
            cp = ControlPoints(calibrant=LaB6, wavelength=wl)
            # revert the order of assignment if needed !!
            if tth_hi < tth_low:
                peak = peak[-1::-1]
            for p, r in zip(peak, numpy.where(valid_peaks)[0]):
                cp.append([p], ring=r)
            img = ds[i].reshape((-1, 1))
            sg = gonioref.new_geometry(frame_name,
                                       image=img,
                                       metadata=i,
                                       control_points=cp,
                                       calibrant=LaB6)
            sg.geometry_refinement.data = numpy.array(cp.getList())
            # print(frame_name, len(sg.geometry_refinement.data))

    print("Number of peaks found after re-scan:",
          sum([len(sg.geometry_refinement.data)
               for sg in gonioref.single_geometries.values()]))
    return gonioref


# In[44]:

gonio8 = complete_gonio(module_id=8)
gonio8.refine2()


# In[45]:

gonio7 = complete_gonio(module_id=7)


# In[46]:

gonio7.refine2()


# In[47]:

gonio6 = complete_gonio(module_id=6)
gonio6.refine2()


# In[48]:

gonio5 = complete_gonio(module_id=5)
gonio5.refine2()


# In[49]:

gonio4 = complete_gonio(module_id=4)
gonio4.refine2()


# In[50]:

gonio3 = complete_gonio(module_id=3)
gonio3.refine2()


# In[51]:

gonio2 = complete_gonio(module_id=2)
gonio2.refine2()


# In[52]:

gonio1 = complete_gonio(module_id=1)
gonio1.refine2()


# In[53]:

gonio0 = complete_gonio(module_id=0)
gonio0.refine2()


# In[54]:

# Rescan module0 which looks much different:
gonio0.single_geometries.clear()
gonio0 = complete_gonio(module_id=0)
gonio0.refine2()


# ## Discard wronly assigned peaks
#
# We have seen previously that some modules have a much higher
# residual error, while all have almost the same number of peaks
# recorded and fitted.
#
# Some frames are contributing much more than all the other in those
# badly-fitted data. Let's spot them and re-assign them

# In[55]:

# search for mis-assigned peaks in module #0
labels = []
errors = []

for lbl, sg in gonio0.single_geometries.items():
    labels.append(lbl)
    errors.append(sg.geometry_refinement.chi2())

s = numpy.argsort(errors)
for i in s[-10:]:
    print(labels[i], errors[i])

# In[56]:

# remove wrongly assigned peak for frame 480
print(gonio0.single_geometries.pop("data_02_0480").control_points)
gonio0.refine2()
gonio0 = complete_gonio(module_id=0)
gonio0.refine2()


# In[57]:

def search_outliers(module_id=None, name=None, threshold=1.2):
    "Search for wrongly assigned peaks"
    if name is None:
        name = ds_names[module_id]
    gonioref = goniometers[name]
    labels = []
    errors = []

    for lbl, sg in gonioref.single_geometries.items():
        labels.append(lbl)
        errors.append(sg.geometry_refinement.chi2())
    s = numpy.argsort(errors)
    last = errors[s[-1]]
    to_remove = []
    for i in s[-1::-1]:
        lbl = labels[i]
        current = errors[i]
        print(lbl, current, last, last/current)
        if threshold * current < last:
            break
        last = current
        to_remove.append(lbl)
    return to_remove


for lbl in search_outliers(8):
    gonio8.single_geometries.pop(lbl)
gonio8.refine2()
gonio8 = complete_gonio(module_id=8)
gonio8.refine2()


# In[58]:

print(gonio7.chi2())
for lbl in search_outliers(7):
    gonio7.single_geometries.pop(lbl)
gonio7.refine2()
gonio7 = complete_gonio(module_id=7)
gonio7.refine2()


# In[59]:

print(gonio0.chi2())
print(len(search_outliers(0)))
# for lbl in search_outliers(7):
#     gonio7.single_geometries.pop(lbl)
# gonio7.refine2()
# gonio7 = complete_gonio(module_id=7)
# gonio7.refine2()


# ## Overlay of the differents results
#
# We are getting to an end. Here are the first actually integrated data

# In[60]:

fig, ax = plt.subplots()
summed, counted, radial = None, None, None

for i in range(9):
    name = ds_names[i]
    ds = data[name]
    gonioref = goniometers[name]
    mg = gonioref.get_mg(position)
    mg.radial_range = (0, 95)
    images = [i.reshape(-1, 1) for i in ds]
    res_mg = mg.integrate1d(images, 50000)
    results[name] = res_mg
    if summed is None:
        summed = res_mg.sum
        counted = res_mg.count
    else:
        summed += res_mg.sum
        counted += res_mg.count
    radial = res_mg.radial
    jupyter.plot1d(res_mg, label="%i %s" % (i, name), calibrant=LaB6, ax=ax)

ax.plot(radial, summed/counted, label="Merged")
ax.legend()
fig.show()


# ## Multi-Gonio fit
#
# Can we fit everything togeather ?
#
# Just assume energy and scale parameter of the goniometer are the
# same for all modules and fit everything.

# In[61]:

class MultiGoniometer:
    def __init__(self, list_of_goniometers,
                 param_name_split,
                 param_name_common):
        self.nb_gonio = len(list_of_goniometers)
        self.goniometers = list_of_goniometers
        self.names_split = param_name_split
        self.names_common = param_name_common
        self.param = None

    def init_param(self):
        param = []
        for gonio in self.goniometers:
            param += list(gonio.param[:len(self.names_split)])
        param += list(gonio.param[len(self.names_split):])
        self.param = numpy.array(param)

    def residu2(self, param):
        "Actually performs the calulation of the average of the error squared"
        sumsquare = 0.0
        for idx, gonio in enumerate(self.goniometers):
            gonio_param = numpy.concatenate((param[len(self.names_split)*idx:len(self.names_split)*(1+idx)],  # NOQA
                                             param[len(self.names_split)*len(self.goniometers):]))  # NOQA
            sumsquare += gonio.residu2(gonio_param)
        return sumsquare

    def chi2(self, param=None):
        """Calculate the average of the square of the error for a given parameter set
        """
        if param is not None:
            return self.residu2(param)
        else:
            if self.param is None:
                self.init_param()
            return self.residu2(self.param)

    def refine2(self, method="slsqp", **options):
        """Geometry refinement tool

        See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html  # NOQA

        :param method: name of the minimizer
        :param options: options for the minimizer
        """
        if method.lower() in ["simplex", "nelder-mead"]:
            method = "Nelder-Mead"

        former_error = self.chi2()
        print("Cost function before refinement: %s" % former_error)
        param = numpy.asarray(self.param, dtype=numpy.float64)
        print(param)
        res = minimize(self.residu2, param, method=method,
                       tol=1e-12,
                       options=options)
        print(res)
        newparam = res.x
        new_error = res.fun
        print("Cost function after refinement: %s" % new_error)

        if new_error < former_error:
            self.param = newparam
        return self.param

    def integrate(self, list_of_dataset, npt=50000, radial_range=(0, 100)):
        summed = None
        counted = None
        param = self.param
        for idx, ds in enumerate(list_of_dataset):
            gonio = self.goniometers[idx]
            gonio_param = numpy.concatenate((param[len(self.names_split)*idx:len(self.names_split)*(1+idx)],  # NOQA
                                             param[len(self.names_split)*len(self.goniometers):]))  # NOQA
            print(gonio_param)
            gonio.param = gonio_param
            mg = gonio.get_mg(position)
            mg.radial_range = radial_range
            images = [i.reshape(-1, 1) for i in ds]
            res_mg = mg.integrate1d(images, 50000)
            if summed is None:
                summed = res_mg.sum
                counted = res_mg.count
            else:
                summed += res_mg.sum
                counted += res_mg.count
            radial = res_mg.radial
        res = Integrate1dResult(radial, summed/numpy.maximum(counted, 1e-10))
        res._set_unit(res_mg.unit)
        res._set_count(counted)
        res._set_sum(summed)
        return res


# In[62]:

multigonio = MultiGoniometer([goniometers[ds_names[i]] for i in range(9)],
                             ["dist", "poni1", "poni2", "rot1", "offset"],
                             ["scale", "nrj"])


# In[63]:


print(multigonio.chi2())

multigonio.param = numpy.array([7.20594053e-01,  3.22408604e-02, 4.05228023e-03, -2.75578440e-05,  # NOQA
                                -8.27999414e+01, 7.20612302e-01, 3.36369797e-02, 4.02094516e-03,  # NOQA
                                -1.74996556e-05, -7.71999791e+01, 7.20636130e-01, 3.47920978e-02,  # NOQA
                                4.01341931e-03, -1.21330600e-05, -7.15999090e+01, 7.20757808e-01,  # NOQA
                                3.33850817e-02, 3.95036100e-03, 3.46517345e-05, -6.57999267e+01,  # NOQA
                                7.20813915e-01, 3.22167822e-02, 3.97128822e-03, 2.00055269e-05,  # NOQA
                                -6.00000525e+01, 7.20881596e-01, 3.33801850e-02, 3.97760147e-03,  # NOQA
                                1.47074593e-05, -5.43998157e+01, 7.21048510e-01, 3.22346939e-02,  # NOQA
                                4.02104962e-03, -1.69519259e-05, -4.85998856e+01, 7.21074630e-01,  # NOQA
                                3.08484557e-02, 4.09385968e-03, -6.91378973e-05, -4.27999030e+01,  # NOQA
                                7.21154891e-01, 3.20619921e-02, 4.24950906e-03, -1.81328256e-04,  # NOQA
                                -3.71999987e+01, 9.99038595e-01, 1.70266104e+01])  # NOQA
print(multigonio.chi2())


# In[64]:

multigonio.refine2()


# In[65]:

LaB6_new = get_calibrant("LaB6")
LaB6_new.wavelength = 1e-10*hc/multigonio.param[-1]
print(LaB6, "\n", LaB6_new)


# In[66]:

res = multigonio.integrate([data[ds_names[i]] for i in range(9)])


# In[67]:

ax = jupyter.plot1d(res, calibrant=LaB6_new)
ax.figure.show()

# In[68]:

# fig, ax = plt.subplots()
# ax.plot(*calc_fwhm(res, LaB6_new, 10, 95), "o", label="FWHM")
# ax.plot(*calc_peak_error(res, LaB6_new, 10, 95), "o", label="error")
# ax.set_title("Peak shape & error as function of the angle")
# ax.set_xlabel(res.unit.label)
# ax.legend()
# fig.show()


# In[69]:

print("total run time: ", time.time()-start_time)


# ## Conclusion The calibration works and the FWHM of every single
# peak is pretty small: 0.02°.  The geometry has been refined with the
# wavelength: The goniometer scale parameter refines to 0.999 instead
# of 1 and the wavelength is fitted with a change at the 5th digit
# which is pretty precise.

# In[103]:

# remove the nan values
intensity = summed/counted
summ = numpy.vstack([radial, intensity/1000.])[:, ~numpy.isnan(intensity)]
numpy.savetxt("summ_1555.dat", summ.T)
numpy.savetxt("calib_1555.cal", multigonio.param)

# In[102]:

print(multigonio.param)
