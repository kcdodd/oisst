
from __future__ import annotations
from jax import (
  numpy as jnp,
  scipy as jsp,
  random)
import jax
from PIL import (
  Image,
  ImageDraw,
  ImageFont)
from time import time
from datetime import datetime, timezone
from functools import partial, wraps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import wave
import scipy as sp
import h5py
from nohm.discrete import (
  pad_shift)
from nohm.core import (
  Floating,
  Static,
  routine,
  Enclose)
from nohm.plot import (
  Figure,
  Line,
  Plot1D,
  Plot2D)

ROOT = Path(__file__).parent
DATA_DIR = ROOT/'oisst_sst_daily'
OUT_DIR = ROOT/'out'

#https://storage.googleapis.com/berkeley-earth-temperature-hr/global/gridded/Global_TAVG_Gridded_1deg.nc
land_file = ROOT/'Global_TAVG_Gridded_1deg.nc'
sst_file = DATA_DIR/"sst_full.h5"
combined_file = DATA_DIR/"combined.h5"

# combined_file.unlink(missing_ok=True)

if not combined_file.exists():
  with (
    h5py.File(land_file, 'r') as land_dset,
    h5py.File(sst_file, 'r') as sst_full,
    h5py.File(combined_file, 'x') as combined_dset):

    sst = sst_full['sst_delta']
    sst_mask = jnp.isfinite(sst[0])
    sst_mask = np.roll(sst_mask, 720, axis=1)

    sst_time = np.array(sst_full['time'])
    sst_mean = np.array(sst_full['sst_mean'])

    land_mask = np.array(land_dset['land_mask'])
    areal_weight = np.array(land_dset['areal_weight'])
    climatology = np.array(land_dset['climatology'])
    print(f"{land_dset.keys()=}")

    temperature = land_dset['temperature']
    shape = temperature.shape[1:]
    time = np.array(land_dset['time'])

    time_mask = (time >= 1981)
    time = time[time_mask]
    temperature = temperature[time_mask]

    land_mask = np.all(np.isfinite(temperature), axis=0)

    nframes = len(time)

    years = np.floor(time)
    months = np.floor(12*(time - years)).astype(np.int32)
    land_time = np.array([
      datetime(int(year), 1+int(month), 15, tzinfo=timezone.utc).timestamp()
      for month, year in zip(months, years, strict=True)],
      np.float64)

    temperature = temperature + climatology[months]
    temperature = np.where(land_mask, temperature, 0.0)

    # for k in range(len(temperature)):
    #   Figure(Plot2D((temperature[k]).T), title=f"{k}:").fig()

    combined_dset['time'] = sst_time
    combined = combined_dset.create_dataset(
      'temperature',
      shape = (len(sst), 720, 1440),
      dtype = np.float16,
      chunks = (240, 45, 45),
      compression="gzip",
      compression_opts=5)

    combined_mean = np.zeros_like(sst[0])

    modes, s, basis = np.linalg.svd(temperature.reshape(nframes, -1), full_matrices=False)
    basis = basis.reshape(-1, *shape)
    print(f"{modes.shape=}, {basis.shape=}")
    print(f"svals: {s[:10]/s[0]}")
    print(f"rank (1e-3): {np.sum(s/s[0] > 1e-3)}")
    modes = modes*s[None,:]
    del s

    # basis = basis[:240]
    # modes = modes[:,:240]

    interp = jax.vmap(jnp.interp, [None, None, 1], 1)
    nbatch = 240

    print(f"  - upsampling {shape} -> {(720,1440)} ")
    u = (np.pi/180)*(jnp.array(land_dset['latitude']) + 90)
    v = (np.pi/180)*jnp.array(land_dset['longitude'])

    _u = np.pi*(0.5 + np.arange(720))/720
    _v = np.arange(1440)*2*np.pi/1440 - np.pi
    combined_dset['lat'] = _u
    combined_dset['lon'] = _v

    _basis = np.stack([
      sp.interpolate.RectSphereBivariateSpline(u, v, basis[k])(_u, _v, grid=True)
      for k in range(len(basis))])

    del basis

    for k in range(0, len(sst), nbatch):
      print(f"- {k/len(sst):.2%}")

      _time = sst_time[k:k+nbatch]
      _sst = sst[k:k+nbatch] + sst_mean[None]
      _sst = np.roll(_sst, 720, axis=2)

      _modes = interp(_time, land_time, modes)
      _land = jnp.einsum('km,mij->kij', _modes, _basis)

      _combined = jnp.where(sst_mask[None], _sst, _land)
      combined[k:k+nbatch] = _combined
      combined_mean += jnp.sum(_combined, axis=0)

      if k == 0:
        Figure(
          Plot2D(_sst[0].T, colorbar=True),
          Plot2D(_combined[0].T, colorbar=True),
          Plot2D(_combined[1].T-_combined[0].T, colorbar=True)).fig()

      del _combined, _modes, _land

    combined_mean /= len(sst)
    combined_dset['mean'] = combined_mean

#===============================================================================
with h5py.File(combined_file, 'r') as combined_dset:
  temperature = combined_dset['temperature']
  mean = combined_dset['mean']

  # for k in range(len(temperature)):
  #   Figure(
  #     Plot2D((temperature[k+1]-temperature[k]).T, colorbar=True)).fig()

  X = np.array(temperature[-4*365:], np.float32)
  shape = X.shape[1:]
  X = X.reshape(len(X), -1).T
  X0 = X[:,:-1]
  X1 = X[:,1:]
  del X
  U, s, Vh = np.linalg.svd(X0, full_matrices=False)

  _s = jnp.where(s/s[0] > 2e-3, 1/s, 0.0)
  print(f"rank: {np.count_nonzero(_s)}")

  S = jnp.einsum('ji,jk,lk', U, X1, Vh*_s[:,None])
  print(f"{S.shape=}, {U.shape=}")
  eigvals, eigvecs = jnp.linalg.eig(S)

  idx = np.argsort(np.abs(eigvals))[::-1]
  eigvals = eigvals[idx]
  eigvecs = eigvecs[:,idx]
  print(f"{eigvals[:10]}")

  eigvecs = jnp.einsum('ij,jk', U, eigvecs)
  eigvecs = eigvecs.transpose(1,0).reshape(-1, *shape)


  for k in range(len(eigvecs)):
    Figure(Plot2D((eigvecs[k]).T), title=f"{k}: {np.abs(eigvals[k]):.3f}").fig()


  # Figure(Plot2D(modes[:,:20])).fig()
  # for k in range(len(basis)):
  #   Figure(Plot2D((basis[k]).T), title=f"{k}: {s[k]/s[0]}").fig()

  # Figure(Plot2D(land_mask.T, colorbar=True), Plot2D(areal_weight.T, colorbar=True)).fig()


  # time_mask = (time >= 1981)&(time < 2025)
  # time = time[time_mask]
  # nframes = len(time)
  # temperature = temperature[time_mask]
  # land_mask = land_mask*np.all(np.isfinite(temperature), axis=0)

  # temperature = climatology[None] + temperature.reshape(-1, 12, *shape)
  # temperature = temperature.reshape(-1, *shape)
  # temperature = land_mask*np.where(land_mask, temperature, 0.0)

  # # for k in range(nframes):
  # #   Figure(Plot2D((temperature[k]).T), title=f"{k}").fig()


  # _temperature = temperature.reshape(nframes, -1)
  # modes, s, basis = np.linalg.svd(_temperature, full_matrices=False)
  # basis = basis.reshape(-1, *shape)
  # print(f"{modes.shape=}, {basis.shape=}")
  # print(f"svals: {s[:10]/s[0]}")
  # print(f"rank (1e-3): {np.sum(s/s[0] > 1e-3)}")

  # # Figure(Plot2D(modes[:,:20])).fig()
  # for k in range(len(basis)):
  #   Figure(Plot2D((basis[k]).T), title=f"{k}: {s[k]/s[0]}").fig()

  # for i, (time, temp) in enumerate(zip(time, temperature)):
  #   temp = np.array(temp)
  #   temp = np.where(land_mask, temp, 0.0)
  #   Figure(Plot2D(temp.T), title=f"{time}").fig()

  # # dynamic mode decomposition
  # _temperature = _temperature.T
  # X0 = _temperature[:,:-1]
  # X1 = _temperature[:,1:]
  # U, s, Vh = np.linalg.svd(X0, full_matrices=False)
  # _s = jnp.where(s > 0.0, s, 0.0)
  # S = jnp.einsum('ji,jk,lk', U, X1, Vh*_s[:,None])
  # print(f"{S.shape=}, {U.shape=}")
  # eigvals, eigvecs = jnp.linalg.eig(S)
  # print(f"{eigvals[:10]}")
  # eigvecs = jnp.einsum('ij,jk', U, eigvecs)
  # eigvecs = eigvecs.transpose(1,0).reshape(-1, *shape)


  # for k in range(nframes):
  #   Figure(Plot2D((eigvecs[k]).T), title=f"{k}: {eigvals[k]}").fig()

exit()
