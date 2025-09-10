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
from datetime import datetime
from functools import partial, wraps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import wave
import scipy as sp
import h5py
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

#===============================================================================
def file_reducer(reduce_fn):
  @wraps(reduce_fn)
  def _wrapper(*args, in_files):
    out_file = OUT_DIR/f"{reduce_fn.__name__}.npy"
    out_filez = OUT_DIR/f"{reduce_fn.__name__}.npz"

    if out_file.exists():
      return np.load(out_file)

    if out_filez.exists():
      return np.load(out_filez)

    def _iter():
      for file in in_files:
        print(f"- {out_file.name} += {file.name}")
        yield np.load(file)

    out = reduce_fn(_iter(), *args)

    if isinstance(out, dict):
      np.savez_compressed(out_filez, **out)
    elif isinstance(out, tuple):
      np.savez_compressed(out_filez, *out)
    else:
      np.save(out_file, out)
    return out

  return _wrapper

#===============================================================================
@file_reducer
def temp_mean(series):
  mean = None

  for i, data in enumerate(series):
    sst = jnp.mean(data['sst'], axis=0)
    mean = sst if mean is None else mean+sst

  return mean/(i+1)

#===============================================================================
@file_reducer
def annual_mean_cycle(series):
  daily: dict[tuple[int,int], Floating[:,:]] = {}
  counts: dict[tuple[int,int], int] = {}


  for i, data in enumerate(series):
    times = [datetime.fromisoformat(str(v)) for v in  data['time']]

    for time, sst in zip(times, data['sst'], strict=True):
      key = (time.month, time.day)
      mean = daily.get(key)

      if mean is None:
        daily[key] = sst
        counts[key] = 1
      else:
        daily[key] = sst + mean
        counts[key] += 1

  daily_means = {
    f"sst_{month}_{day}": daily[month,day]/counts[month,day]
      for month, day in sorted(daily.keys())}

  return daily_means

#===============================================================================
@file_reducer
def delta_temp_mean(series):
  mean = None

  for i, data in enumerate(series):
    sst = jnp.mean(np.diff(data['sst'], axis=0), axis=0)
    mean = sst if mean is None else mean+sst

  return mean/(i+1)

#===============================================================================
@file_reducer
def temp_variance(series, mean):
  mean = None

  for i, data in enumerate(series):
    sst = jnp.mean((data['sst'] - mean)**2, axis=0)
    mean = sst if mean is None else mean+sst

  return mean/(i+1)

#===============================================================================
def animate_sst(nframes, times, sst_delta):
  shape = sst_delta.shape[1:]
  mask = np.isfinite(sst_delta[0]).astype(np.float64)

  if True:
    width = 2560
    height = 1440
    _shape = (height, width)
    _mask = sp.interpolate.RegularGridInterpolator(
      (np.linspace(0,1,shape[0]), np.linspace(0,1,shape[1])),
      mask)(
      np.stack(np.meshgrid(
        np.linspace(0,1,height),
        np.linspace(0,1,width),
        indexing='ij'),
        axis=-1))

    print(f"{_mask.shape=}")
  else:
    width = 1440
    height = 720
    _mask = mask

  fps = 48
  ninterp = 2
  nbatch = 64
  out_dir = OUT_DIR/"animated_delta"
  out_dir.mkdir(exist_ok=True)
  font = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 32)
  font_cb = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 16)

  print(f"Estimated video length: {nframes*ninterp/fps/60:.1f} min")

  # cmap = plt.get_cmap('berlin')
  import colorcet as cc
  # colors = np.concatenate([
  #   cc.cm.CET_L6_r(jnp.arange(100)/99),
  #   cc.cm.CET_L8(jnp.arange(156)/155)],
  #   axis=0)
  NC = 4096
  colors = np.concatenate([
    cc.cm.CET_L6_r(jnp.arange(128)/127),
    cc.cm.CET_L8(jnp.arange(128)/127)],
    axis=0)

  colorbar_data = np.clip(
    255*colors[None,:,:3]*jnp.ones(20)[:,None,None],
    0,
    255).astype(np.uint8)

  cmap = LinearSegmentedColormap.from_list("mycmap", colors, N=NC)


  import av
  out_file = out_dir/f"{out_dir.name}_{fps}_{nframes}_{width}_{height}.mp4"
  out_file.unlink(missing_ok=True)
  container = av.open(out_file, mode="w")
  stream = container.add_stream("libx264", rate=fps)
  stream.width = width
  stream.height = height
  stream.pix_fmt = "yuv420p"

  if mean is None:
    cmap_range = [0, 30]
  else:
    cmap_range = [-10, 10]


  cb = Image.fromarray(colorbar_data, mode='RGB')
  draw = ImageDraw.Draw(cb)
  _width, _ = cb.size
  draw.text((0, 0), f"{cmap_range[0]:+.0f}", (32,)*3, font=font_cb)
  draw.text((_width-30, 0), f"{cmap_range[1]:+.0f}", (32,)*3, font=font_cb)
  colorbar_data = np.asarray(cb)
  # Figure(Plot2D(colorbar_data)).fig()

  scale = 1.0/(cmap_range[1] - cmap_range[0])
  vmin = cmap_range[0]
  barcode = []
  time_init = time()
  times = [datetime.fromtimestamp(t) for t in times]

  last = None

  for frame_start in range(0, nframes, nbatch):
    frame_stop = min(frame_start+nbatch, nframes)

    if frame_start > 0:
      elapsed = time() - time_init
      remaining = (nframes - frame_start)*elapsed/frame_start
      print(f"{frame_start/nframes:.1%}: elapsed {elapsed/3600:.2f} hr, remaining {remaining/3600:.2f} hr")

    _sst = sst_delta[frame_start:frame_stop]
    _sst = np.where(mask, _sst, 0.0)

    if (height, width) != shape:
      # NOTE: currently only can up-sample
      _sst_f = np.fft.fft2(_sst, axes=(1,2), norm='ortho')

      hpad = height - shape[0]
      wpad = width - shape[1]
      # print(f"{shape=}, {height=}, {hpad=}, {width=}, {wpad=}")
      print(f"  - upsampling {shape} + {(hpad, wpad)}-> {(height,width)} ")

      _sst_f = np.fft.fftshift(_sst_f, axes=(1,2))
      _sst_f = np.pad(
        _sst_f, [
          (0,0),
          (hpad//2, hpad - hpad//2),
          (wpad//2, wpad - wpad//2)])
      _sst_f = np.fft.ifftshift(_sst_f, axes=(1,2))

      # before = _sst
      _sst = np.fft.ifft2(_sst_f, axes=(1,2), norm='ortho').real
      # Figure(Plot2D(before[0]), Plot2D(_sst[0])).fig()

    if last is None:
      last = _sst[0]

    for i, frame in enumerate(range(frame_start, frame_stop)):
      time_str = times[frame].strftime('%d %b %Y')
      print(f"- {frame}: {time_str}")
      T0 = last
      T1 = _sst[i]
      last = T1

      for j in range(ninterp):
        print(f"  - {j+1}/{ninterp}")

        s = j/ninterp
        T = (1-s)*T0 + s*T1

        rgb = cmap(scale*(T - vmin))
        rgb = _mask[:,:,None]*rgb
        rgb = rgb[::-1,:,:3]
        rgb = np.clip(rgb*255, 0, 255).astype(np.uint8)
        barcode.append(np.clip(np.mean(rgb, axis=1), 0, 255).astype(np.uint8))

        rgb[-42:-22,300:556] = colorbar_data

        im = Image.fromarray(rgb, mode='RGB')
        draw = ImageDraw.Draw(im)
        width, height = im.size
        draw.text((10, height-50), time_str, (185,)*3, font=font)
        # im.save(fname)

        rgb24 = np.asarray(im)

        frame = av.VideoFrame.from_ndarray(rgb24, format="rgb24")
        for packet in stream.encode(frame):
          container.mux(packet)

  # Flush stream
  for packet in stream.encode():
    container.mux(packet)

  # Close the file
  container.close()

  barcode = np.stack(barcode, axis=1)
  print(f"{barcode.shape=}")
  barcode = np.clip(
    sp.ndimage.zoom(barcode, [1.0, 1440.0/barcode.shape[1], 1.0]),
    0, 255).astype(np.uint8)
  im = Image.fromarray(barcode, mode='RGB')
  im.save(out_file.parent/(out_file.name[:-4]+'.png'))


#===============================================================================
def animate_modes(nframes, times, basis, modes):
  nmodes = modes.shape[1]
  out_dir = OUT_DIR/"animated_modes"
  out_dir.mkdir(exist_ok=True)
  font = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 32)
  font_cb = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 16)

  # cmap = plt.get_cmap('berlin')
  import colorcet as cc
  # colors = np.concatenate([
  #   cc.cm.CET_L6_r(jnp.arange(100)/99),
  #   cc.cm.CET_L8(jnp.arange(156)/155)],
  #   axis=0)
  colors = np.concatenate([
    cc.cm.CET_L6_r(jnp.arange(128)/127),
    cc.cm.CET_L8(jnp.arange(128)/127)],
    axis=0)

  colorbar_data = np.clip(
    255*colors[None,:,:3]*jnp.ones(20)[:,None,None],
    0,
    255).astype(np.uint8)

  cmap = LinearSegmentedColormap.from_list("mycmap", colors)


  import av
  out_file = out_dir/f"{out_dir.name}_{nmodes}.mp4"
  out_file.unlink(missing_ok=True)
  container = av.open(out_file, mode="w")
  stream = container.add_stream("libx264", rate=24)
  stream.width = 1440
  stream.height = 720
  stream.pix_fmt = "yuv420p"

  if mean is None:
    cmap_range = [0, 30]
  else:
    cmap_range = [-15, 15]


  cb = Image.fromarray(colorbar_data, mode='RGB')
  draw = ImageDraw.Draw(cb)
  width, height = cb.size
  draw.text((0, 0), f"{cmap_range[0]:+.0f}", (32,)*3, font=font_cb)
  draw.text((width-30, 0), f"{cmap_range[1]:+.0f}", (32,)*3, font=font_cb)
  colorbar_data = np.asarray(cb)
  # Figure(Plot2D(colorbar_data)).fig()

  scale = 1.0/(cmap_range[1] - cmap_range[0])
  vmin = cmap_range[0]
  barcode = []


  times = [datetime.fromtimestamp(t) for t in times]
  basis = np.roll(basis, 675, axis=2)

  for frame in range(nframes):
    time_str = times[frame].strftime('%d %b %Y')
    sst = jnp.einsum('m,mij', modes[frame], basis)

    print(f"- {time_str}")
    rgb = cmap(scale*(sst - vmin))[::-1,:,:3]
    rgb = np.clip(rgb*255, 0, 255).astype(np.uint8)
    barcode.append(np.clip(np.mean(rgb, axis=1), 0, 255).astype(np.uint8))

    rgb[-42:-22,300:556] = colorbar_data

    im = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(im)
    width, height = im.size
    draw.text((10, height-50), time_str, (185,)*3, font=font)
    # im.save(fname)

    rgb24 = np.asarray(im)

    frame = av.VideoFrame.from_ndarray(rgb24, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)

  # Flush stream
  for packet in stream.encode():
    container.mux(packet)

  # Close the file
  container.close()

  barcode = np.stack(barcode, axis=1)
  print(f"{barcode.shape=}")
  barcode = np.clip(
    sp.ndimage.zoom(barcode, [1.0, 1440.0/barcode.shape[1], 1.0]),
    0, 255).astype(np.uint8)
  im = Image.fromarray(barcode, mode='RGB')
  im.save(out_file.parent/(out_file.name[:-4]+'.png'))

#===============================================================================
def to_blocks(arr):
  arr = arr.reshape(arr.shape[0], 24, 30, 48, 30)
  arr = arr.transpose(0, 1, 3, 2, 4)
  arr = arr.reshape(*arr.shape[:3], -1)
  return arr

#===============================================================================
def gaussian1d(arr, radius: int, axis: int|None = None):
  if radius == 0:
    return arr

  x = jnp.linspace(-3, 3, 2*radius + 1)
  window = jsp.stats.norm.pdf(x)
  window = window/jnp.sum(window)

  if axis is not None:
    s = (None,)*axis + (slice(None),) + (None,)*(arr.ndim-1-axis)
    window = window[s]

  return jsp.signal.convolve(arr, window, mode='same')


#===============================================================================
@routine
def resample_modes(arr, *, nmodes: Static[int]):
  # extract dominant modes using SVD
  U, s, Vh = jnp.linalg.svd(
    arr.reshape(arr.shape[0],-1),
    full_matrices=False)
  U = jnp.einsum('...ij,...j->...ij', U[:,:nmodes], s[:nmodes])
  # reshaping with original time axis as slowest index,
  # stacked modes become higher frequency components
  U = U.ravel()

  # reduce amplitude of outliers
  var = jnp.var(U)
  # print(f"{var=}")
  U = U*jnp.exp(-0.5*U**2/var)
  # print(f">>> {U.shape=}")

  return U

#===============================================================================
def sample_audio(files, mean):
  """ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4
  """

  frame_rate = 24
  samplerate = 44100
  nmodes = 1

  out_file = OUT_DIR/f"temp_deviation_{nmodes:02d}.wav"
  out_file.unlink(missing_ok=True)

  # y, x = jnp.meshgrid(
  #   jnp.linspace(-90, 90, 720),
  #   jnp.linspace(-180, 180, 1440))

  # z = 90
  # r = (x**2 + y**2 + z**2)**0.5
  # delay = r - jnp.amin(r)
  # delay = delay/jnp.amax(delay)

  # A note on the left channel for 1 second.
  # t = np.linspace(0, 1, samplerate)
  # left_channel = 0.5 * np.sin(2 * np.pi * 440.0 * t)

  # # Noise on the right channel.
  # right_channel = np.random.random(size=samplerate)
  _resample_modes = resample_modes.jit

  mask = None
  left = []
  right = []
  num_frames = 0

  last = None

  for file in sorted(files):
    data = np.load(file)
    times = data['time']
    print(f"{times[0]=}")
    # print(f"{time.shape}, {time.dtype}, {time}")

    sst = data['sst']
    sst = jnp.roll(sst, 675, axis=2)

    num_frames += sst.shape[0]

    sst = jnp.where(jnp.isfinite(sst), sst, 0.0)

    if last is None:
      last = sst[:1]

    delta = jnp.diff(sst, axis=0, prepend=last)
    last = sst[-1:]

    print("  - computing left channel")
    _left = _resample_modes(delta[:,:,:480], nmodes=nmodes)
    print("  - computing center channel")
    _mid = _resample_modes(delta[:,:,480:960], nmodes=nmodes)
    print("  - computing right channel")
    _right = _resample_modes(delta[:,:,960:], nmodes=nmodes)

    left.append(np.asarray(_left + 0.5*_mid))
    right.append(np.asarray(_right + 0.5*_mid))

  nsamples = int((num_frames/frame_rate)*samplerate)
  left = jnp.concatenate(left)
  right = jnp.concatenate(right)

  print("Upsampling and filtering")
  radius = 7
  # upsample to desired length
  left = sp.signal.resample(left, nsamples, axis=0)
  # lowpass a little to reducing popping
  left = gaussian1d(left, radius=radius)

  right = sp.signal.resample(right, nsamples, axis=0)
  right = gaussian1d(right, radius=radius)

  audio = np.stack([left, right], axis=-1)

  audio = audio-jnp.amin(audio)
  audio = audio/jnp.amax(audio)

  # Figure(
  #   Plot1D([Line(audio[:,0])])).fig()

  # Convert to (little-endian) 16 bit integers.
  audio = (audio * (2 ** 15 - 1)).astype("<h")

  print(f"Writing sound file: {out_file}")

  with wave.open(str(out_file), "w") as f:
    # 2 Channels.
    f.setnchannels(2)
    # 2 bytes per sample.
    f.setsampwidth(2)
    f.setframerate(samplerate)
    f.writeframes(audio.tobytes())

#===============================================================================
def temporal_covariance():
  full_file = DATA_DIR/"sst_full.h5"
  cov_file = DATA_DIR/"sst_cov.h5"
  cov_file.unlink(missing_ok=True)

  tradius = 48
  twindow = jsp.stats.norm.pdf(jnp.linspace(-3, 3, 2*tradius + 1))
  twindow = twindow/jnp.sum(twindow)

  if not cov_file.exists():
    with h5py.File(full_file, 'r') as sst_full, h5py.File(cov_file, 'x') as sst_cov:
      shape = sst_full['sst_delta'].shape
      print(f"{shape=}")
      mean_dset = sst_cov.create_dataset(
        'mean',
        shape = shape,
        dtype = np.float16,
        chunks = (24, 720, 1440),
        # maxshape = (None, 720, 1440),
        compression="gzip",
        compression_opts=6)

      cov_dset = sst_cov.create_dataset(
        'cov',
        shape = (shape[0], 8, *shape[1:]),
        dtype = np.float16,
        chunks = (24, 8, 720, 1440),
        # maxshape = (None, 8, 720, 1440),
        compression="gzip",
        compression_opts=6)

  with h5py.File(full_file, 'r') as sst_full, h5py.File(cov_file, 'r+') as sst_cov:
    sst_delta = sst_full['sst_delta']
    mean_dset = sst_cov['mean']
    cov_dset = sst_cov['cov']

    nwindows = len(sst_delta)//tradius

    last = jnp.array(sst_delta[:tradius])
    cur = last

    for i in range(nwindows):
      print(f"mean window {i}/{nwindows}:")

      next = jnp.array(
        sst_delta[-tradius:]
        if i == nwindows-1 else
        sst_delta[(i+1)*tradius:(i+2)*tradius])

      sst_window = jnp.concat([last, cur, next])

      mean = jsp.signal.convolve(sst_window, twindow[:,None,None], mode='valid')
      mean_dset[i*tradius:(i+1)*tradius] = mean

      last, cur = cur, next


    last = jnp.array(sst_delta[:tradius])
    cur = last

    mean_last = jnp.array(mean_dset[:tradius])
    mean_cur = mean_last

    for i in range(nwindows):
      print(f"cov window {i}/{nwindows}:")
      next = jnp.array(
        sst_delta[-tradius:]
        if i == nwindows-1 else
        sst_delta[(i+1)*tradius:(i+2)*tradius])

      mean_next = (
        mean_dset[-tradius:]
        if i == nwindows-1 else
        mean_dset[(i+1)*tradius:(i+2)*tradius])

      sst_window = jnp.concat([last, cur, next])
      mean_window = jnp.concat([mean_last, mean_cur, mean_next])

      cov = jsp.signal.convolve(
        (sst_window - mean_window)**2,
        twindow[:,None,None],
        mode='valid')

      cov_dset[i*tradius:(i+1)*tradius, 0] = cov

      last, cur = cur, next
      mean_last, mean_cur = mean_cur, mean_next

      # if i == 0:
      #   Figure(Plot2D(cov[22], colorbar=True)).fig()


#===============================================================================
files = sorted(list(file for file in DATA_DIR.glob('*.npz') if not file.name.startswith('tmp')))
# mean_file = DATA_DIR/'mean.npy'
# var_file = DATA_DIR/'variance.npy'

mean = temp_mean(in_files=files)

full_file = DATA_DIR/"sst_full.h5"

quantized_file = DATA_DIR/"sst_quantized.npz"

if not full_file.exists():

  with h5py.File(full_file, 'x') as sst_full:
    sst_full.create_dataset('sst_mean', data = mean)

    sst_dset = sst_full.create_dataset(
      'sst_delta',
      shape = (0, 720, 1440),
      dtype = np.float16,
      chunks = (240, 45, 45),
      maxshape = (None, 720, 1440),
      compression="gzip",
      compression_opts=7)

    time_dset = sst_full.create_dataset(
      'time',
      shape = (0,),
      dtype = np.float64,
      chunks = (24,),
      maxshape = (None,))

    for file in files:
      print(f"- storing << {file.name}")
      data = np.load(file)

      sst = data['sst'] - mean

      times = np.array([
        datetime.fromisoformat(str(v)).timestamp()
        for v in  data['time']],
        dtype=np.float64)

      offset = sst_dset.shape[0]
      n = sst.shape[0]
      assert n == len(times)

      sst_dset.resize((offset+n, *sst.shape[1:]))
      time_dset.resize((offset+n,))

      sst_dset[offset:offset+n] = sst
      time_dset[offset:offset+n] = times

with h5py.File(full_file, 'r') as sst_full:
  times = np.array(sst_full['time'])

block_size = 45
block_shape = (720//block_size, 1440//block_size)

# number of spatial modes to project for frequency generation
# this should be multiple of 2, every two modes approximate conjugate eigenvalues
nmodes = 512
assert nmodes%2 == 0

nmodes_global = 8

# oversample during randomized SVD, nmodes is closer to the true rank
nextra = 8
xmodes = nmodes + nextra

nbatch = 256

# modes_file = DATA_DIR/f"sst_diff_modes_{nmodes:03d}.npz"
modes_file = OUT_DIR/f"global_modes_{nmodes_global:03d}.h5"

block_modes_file = OUT_DIR/(f'block_{block_size:d}_' + modes_file.name)
# modes_file.unlink(missing_ok=True)

diffed = modes_file.name.startswith('sst_diff')

if not modes_file.exists():
  # approximate, random SVD
  with h5py.File(full_file, 'r') as sst_full, h5py.File(modes_file, 'x') as sst_modes:
    sst_delta = sst_full['sst_delta']
    nframes = len(sst_delta)

    sst_modes['time'] = np.asarray(sst_full['time'])
    sst_modes['mask'] = np.isfinite(sst_delta[0])

    modes = sst_modes.create_dataset(
      'modes',
      shape = (nframes, nmodes_global),
      dtype = np.float16,
      compression="gzip",
      compression_opts=7)

    basis = sst_modes.create_dataset(
      'basis',
      shape = (nmodes_global, 720, 1440),
      dtype = np.float16,
      compression="gzip",
      compression_opts=7)

    errors = sst_modes.create_dataset(
      'errors',
      shape = (nframes,),
      dtype = np.float16,
      compression="gzip",
      compression_opts=7)

    Y = sst_modes.create_dataset(
      'Y',
      shape = (nframes, nmodes_global),
      dtype = np.float64)


    mask = ~jnp.isfinite(sst_delta[0])
    valid = Enclose(jnp.where, mask, 0.0)

    def _get_batch(k, last):
      if not diffed:
        return valid(sst_delta[k:k+nbatch]), None

      _batch = valid(sst_delta[k:k+nbatch])
      batch = jnp.diff(_batch, prepend=last, axis=0)
      last = _batch[-1:]
      return batch, last


    Z = random.normal(random.key(123), (nmodes_global, 720, 1440))

    last = valid(sst_delta[:1])

    for k in range(0, nframes, nbatch):
      print(f"Y <- A Z: {k}/{nframes}")
      batch, last = _get_batch(k, last)
      Y[k:k+nbatch] = jnp.einsum('kij,mij->km', batch, Z)
      del batch

    del Z

    print("Q R <- Y")
    # Q -> (nframes, nmodes)
    Q, R = jnp.linalg.qr(jnp.asarray(Y))
    print(f"{Y.shape=} -> {Q.shape=}")
    del Y, R, sst_modes['Y']

    # Figure(Plot2D(Q, title='Q')).fig()

    B = jnp.zeros((nmodes_global, 720, 1440), Q.dtype)
    last = valid(sst_delta[:1])

    for k in range(0, nframes, nbatch):
      print(f"B <- Q^T A: {k}/{nframes}")
      batch, last = _get_batch(k, last)
      B = B + jnp.einsum('km,kij->mij', Q[k:k+nbatch], batch)
      del batch

    print("_U s Vh <- B")
    _U, s, _basis = jnp.linalg.svd(B.reshape(nmodes_global, -1), full_matrices=False)
    del B
    _basis = _basis.reshape(nmodes_global, 720, 1440)

    _U = _U*s[None,:]
    print("U <- Q _U")
    _modes = jnp.einsum('ki,im -> km', Q, _U)
    del _U, s

    modes[:] = _modes
    basis[:] = _basis

    del _modes, _basis


    # Figure(Plot2D((basis[0] + 1j*basis[1]).T), Plot2D((basis[2] + 1j*basis[3]).T), title='basis').fig()
    # Figure(Plot2D(modes, title='modes')).fig()

#===============================================================================
# with h5py.File(full_file, 'r') as sst_full, h5py.File(modes_file, 'r') as sst_modes:
#   sst = sst_full['sst_delta']
#   basis = jnp.asarray(sst_modes['basis'], np.float64)
#   modes = sst_modes['modes']

#   for i in range(len(modes)):
#     approx = jnp.einsum('m,mij->ij', jnp.asarray(modes[i], np.float64), basis)
#     Figure(Plot2D(approx.T, colorbar=True), Plot2D((approx-sst[i]).T, colorbar=True)).fig()


#===============================================================================
with h5py.File(modes_file, 'r') as sst_modes:
  basis = sst_modes['basis']

  # for k in range(0, len(basis), 8):
  #   Figure([[
  #     Plot2D((basis[k] + 1j*basis[k+1]).T, title=f'basis {k}-{k+1}'),
  #     Plot2D((basis[k+2] + 1j*basis[k+3]).T, title=f'basis {k+2}-{k+3}')],[
  #     Plot2D((basis[k+4] + 1j*basis[k+5]).T, title=f'basis {k+4}-{k+5}'),
  #     Plot2D((basis[k+6] + 1j*basis[k+7]).T, title=f'basis {k+6}-{k+7}')]]).fig()

  for k in range(0, len(basis), 4):
    Figure([[
      Plot2D((basis[k]).T, title=f'basis {k}'),
      Plot2D((basis[k+1]).T, title=f'basis {k+1}')],[
      Plot2D((basis[k+2]).T, title=f'basis {k+2}'),
      Plot2D((basis[k+3]).T, title=f'basis {k+3}')]]).fig()


  Figure(Plot2D(sst_modes['modes'], title='modes')).fig()

exit()

# #===============================================================================
# with h5py.File(modes_file, 'r') as sst_modes:
#   basis = np.where(sst_modes['mask'], sst_modes['basis'], np.nan)
#   nframes = len(sst_modes['modes'])
#   nframes = 3*365
#   animate_modes(nframes, sst_modes['time'], basis, sst_modes['modes'])

#===============================================================================
# with h5py.File(full_file, 'r') as sst_full:
#   nframes = len(sst_full['time'])
#   # nframes = 24
#   animate_sst(nframes, sst_full['time'], sst_full['sst_delta'])

# exit()

#===============================================================================
# if not block_modes_file.exists():
#   with h5py.File(full_file, 'r') as sst_full:
#     sst_delta = sst_full['sst_delta']
#     nframes = len(sst_delta)
#     nbatch = 256

#     mask = ~jnp.isfinite(sst_delta[0])
#     valid = Enclose(jnp.where, mask, 0.0)

#     def _get_batch(k, last):
#       if not diffed:
#         return valid(sst_delta[k:k+nbatch]), None

#       _batch = valid(sst_delta[k:k+nbatch])
#       batch = jnp.diff(_batch, prepend=last, axis=0)
#       last = _batch[-1:]
#       return batch, last

#     # -> (16, 32, nmodes, 45, 45)
#     block_basis = basis.reshape(nmodes, 16, 45, 32, 45).transpose(1,3,0,2,4)
#     del basis

#     # (nframes, 16, 32, nmodes)
#     block_modes = []
#     last = valid(sst_delta[:1])

#     for k in range(0, nframes, nbatch):
#       print(f"W <- V^T A: {k}/{nframes}")
#       batch, last = _get_batch(k, last)

#       # -> (nframes, 16, 32, 45, 45)
#       batch = batch.reshape(len(batch), 16, 45, 32, 45).transpose(0,1,3,2,4)
#       _modes = jnp.einsum('...mij,k...ij->k...m', block_basis, batch).astype(jnp.float16)
#       block_modes.append(np.asarray(_modes))
#       del batch, _modes
#       # print(f"  {sum(b.nbytes for b in block_modes)/1e9}, {block_modes[-1].shape=}")

#     block_modes = np.concatenate(block_modes, axis=0)
#     # -> (16, 32, nframes, nmodes)
#     block_modes = block_modes.transpose(1,2,0,3)

#     np.savez(
#       block_modes_file,
#       block_modes = block_modes,
#       block_basis = block_basis)
# else:
#   data = np.load(block_modes_file)
#   block_modes = data['block_modes']
#   block_basis = data['block_basis']
#   del data

# Figure(Plot2D(block_modes[5,16], title='modes')).fig()

if not block_modes_file.exists():
  with h5py.File(full_file, 'r') as sst_full, h5py.File(block_modes_file, 'x') as sst_modes:
    sst_delta = sst_full['sst_delta']
    nframes = len(sst_delta)
    modes_shape = (16, 32, nframes, nmodes)
    basis_shape = (16, 32, nmodes, 45, 45)

    modes_dset = sst_modes.create_dataset(
      'modes',
      shape = modes_shape,
      dtype = np.float16,
      compression="gzip",
      compression_opts=7)

    basis_dset = sst_modes.create_dataset(
      'basis',
      shape = basis_shape,
      dtype = np.float32,
      compression="gzip",
      compression_opts=7)

    for i in range(block_shape[0]):
      for j in range(block_shape[1]):
        print(f"block: {i}, {j}")

        idx = (
          slice(None),
          slice(i*block_size, (i+1)*block_size),
          slice(j*block_size, (j+1)*block_size))

        block = jnp.asarray(sst_delta[idx])
        mask = jnp.isfinite(block)
        block = jnp.where(mask, block, 0.0)

        if not jnp.any(mask):
          print("- skipping empty block")
          del block, mask
          continue

        if diffed:
          # compute spatial modes of the first derivative
          block = jnp.diff(block, axis=0)

        block = block.reshape(len(block),-1).astype(jnp.float64)

        # extract dominant spatial modes over all times
        U, s, Vh = jnp.linalg.svd(block, full_matrices=False)
        basis = Vh[:nmodes,:]
        modes = U[:,:nmodes]*s[None,:nmodes]
        err_rms = jnp.mean((jnp.einsum('ij,jk', modes, basis) - block)**2, axis=1)**0.5
        print(f"  - error: {jnp.mean(err_rms):.2e}")

        modes_dset[i,j] = modes
        basis_dset[i,j] = basis.reshape(nmodes, 45, 45)
        del U, s, Vh, block, mask

#-------------------------------------------------------------------------------
# with h5py.File(modes_file, 'r') as sst_modes:
#   basis = jnp.asarray(sst_modes['sst_basis'])

# freq_sq = jnp.sum(jnp.stack(jnp.meshgrid(
#   jnp.fft.fftfreq(45),
#   jnp.fft.fftfreq(45),
#   indexing='ij'))**2,
#   axis=0)[None,None,None]

# _basis = jnp.fft.fft2(basis, axes=(3,4))
# del basis
# _basis = _basis.conj()*_basis

# norm = jnp.sum(_basis, axis=(3,4))
# freq_sq_mean = jnp.where(
#   norm > 0.0,
#   jnp.sum(freq_sq*_basis, axis=(3,4))/norm,
#   0.0)

# del _basis, norm

# idx = jnp.argsort(freq_sq_mean, axis=0)
# del freq_sq_mean


# step when inserting frequency modes, skipped frequencies are zero amplitude
freq_step = 1
# hertz, exponential decay rate for frequency equilization
freq_decay = 1000
freq_min = 15

frame_rate = 24
samplerate = 44100
# nmodes = 128

with h5py.File(block_modes_file, 'r') as sst_modes:
  modes = sst_modes['modes']
  nframes = modes.shape[2]
  nsamples = int((nframes/frame_rate)*samplerate)
  frame_samples = int(samplerate/frame_rate)
  frame_rate = samplerate/frame_samples
  # print(f">> {frame_rate=}")
  # print(f"{frame_samples=}")
  # print(f"{frame_samples*nframes/nsamples=}")

  left = None
  right = None

  # for i in range(block_shape[0]):
  for i in [5]:
    for j in range(block_shape[1]):
    # for j in [15]:
      left_weight = 0.5*(1.0 + np.cos(np.pi*j/32))

      print(f"block: {i}, {j}: left = {left_weight:.2f}, right = {1-left_weight:.2f}")

      # (nframes, nmodes)
      q = jnp.asarray(modes[i,j,:,:nmodes], jnp.float64)
      q = q.reshape(len(q), -1, 2).view(jnp.complex128)[:,:,0]

      tframe = (1/frame_rate)*jnp.arange(nframes)
      # frequency within each frame, enough for twice the frame size for interpolation
      freqs = jnp.pi*samplerate*jnp.fft.fftfreq(2*frame_samples)
      # match carrier phase at beginning of each frame
      frame_phase = jnp.exp(1j*freqs[None,:]*tframe[:,None])

      # compute equilization weights
      f = jnp.abs(freqs)/(jnp.pi*freq_decay)
      # equilizer = jnp.exp(-f**2/(f + 0.5))
      equilizer = 1.0/(1 + f**2)
      equilizer_norm = jnp.sum(equilizer)
      # Figure(Plot1D([Line(equilizer)])).fig()

      frame_spectrum = jnp.zeros_like(frame_phase)
      # set frequency amplitude modulation from projected spatial modes
      frame_spectrum = frame_spectrum.at[:,1:1+freq_step*q.shape[1]:freq_step].set(q)
      frame_spectrum = frame_spectrum.at[:,-freq_step*q.shape[1]::freq_step].set(q[:,::-1].conj())
      # frame_spectrum = frame_spectrum.at[:,20].set(1.0)
      # frame_spectrum = frame_spectrum.at[:,-20].set(1.0)
      # frame_spectrum = frame_spectrum + ddf_mean[:,None]
      # Figure(Plot2D(frame_spectrum[:20])).fig()
      frame_spectrum = equilizer[None,:]*frame_phase*frame_spectrum

      if freq_min > 1:
        frame_spectrum = frame_spectrum.at[:,:freq_min].set(0.0)
        frame_spectrum = frame_spectrum.at[:,-(freq_min-1):].set(0.0)


      signal = jnp.fft.ifft(frame_spectrum, axis=1).real
      # Interpolate a beginning of each frame to minimize discontinuity
      a = signal[:,:frame_samples].ravel()
      b = jnp.roll(signal[:,frame_samples:].ravel(), frame_samples)
      # gaussian interpolation
      w = jnp.tile(
        jnp.exp(-jnp.linspace(0, 10, frame_samples)**2),
        nframes)

      combined = (1.0-w)*a + w*b

      # Figure(
      #   Plot1D([Line(w[:4000], label='w')]),
      #   Plot1D([
      #     Line(a[:4000], label='a'),
      #     Line(b[:4000], label='b'),
      #     Line(combined[:4000], label='left')])).fig()

      # accumulate to left or right channel weighted by block location
      if left is None:
        left = left_weight*combined
        right = (1.0-left_weight)*combined
      else:
        left = left + left_weight*combined
        right = right + (1.0-left_weight)*combined

audio = jnp.stack([left, right], axis=-1)

m = jnp.sin(0.5*jnp.pi*jnp.linspace(0,1,samplerate))**2
audio = audio.at[:samplerate].mul(m[:,None])
audio = audio.at[-samplerate:].mul(m[::-1,None])

# audio = audio-jnp.mean(audio)
# audio = audio/jnp.std(audio)

# # left = audio[:,0]
# # print(f"{jnp.amin(jnp.abs(left))=}")
# # Figure(Plot1D([Line(jnp.log10(jnp.maximum(1e-3, jnp.abs(left))))])).fig()

# sigma = 1
# w = jnp.exp(-jnp.maximum(0.0, jnp.abs(audio)-sigma)**2)
# audio = w*audio
# audio = jnp.clip(w*audio, -sigma, sigma)
# audio = (audio + sigma)/(2*sigma)
audio = audio - jnp.amin(audio)
audio = 0.9*audio/jnp.amax(audio)

# Figure(
#   Plot1D([Line(audio[:,0])])).fig()


# Convert to (little-endian) 16 bit integers.
audio = (audio * (2 ** 15 - 1)).astype("<h")

out_file = OUT_DIR/f"modes-{nmodes:02d}.wav"
print(f"Writing sound file: {out_file}")

with wave.open(str(out_file), "w") as f:
  # 2 Channels.
  f.setnchannels(2)
  # 2 bytes per sample.
  f.setsampwidth(2)
  f.setframerate(samplerate)
  f.writeframes(audio.tobytes())
