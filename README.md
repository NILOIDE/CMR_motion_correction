# CMR Motion Correction

Inter-slice motion estimation and correction for cardiac MRI (UK Biobank cine SAX/LAX
acquisitions), plus a 3D viewer for inspecting alignment results.

Different cine acquisitions (each short-axis slice, each long-axis view) are breath-held
independently, so they commonly disagree about where the heart actually is in scanner space.
This pipeline estimates a rigid pose correction per slice/view by comparing image intensities
along the 3D lines where pairs of planes intersect, then writes out motion-corrected NIfTIs.

## Pipeline

```
download_ukbb.py  /  download_oasis.py   -> raw per-subject NIfTIs (SSH from a remote data store)
            |
            v
prepare_ukbb_slices.py                    -> single-slice layout expected by motion_gradient.py
            |
            v
motion_gradient.py                        -> estimates per-plane rigid poses, writes corrected NIfTIs (+ optional metrics/figures)
            |
            v
slice_viewer_3d.py                        -> interactive 3D viewer over the optimization history
```

### `download_ukbb.py` / `download_oasis.py`

Download subjects from a remote dataset over a single SSH connection (tar-streamed, so you're
only prompted for a password/key once). `download_ukbb.py` pulls the UK Biobank cardiac
segmentations dataset in sorted batches and tracks already-downloaded subjects
(`.downloaded_subjects.txt`) so reruns only fetch new ones; `download_oasis.py` pulls a fixed
number of train/val/test subjects from an OASIS-style split layout.

```
python download_ukbb.py --host user@server --n-subjects 50
python download_oasis.py --host user@server --n-train 10 --n-val 2 --n-test 2
```

Run with `--help` for all options (remote/local directories, batch size, `--refresh-list`,
`--flatten`, ...).

### `prepare_ukbb_slices.py`

Converts the downloaded per-subject layout (SAX stack, LAX_2Ch/3Ch/4Ch, nnU-Net segmentations)
into the single-slice-per-file layout `motion_gradient.py` expects: every SAX slice becomes its
own NIfTI with its own affine (origin shifted along the stack normal), LAX views are copied
unchanged, and segmentation labels are rounded to clean integers. Wrapper folders (e.g.
`Instance_2`) are detected and flattened automatically.

```
python prepare_ukbb_slices.py --src <downloaded_dir> --dst <slices_dir> --workers 8
```

See the module docstring for the exact source/destination layout and `--help` for all options
(`--max-subjects`, `--overwrite`, ...).

### `motion_gradient.py`

For every pair of non-parallel planes (SAX-LAX and LAX-LAX), samples intensity profiles along
their 3D intersection line (restricted to a dilated heart mask from the LAX segmentations) and
optimizes a 6-DOF rigid pose per plane (PyTorch, Adam/SGD) to maximize agreement across cardiac
phases. Optional SAX-stack regularizers keep neighbouring slices from tilting, drifting apart, or
crossing. Always writes motion-corrected NIfTIs in the input layout; with `--logging`, also writes
per-subject and dataset-level metrics (CSV) and figures (PNG) under the run directory.

```
python motion_gradient.py --root <slices_dir> --out-root <alignment_dir> --max-subjects -1 --logging true
```

This script has a large number of tuning options (optimizer/schedule, blur coarse-to-fine,
stack regularizers, similarity metric, point-spread-function model, mask annealing, ...) — see
the module docstring for the full method description and `--help` for every flag.

### `slice_viewer_3d.py`

Interactive PyVista viewer that renders every SAX slice and LAX view as a textured quad in
scanner/world space, using the poses from a `motion_gradient.py` run's optimization history
(`metrics/history.csv`). A slider scrubs through optimization steps; segmentation overlays,
per-plane similarity-metric coloring, intersection lines and mask contours can be toggled.

```
python slice_viewer_3d.py                      # most recently created run, its first subject
python slice_viewer_3d.py 1000071               # a specific subject
python slice_viewer_3d.py 1000071 --run <run-name>
```

See the module docstring for the full keyboard/mouse interaction reference.

## Dependencies

- `numpy`, `pandas`, `scipy`
- `nibabel` (NIfTI I/O)
- `torch` (pose optimization)
- `pyvista` (3D viewer)
- `matplotlib` (figures, `motion_gradient.py --logging`)
- `paramiko` (`download_ukbb.py`)

```
pip install numpy pandas scipy nibabel torch pyvista matplotlib paramiko
```
