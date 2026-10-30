# CMR Motion Correction

Inter-slice motion estimation and correction for cardiac cine MRI (UK Biobank SAX/LAX acquisitions, or any set of
cine slices and stacks as NIfTI or DICOM), plus a 3D viewer for inspecting alignment results.

Different cine acquisitions (each short-axis slice, each long-axis view or stack) are breath-held independently, so
they commonly disagree about where the heart actually is in scanner space. This pipeline estimates a rigid pose
correction per slice by comparing image intensities along the 3D lines where pairs of planes intersect, then writes
out the motion-corrected images in the format they were read in.

These the motion corrected slices (as well as the intermediate optimization process) can be visualized using the
provided visualization tool.
![Optimization process in the 3D viewer (slice_viewer_3d.py)](media/optimization_sa09.gif)

#### Note: This project was vibe-coded within a couple of days. All praise Claude Code.

-------------------------


## Pipeline

```
<root>/<subject>/...                      -> your data, one folder per subject (NIfTI or DICOM; layouts below)
            |
            v
moco_nifti.py  /  moco_dicom.py           -> entry points: load a subject (NIfTI / DICOM), hand its slices to the
            |                                motion correction, write the corrected images in the same format
            v
motion_gradient.py                        -> the motion correction (library): estimates per-slice rigid poses
            |                                (+ optional metrics/figures)
            v
slice_viewer_3d.py  /  slice_viewer_dicom.py   -> interactive 3D viewers over the optimization history
                                                 (UKBB NIfTI runs / DICOM runs with hundreds of slices)
```

### `moco_nifti.py` (UK Biobank NIfTI)

Expects UK Biobank cine SAX/LAX acquisitions as NIfTI, one folder per subject, used as is: no preparation step.
Each subject folder is searched recursively (wrapper folders such as `Instance_2` don't matter), and files are
classified by name:

```
<root>/
  <subject>/
    Instance_2/                    # any folder structure
      SAX.nii.gz                   # SAX stack (X, Y, Z, T), and/or single-slice files SAX_slice<nn>.nii.gz
      LAX_2Ch.nii.gz               # LAX views (X, Y, 1, T); any other NIfTI is a LAX image of its own
      LAX_3Ch.nii.gz
      LAX_4Ch.nii.gz
      nnUNet_segs/                 # optional; matched to their image by name (SAX slices by position)
        SAX_nnUNetSeg.nii.gz
        LAX_2Ch_nnUNetSeg.nii.gz
        ...
```

Missing images or segmentations are allowed. The nnU-Net segmentations are expected to be the ones produced by the
[Ugurlu et al. pipeline](https://github.com/CEMRG-publications/Ugurlu_2025_PLoS_One) [1]. They define the dilated
heart masks and the Dice / Hausdorff metrics. NIfTI headers only store the slice spacing, so the thickness is set with
`--sax-thickness` (8 mm) / `--lax-thickness` (6 mm).

Output, per subject: `SAX_slice<nn>.nii.gz`, `LAX_2Ch.nii.gz`, ... with the corrected affines, and the segmentations
in `nnUNet_segs/`. `--image-dtype uint8` (default `uint16`) saves ~30% disk space.

```
python moco_nifti.py --root <ukbb_dir> --out-root <alignment_dir>
```

### `moco_dicom.py` (DICOM)

For other cine data, e.g. several stacks in different orientations. Every DICOM series (files sharing a
`SeriesInstanceUID`) below a subject folder is one image, named after its folder. Its slices are the distinct
positions/orientations and its cardiac frames the files at one slice (ordered by `TriggerTime` / `InstanceNumber`).
The slice thickness comes from `SliceThickness`. Real/imaginary image pairs are combined into the magnitude. DICOM
data has no segmentations, so the lines are masked by a disc around the centre of each stack instead
(`--center-mask-radius-mm`, default 40 mm), grown and annealed like the heart masks (radius 120 mm at the start, 60 mm
at the end with the default dilation); `--center-mask-radius-mm 0` uses the whole image overlap.

```
<root>/
  <subject>/
    <series_1>/phase_01/*.dcm ... phase_25/*.dcm      # any folder layout
    <series_2>/...
```

Output: every input file rewritten at the same relative path, with the corrected `ImagePositionPatient` /
`ImageOrientationPatient` and new UIDs; pixel data are unchanged (`--image-dtype uint8` for 8-bit). With `--logging`,
each subject's `metrics/` also gets line plots of the metrics over the optimization: `metric_trends.png` (the optimized
objective, then NCC / LNCC / NMI per slice: mean, min / max and inter-quartile range, and the slope over the last 25% of
the steps) and `metric_trends_stacks.png` (the same per stack), to see whether a run improved, is still improving, or
diverged.

```
python moco_dicom.py --root <dicom_dir>                     # output: <dicom_dir>_aligned
python moco_dicom.py --root <dicom_dir> --pair-keep 0.25 --sample-dtype float16   # many slices, small GPU
```

Every option is defined with a general default in `motion_gradient.build_parser`. An entry point adds its
data-format options (slice thickness, `--image-dtype`) and repeats the options worth seeing at a glance (paths, run
options, hardware, ...) with its own defaults: a repeated option replaces the general one (e.g. `--blur-frac` is 0.9 in
`motion_gradient.py` and 0.1 in `moco_dicom.py`). See `--help` for everything. A new data format only needs a new entry
script that loads a subject into a list of 2D+time slices (`motion_gradient.SliceImage`: cine, affine, thickness,
optional segmentation) and writes them back with the corrected affines.

### `motion_gradient.py`

The motion correction, used by the entry points (not run on its own).

Every two slices whose acquired normals differ by at least `--min-pair-angle-deg` (20°) are paired. Slices of one
stack never pair. Between densely spaced stacks the pairs are thinned to partners about a slice thickness apart
(`--pair-spacing-mm`). For each pair it samples intensity profiles along their 3D intersection line (with a PSF band
for the slice thickness, restricted to a dilated heart mask where segmentations exist) and optimizes a rigid pose per
slice (PyTorch, SGD/Adam) to maximize agreement across cardiac phases. Optional SAX-stack regularizers keep
neighbouring slices from tilting, drifting apart, or crossing. The crossing terms (`--sax-sax-weight`: image
dissimilarity along the line where a slice crosses the one above / below it; `--sax-gap-reg`: a gap barrier) act on
every stack, so also on the oblique DICOM stacks (exposed in `moco_dicom.py`). With `--logging`, it also writes per-subject and
dataset-level metrics (CSV) and figures (PNG) under the run directory. The method builds on the intersection-line
intensity-matching approach of [Stolt-Ansó et al.](https://arxiv.org/pdf/2404.00767) [2].

We found best results with plain SGD (the default, no momentum) rather than Adam. There is also a substantial amount
of non-rigid deformation in the torso between acquisitions (the heart itself moves close to rigidly, but the
surrounding anatomy does not), so the intersection-line optimization is restricted to a dilated foreground (heart)
mask derived from the LAX segmentations, rather than the full extent of each line, to keep it from being pulled off
by non-rigid motion further out.

The pose model (`--dof`, default 6) depends on which images a subject has. A segmented LAX view counts only if its
segmentation contains the heart; an unsegmented image takes part unmasked.

| Available images                                                   | Optimized with `--dof 6`                 |
|--------------------------------------------------------------------|------------------------------------------|
| 3 or more, at least one a stack (SAX + 2 LAX, or 3+ oblique stacks) | full 6-DOF (translation + 3D rotation)   |
| 2, or single-slice LAX views only                                  | translation + in-plane rotation (4-DOF)  |
| fewer than 2                                                       | nothing (images written unaligned)       |

With fewer crossing orientations, a plane's tilt is unconstrained, and optimizing it leads to degenerate solutions.
`--dof 4` forces translation + in-plane rotation, and `--dof 3` translation only. Every image present is always
written to the output folder; images that could not be aligned keep their acquired affine. The `pose_model` column of
the summary CSV records which model each subject used.

This script has a large number of tuning options (optimizer/schedule, blur coarse-to-fine, stack regularizers,
similarity metric, point-spread-function model, mask annealing, ...); see the module docstring for the full method
description and `--help` of either entry point for every flag.

### `slice_viewer_3d.py`

Interactive PyVista viewer that renders every SAX slice and LAX view as a textured quad in
scanner/world space, using the poses from a `moco_nifti.py` run's optimization history
(`metrics/history.csv`). A slider scrubs through optimization steps; segmentation overlays,
per-plane similarity-metric coloring, intersection lines and mask contours can be toggled.
It reads the UK Biobank NIfTI layout (SAX slices, `LAX_2Ch` / `3Ch` / `4Ch`, `nnUNet_segs/`), not DICOM runs.

```
python slice_viewer_3d.py                      # most recently created run, its first subject
python slice_viewer_3d.py 1000071               # a specific subject
python slice_viewer_3d.py 1000071 --run <run-name>
```

See the module docstring for the full keyboard/mouse interaction reference.

### `slice_viewer_dicom.py`

Viewer for `moco_dicom.py` runs, built for subjects with hundreds of slices (e.g. several stacks of 40-60 slices).
Slices are grouped by stack (series), and only a subset is drawn at a time. Each stack shows its current slice and
optionally +-w neighbours, which its ghost button turns into faint outlines in the stack's colour: selecting a slice
moves the other stacks to the slices passing through it, each stack's slice slider scrolls through its slices, and a
button jumps every stack to its worst slice by the chosen metric (NCC / LNCC / NMI, or motion from the acquired pose). The selected slice (keys or double click) shows its
metrics and its intensity profiles against the crossing slice of each other stack, plus against the slices before and
after it in its own stack (where the poses make them cross; the `--sax-sax-weight` lines, also drawn in 3D with the
neighbour-lines toggle). As in `slice_viewer_3d.py`, two toggles show the images as the loss saw them at the shown
step: image blurring (the coarse-to-fine in-plane blur, on the slices and in the profiles) and PSF in profiles (each
profile averaged across the other slice's slab). Pixel data are read from the run's input DICOMs and the poses from
its history (`--logging`).

```
python slice_viewer_dicom.py                                   # newest run under --align-root, first subject
python slice_viewer_dicom.py vol1302 --run <run-name> --neighbours 3
```

### `analyze_alignment.py`

Pre- vs post-optimization alignment of every subject of a run (made with `--logging`), as sortable CSVs in
`<run>/overall_metrics/alignment_analysis/`: `alignment_subjects.csv` (one row per subject), `alignment_slices.csv`
(one row per subject-slice) and `alignment_lines.csv` (one row per intersection line). Each metric has `_pre`
(acquired geometry), `_post` (saved pose) and `_delta` columns: mean NCC / LNCC / NMI / Dice / Hausdorff, the worst
line's NCC / Dice / Hausdorff, the mean over the worst 10% of lines, and the correction (largest / mean translation and
rotation). It also writes and prints the best / worst N subjects and subject-slices per metric.

```
python analyze_alignment.py <run_dir>                        # ranked by the post-optimization values
python analyze_alignment.py <run_dir> --stage pre --top 10   # the best / worst aligned as acquired
```

## Dependencies

- `numpy`, `pandas`, `scipy`
- `nibabel` (NIfTI I/O)
- `pydicom` (DICOM I/O, `moco_dicom.py`)
- `torch` (pose optimization)
- `pyvista` (3D viewer)
- `matplotlib` (figures, `--logging`)

```
pip install numpy pandas scipy nibabel pydicom torch pyvista matplotlib
```

## References

[1] Ugurlu, D., Qian, S., Fairweather, E., Mauger, C., Ruijsink, B., Dal Toso, L., Deng, Y.,
Strocchi, M., Razavi, R., Young, A., et al. (2025). Cardiac digital twins at scale from MRI: open
tools and representative models from ~55000 UK Biobank participants. *PLOS ONE*, 20(7), e0327158.
Code: https://github.com/CEMRG-publications/Ugurlu_2025_PLoS_One

```bibtex
@article{ugurlu2025cardiac,
  title={Cardiac digital twins at Scale from MRI: open tools and representative models from~55000 UK Biobank participants},
  author={Ugurlu, Devran and Qian, Shuang and Fairweather, Elliot and Mauger, Charlene and Ruijsink, Bram and Toso, Laura Dal and Deng, Yu and Strocchi, Marina and Razavi, Reza and Young, Alistair and others},
  journal={Plos one},
  volume={20},
  number={7},
  pages={e0327158},
  year={2025},
  publisher={Public Library of Science, San Francisco, CA USA}
}
```

[2] Stolt-Ansó, N., Sideri-Lampretsa, V., Dannecker, M., & Rueckert, D. (2024). Intensity-based 3D
motion correction for cardiac MR images. In *2024 IEEE International Symposium on Biomedical
Imaging (ISBI)* (pp. 1-4). IEEE.

```bibtex
@inproceedings{stolt2024intensity,
  title={Intensity-based 3D motion correction for cardiac MR images},
  author={Stolt-Ans{\'o}, Nil and Sideri-Lampretsa, Vasiliki and Dannecker, Maik and Rueckert, Daniel},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```
