# CMR Motion Correction

Inter-slice motion estimation and correction for cardiac cine MRI: UK Biobank SAX/LAX acquisitions, or any set of
cine slices and stacks as NIfTI or DICOM. A 3D viewer lets you inspect the results.

Every cine acquisition (each short-axis slice, each long-axis view) is taken in its own breath-hold. As a result, the
slices often disagree about where the heart is in scanner space. This pipeline estimates a rigid pose correction for
every slice by comparing image intensities along the lines where pairs of slices intersect. It then writes the
corrected images back in the format they were read in.

![Optimization process in the 3D viewer (slice_viewer_3d.py)](media/optimization_sa09.gif)

#### Note: This project was vibe-coded within a couple of days. All praise Claude Code.

-------------------------

## Pipeline

```
<root>/<subject>/...                      -> your data, one folder per subject (NIfTI or DICOM; layouts below)
            |
            v
moco_nifti.py  /  moco_dicom.py           -> entry points: load a subject, run the motion correction,
            |                                write the corrected images in the same format
            v
motion_gradient.py                        -> the motion correction (library): per-slice rigid poses,
            |                                plus optional metrics and figures (--logging)
            v
slice_viewer_3d.py  /  slice_viewer_dicom.py   -> interactive 3D viewers of the optimization history
```

A first run and a look at its result:

```
python moco_nifti.py --root <ukbb_dir> --out-root <alignment_dir> --run-name my_run
python slice_viewer_3d.py --run-dir <alignment_dir>/my_run
```

## Input data

### UK Biobank NIfTI (`moco_nifti.py`)

Point `--root` at a folder with one subfolder per subject. No preparation is needed: each subject folder is searched
recursively, and files are classified by name.

```
<root>/
  <subject>/
    Instance_2/                    # any folder structure
      SAX.nii.gz                   # SAX stack (X, Y, Z, T), and/or single-slice files SAX_slice<nn>.nii.gz
      LAX_2Ch.nii.gz               # LAX views (X, Y, 1, T); any other NIfTI is a LAX image of its own
      LAX_3Ch.nii.gz
      LAX_4Ch.nii.gz
      ao.nii.gz                    # optional aortic slice
      nnUNet_segs/                 # optional; matched to their image by name (SAX slices by position)
        SAX_nnUNetSeg.nii.gz
        LAX_2Ch_nnUNetSeg.nii.gz
        ...
```

- **Segmentations** are the nnU-Net ones of the [Ugurlu et al. pipeline](https://github.com/CEMRG-publications/Ugurlu_2025_PLoS_One) [1].
  They define the heart masks and the Dice / Hausdorff metrics. For the SAX stack, `seg_sa.nii.gz` is preferred when
  present. Missing images or segmentations are allowed.
- **The aortic slice** (`ao.nii.gz`, an axial slice through the ascending aorta) is treated as one more LAX view.
  It has no segmentation, so its lines are masked by the partner image (details below). `--use-ao false` leaves it out.
- **Slice thickness** is not stored in NIfTI headers (only the spacing), so it is set with `--sax-thickness` (8 mm)
  and `--lax-thickness` (6 mm).
- **Output**, per subject: `SAX_slice<nn>.nii.gz`, `LAX_2Ch.nii.gz`, ... with the corrected affines, and the
  segmentations in `nnUNet_segs/`. `--image-dtype uint8` saves about 30% of the disk space.

### DICOM (`moco_dicom.py`)

For other cine data, e.g. several stacks in different orientations. Every DICOM series below a subject folder is one
image, named after its folder; its slices are the distinct positions and its frames are ordered by `TriggerTime`.

```
<root>/
  <subject>/
    <series_1>/phase_01/*.dcm ... phase_25/*.dcm      # any folder layout
    <series_2>/...
```

DICOM data has no segmentations, so the lines are masked by a disc around the centre of each stack instead
(`--center-mask-radius-mm`, 40 mm). The output rewrites every input file at the same relative path, with corrected
`ImagePositionPatient` / `ImageOrientationPatient` and new UIDs. With `--logging`, `metrics/metric_trends.png` shows
whether a run improved, is still improving, or diverged.

```
python moco_dicom.py --root <dicom_dir>                     # output: <dicom_dir>_aligned
python moco_dicom.py --root <dicom_dir> --pair-keep 0.25 --sample-dtype float16   # many slices, small GPU
```

## How the alignment works (`motion_gradient.py`)

`motion_gradient.py` is the motion correction itself; the entry points above call it. In this section we first look at
the idea, then walk through what happens during a run.

### The idea

Every slice gets a **rigid pose**: a 3D translation and a 3D rotation, both starting at zero (the acquired geometry).
To score a set of poses, we pair every two slices whose orientations differ by at least 20°
(`--min-pair-angle-deg`) and sample both images along their **intersection line**. If both slices are in the right
place, the two intensity profiles along that line should look alike.

We measure the agreement with the **local normalized cross-correlation** (`--metric lncc`): the profiles are compared
in 20 mm windows sliding along the line, so every stretch of the line has to match on its own. The score is averaged
over 10 cardiac phases (`--n-frames`) and over all lines. An optimizer (PyTorch, plain SGD) then moves the slices to
maximize it.

How many degrees of freedom a slice gets depends on the images a subject has. With few crossing orientations, a
slice's tilt cannot be determined and is not optimized:

| Available images                                                   | Optimized with `--dof 6`                 |
|--------------------------------------------------------------------|------------------------------------------|
| 3 or more, at least one a stack (SAX + 2 LAX, or 3+ oblique stacks) | full 6-DOF (translation + 3D rotation)   |
| 2, or single-slice LAX views only                                  | translation + in-plane rotation (4-DOF)  |
| fewer than 2                                                       | nothing (images written unaligned)       |

The `pose_model` column of the summary CSV records which model a subject used. Every image is always written to the
output; images that could not be aligned keep their acquired pose.

The method builds on the intersection-line intensity matching of [Stolt-Ansó et al.](https://arxiv.org/pdf/2404.00767) [2].

### How an alignment run proceeds

A run does not optimize the same score from start to finish. Several things change over its 1000 steps (`--steps`):
the learning rate, the heart mask, the image blur and the slice profile. Wide, smooth settings at the start help the
slices find their rough position; tight, sharp settings at the end refine it. All numbers below are the defaults.

#### Learning rate

With plain **SGD** (`--optimizer sgd`), a slice moves by the learning rate times its gradient. There is one learning
rate for translations (`--lr`, 80) and one for rotations (`--rot-lr`, 20).

Both **decay linearly** to 50% of their start value over the run (`--lr-schedule linear`). Why decay at all? At every
step only a random 80% of the lines is used (`--pair-keep`), which makes the gradient noisy. Large early steps move the
slices quickly; smaller late steps let them settle instead of jittering around the optimum.

#### Heart mask and its dilation

The heart moves almost rigidly between breath-holds, but the chest wall and the liver do not. We therefore only use
the part of each line that lies inside a **heart mask**, built from each LAX view's segmentation:

1. **Heart region:** every pixel that a heart label covers in any cardiac frame.
2. **Apex extension:** the region is swept 20 mm toward the apex along the LV long axis (`--apex-extend-mm`). Apical
   segmentations are often incomplete; without this step the apical SAX slices keep few supervised points.
3. **Dilation:** the region is grown by a margin. Only line points inside the grown mask enter the score.

The dilation shrinks from 80 mm to 20 mm over the first 80% of the run (`--mask-dilation-start-mm`,
`--mask-dilation-mm`, `--mask-anneal-frac`). A wide mask gives the optimizer context while the slices are still far
off; the tight mask at the end lets the heart alone decide the final pose.

![Heart mask: annealed dilation and the apex extension](media/mask.png)

The dilation changes in 6 stages (`--anneal-stages`), and all intersection lines are rebuilt at each stage:

| Steps    | 0-132 | 133-266 | 267-399 | 400-532 | 533-666 | 667-799 | 800-1000 |
|----------|-------|---------|---------|---------|---------|---------|----------|
| Dilation | 80 mm | 70 mm   | 60 mm   | 50 mm   | 40 mm   | 30 mm   | 20 mm    |

> **Note: lines come and go.** A line only takes part if at least 40 mm of it lie inside both images and the mask
> (`--min-line-mm`). A pair of slices can therefore contribute at a wide dilation and drop out once the mask is tight.
> The 3D viewer marks such pairs as "(not in loss)" at the steps where they do not contribute.

Images without a segmentation have no mask of their own: their lines with a LAX view use that view's mask. The aortic
slice is compared with the SAX slices only within 120 mm of the LV long axis (`--sax-sax-radius-mm`).

#### Annealed blurring

A profile with many small, sharp features gives a score with many small local optima. To avoid getting stuck early,
we work **coarse-to-fine**: all images are blurred in-plane with a Gaussian that narrows over the run.

The blur starts at a sigma of 5 mm (`--blur-sigma`), decreases linearly to 2 mm over the first 80% of the run
(`--blur-sigma-end`, `--blur-frac`), and then stays at 2 mm. The applied sigma moves in steps of 0.2 mm. The blur is
only used inside the optimization; the saved images are never blurred.

![Coarse-to-fine blur](media/blur.png)

At 2 mm the profile keeps its main edges but loses the finest detail, which keeps the score smooth until the end. At
5 mm only the large structures (blood pool, myocardium) remain.

#### Slice profile (PSF)

A slice is not infinitely thin: a SAX slice is 8 mm thick, a LAX view 6 mm. When we sample one image along a line,
we therefore average it across the slab of the other slice, weighted by that slice's profile (the **point-spread
function**, `--psf-model pulse`).

Like the blur, this band starts wider than physical, 3 times the slice profile (`--psf-scale-start`), and narrows to
the physical width over the first 90% of the run (`--psf-anneal-frac`). Unlike the blur, it smooths an image only
across the intersection line, not along it.

![Slice profile (PSF)](media/psf.png)

#### Choosing the final pose

The saved pose is not simply the last one. Scores computed with a wide mask or a strong blur cannot be compared with
scores at the end of the run, so the optimizer only starts tracking the **best pose** once every schedule has finished:
from step 900 by default. From there, the full score (all lines) is computed every 10 steps (`--eval-every`) and for
the final pose, and the best of these is saved.

> **Note:** a schedule that lasts the whole run (`--mask-anneal-frac`, `--blur-frac` or `--psf-anneal-frac` set to 1)
> leaves no tracking window, and the final pose is used.

#### Keeping slices close to their acquired pose

Across the UK Biobank, 99.9% of the slice corrections are below about 23 mm and 8°. A slice that moves much further has
almost certainly gone wrong, which mostly happens in subjects with few slices. Three mechanisms prevent it, from soft
to hard:

- **Step cap:** at most 2 mm and 1° per optimizer step (`--max-step-mm`, `--max-step-deg`).
- **Dead-zone priors:** free within 20 mm and 10° of the acquired pose; beyond that, a penalty grows with the square of
  the excess (`--shift-prior`, `--shift-free-mm`, `--rot-prior`, `--rot-free-deg`).
- **Hard limits:** never more than 50 mm or 20° (`--max-shift-mm`, `--max-rot-deg`).

A weak **gauge** penalty on the mean pose of all slices (`--gauge`) removes the freedom to move every slice together,
which would not change the score. Setting any of these to 0 switches it off.

#### One run, step by step

The figure below shows one subject aligned with the default settings. On the left are the schedules described above;
on the right the metrics of the poses at each step. All metrics are computed with the final 20 mm mask and without
blur, so the steps are comparable.

![One alignment run with the default settings](media/metrics.png)

After a quick jump in the first 20 steps, the metrics keep improving steadily while the mask, blur and slice
profile tighten: Dice levels off around step 600, while NCC keeps rising until the end. For this subject (1000071), NCC
rises from 0.72 to 0.86, Dice from 0.61 to 0.77, and the mean Hausdorff distance drops from 30 mm to 18 mm.

### Changing the settings

Every option has a general default in `motion_gradient.build_parser`. An entry point repeats the options worth seeing
at a glance with its own defaults, which replace the general ones (e.g. `--sample-dtype` is float32 in
`motion_gradient.py` and float16 in `moco_nifti.py`). `--help` on either entry point lists everything.

To try a variant, align a small batch of subjects into its own run directory and compare:

```
python moco_nifti.py --root <ukbb_dir> --out-root <alignment_dir> --run-name test_blur05 --start-index 0 --end-index 25 --blur-sigma-end 0.5
```

Use a new `--run-name` for every variant. A second run into the same directory overwrites the subjects' images, so
only the newest variant can be opened in the viewer.

The figures in this section are made from one subject by `media/make_readme_figures.py`, using the code itself
(including a fresh alignment for the metrics figure). Regenerate them after changing a default:

```
python media/make_readme_figures.py <ukbb_dir>/<subject>
```

## Viewing the results

### `slice_viewer_3d.py`

An interactive PyVista viewer for `moco_nifti.py` runs. Every slice is drawn in scanner space at its pose; a slider
scrubs through the optimization steps. Segmentations, metric-coloured outlines, intersection lines and mask contours
can be toggled. On the right, the intensity profiles along the selected slice's intersection lines are shown as the
loss sees them, with the selected metric per point below (scroll with the mouse wheel).

```
python slice_viewer_3d.py                                    # newest run, its first subject
python slice_viewer_3d.py 1000071 --run <run-name>           # a given subject and run
python slice_viewer_3d.py --remote user@host:/path/to/run --subjects worst_cases/slice_shift_max.txt
```

With `--remote`, subjects are read over ssh from another machine and downloaded once into a local cache. `--subjects`
steps through a list of subjects with `[` / `]`, e.g. one written by `worst_cases.py`. The module docstring lists every
key and mouse action.

### `slice_viewer_dicom.py`

The viewer for `moco_dicom.py` runs, built for subjects with hundreds of slices. Slices are grouped by stack, and only
a subset is drawn at a time: each stack shows its current slice and optionally a few neighbours. A button jumps every
stack to its worst slice by the chosen metric.

```
python slice_viewer_dicom.py vol1302 --run <run-name> --neighbours 3
```

### Finding the worst cases (`worst_cases.py`)

On a large cohort you cannot look at every subject. `worst_cases.py` ranks the subjects of a run by many metrics
(largest single-slice motion, worst slice Dice, NCC lost by the alignment, ...). It writes one list per metric,
worst first, with the slice where each subject is worst.

```
python worst_cases.py --run-dir <alignment_dir>/<run> --top 50
python worst_cases.py --remote user@host:/path/to/run             # runs on the server, only the lists come back
```

`worst_cases_commands.txt` lists ready-made viewer commands for every list.

## Running on a SLURM cluster

`submit_alignment.sh` aligns a cohort with `--logging` (metrics and history); `submit_alignment_uint8.sh` writes the
aligned images only, as uint8. Both resume an interrupted run (`--skip-processed`), and large cohorts can be split into
chunks of subject folders:

```
sbatch submit_alignment.sh --run-name my_run --start-index 0 --end-index 10000
sbatch submit_alignment.sh --run-name my_run --start-index 10000 --end-index 20000
```

`download_ukbb.py` downloads subjects in batches over SSH. UK Biobank data may only be stored on approved,
access-restricted infrastructure; never download it to personal machines.

## Speed

Three things keep the GPU busy:

- **Batching:** several subjects are aligned in one optimization (`--batch-size`). With the default 0, the first batch
  measures the GPU memory a subject needs, then the batch size doubles for as long as that speeds things up.
- **Worker processes:** loading, preparing the intersection lines and writing the outputs run in parallel processes
  (`--workers`, up to about the number of CPU cores).
- **Coarse early steps:** the first half of the run samples every 4th cine frame and every 2nd line point
  (`--coarse-frac`, `--coarse-frame-stride`, `--coarse-point-stride`).

Seconds per UK Biobank subject with the default settings (1000 steps):

| Machine | Workers | `--logging true` | `--logging false` |
|---|---|---|---|
| Laptop: RTX 5070 Laptop GPU (8 GB), Intel i7-13650HX | 8 | 2.3s | 1.5s |
| Cluster: NVIDIA A40 (46 GB), AMD EPYC 7662 | 40 | 1.3s | not measured |

The laptop numbers are for 64 subjects, including start-up and the end-of-run figures; the GPU is the limit there, and
batch sizes from 8 to 24 run equally fast. The cluster number is the rate of a full-cohort run (`submit_alignment.sh`),
which puts all 57,700 UK Biobank subjects with metrics at about 21 hours.

## Dependencies

- `numpy`, `pandas`, `scipy`
- `nibabel` (NIfTI I/O), `pydicom` (DICOM I/O)
- `torch` (pose optimization)
- `pyvista` (3D viewers)
- `matplotlib` (figures)

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
