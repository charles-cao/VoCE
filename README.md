# VoCE: Voronoi Consistency Ensemble for Multi-View Anomaly Detection

Code of the paper *VoCE: Voronoi Consistency Ensemble for Multi-View Anomaly Detection*.

VoCE repeatedly samples a small set of anchors shared by all views and covers every view with the Voronoi cells
of these anchors.

- **Shared-Anchor Voronoi Consistency** flags a sample whose nearest anchor differs across views. It needs no
  neighborhood size.
- **Cell-Normalized Transfer** grades how far the sample lies from the anchors of its cells, relative to the
  size of the cells.

The two terms are added with unit weight, the smallest weight that ranks every inconsistent sample at or above
every consistent one in each round. VoCE needs no training, runs in time linear in the number of samples, and
has two parameters: the number of anchors per round `psi` (default 16) and the number of rounds `t` (default 200).

## Installation

```bash
git clone <this repository> VoCE && cd VoCE
pip install -r requirements.txt
```
Python 3.9 or later. VoCE itself needs NumPy; the data preparation and the evaluation also use SciPy and
scikit-learn.

## Usage

```python
from voce import VoCE

# views: a list of arrays, one per view, each of shape (n_samples, n_features_of_that_view)
model = VoCE(psi=16, t=200, random_state=0).fit(views)
score = model.decision_function(views)          # higher = more anomalous
T, I = model.decision_components(views)         # transfer score and inconsistency, score = T + I
```

`fit` and the scoring methods check their input: the same number of views and of columns per view as at fit,
the same number of rows in every view, finite values, and `psi <= N`. Distances are computed in float32, which
is exact enough for scaled data; rescale features with very large magnitudes before fitting.

`demo.ipynb` walks through a toy example with the three types of multi-view anomalies, one round of VoCE with
its Voronoi cells, the effect of `psi`, and one setting of the paper.

## Reproducing the results of the paper

**Data.** The raw files are in `datasets/`, copied unchanged from the sources below; please cite the original
sources when you use them. To read them from another folder, set the environment variable `VOCE_DATA`.

| Dataset | File(s) | Source |
|---|---|---|
| Zoo | `zoo.data` | UCI Machine Learning Repository |
| Parkinsons | `parkinsons.data` | UCI Machine Learning Repository |
| WDBC | `wdbc.data` | UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic) |
| Wine | `wine.data` | UCI Machine Learning Repository |
| Pima | `pima.csv` | `data/` folder of the ECMOD repository, github.com/scu-kdde/OAM-ECMOD-2023 |
| Optdigits | `optdigits.tra`, `optdigits.tes` | UCI Machine Learning Repository |
| Caltech101-7 | `Caltech101-7.mat` | github.com/ChuanbinZhang/Multi-view-datasets |
| CiteSeer | `CiteSeer.mat` | same collection |
| BBCSport | `BBCSport.mat` | same collection |
| Reuters | `Reuters-1200.mat` | same collection |

`voce.data.prepare(name, ratio, seed)` builds the data of the paper from these files: min-max scaling, the UCI
datasets split into three views of consecutive features, the anomaly injection code of ECMOD, unchanged, and for
BBCSport and Reuters a per-document z-score within each view after injection.

**Run.**
```bash
python reproduce.py                  # all 30 settings
python reproduce.py --datasets wdbc  # one dataset
```
For every dataset, anomaly ratio and data seed 1 to 10, the script runs VoCE with `t = 200`,
`psi` in {2, 4, ..., 256} and `random_state = 1000 + seed`, selects `psi` by the mean AUC-ROC over the seeds, and
prints AUC-ROC and AUC-PR next to the values of the main results table. All 30 settings and the averages,
0.936 AUC-ROC and 0.830 AUC-PR, equal the paper. Finished runs are kept in `reproduce_cache.csv`, so an
interrupted run resumes; `--jobs` sets the number of processes.

## License

MIT. `voce/data.py` contains the anomaly injection function of ECMOD, copied unchanged.