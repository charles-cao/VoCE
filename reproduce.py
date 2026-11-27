"""Reproduces the VoCE results of the paper: AUC-ROC and AUC-PR on 10 datasets x 3 anomaly ratios, each the mean
over 10 data seeds, at the psi selected by the mean AUC-ROC as in the main results table.

    python reproduce.py                          # all 30 settings
    python reproduce.py --datasets wdbc wine     # a subset
    python reproduce.py --jobs 8                 # parallel processes

Protocol of the paper: data seeds 1 to 10, t = 200, psi in {2, 4, ..., 256} with psi < N, random_state 1000 + data
seed. Every finished run is appended to reproduce_cache.csv, so an interrupted run resumes where it stopped.
The last column compares each value with the paper.
"""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voce import VoCE
from voce.data import prepare

DATASETS = ['zoo', 'parkinsons', 'wdbc', 'wine', 'pima', 'optdigits', 'caltech', 'citeseer', 'bbcsport_tn', 'reuters_tn']
NAMES = {'zoo': 'Zoo', 'parkinsons': 'Parkinsons', 'wdbc': 'WDBC', 'wine': 'Wine', 'pima': 'Pima', 'optdigits': 'Optdigits',
         'caltech': 'Caltech101-7', 'citeseer': 'CiteSeer', 'bbcsport_tn': 'BBCSport', 'reuters_tn': 'Reuters'}
RATIOS = ['258', '555', '852']
PSIS = [2, 4, 8, 16, 32, 64, 128, 256]

# VoCE in the main results table of the paper: (selected psi, AUC-ROC, AUC-PR)
PAPER = {
    ('zoo', '258'): (8, '0.922', '0.687'), ('zoo', '555'): (16, '0.931', '0.751'), ('zoo', '852'): (16, '0.962', '0.812'),
    ('parkinsons', '258'): (16, '0.877', '0.702'), ('parkinsons', '555'): (32, '0.928', '0.832'), ('parkinsons', '852'): (32, '0.963', '0.917'),
    ('wdbc', '258'): (8, '0.956', '0.854'), ('wdbc', '555'): (8, '0.981', '0.929'), ('wdbc', '852'): (16, '0.987', '0.960'),
    ('wine', '258'): (8, '0.898', '0.630'), ('wine', '555'): (16, '0.930', '0.729'), ('wine', '852'): (8, '0.956', '0.810'),
    ('pima', '258'): (4, '0.789', '0.528'), ('pima', '555'): (64, '0.871', '0.697'), ('pima', '852'): (128, '0.930', '0.810'),
    ('optdigits', '258'): (256, '0.959', '0.868'), ('optdigits', '555'): (256, '0.974', '0.921'), ('optdigits', '852'): (256, '0.990', '0.972'),
    ('caltech', '258'): (2, '0.894', '0.744'), ('caltech', '555'): (4, '0.936', '0.842'), ('caltech', '852'): (4, '0.974', '0.937'),
    ('citeseer', '258'): (128, '0.780', '0.636'), ('citeseer', '555'): (128, '0.864', '0.784'), ('citeseer', '852'): (64, '0.945', '0.918'),
    ('bbcsport_tn', '258'): (8, '0.946', '0.827'), ('bbcsport_tn', '555'): (16, '0.968', '0.888'), ('bbcsport_tn', '852'): (8, '0.986', '0.956'),
    ('reuters_tn', '258'): (4, '0.996', '0.982'), ('reuters_tn', '555'): (8, '0.997', '0.988'), ('reuters_tn', '852'): (8, '0.998', '0.993'),
}
PAPER_MEAN = ('0.936', '0.830')
CACHE = 'reproduce_cache.csv'


def run_one(ds, ratio, seed, psis, t):
    """All psi values for one data file; returns rows (dataset, ratio, seed, psi, AUC-ROC, AUC-PR)."""
    views, y = prepare(ds, ratio, seed)
    rows = []
    for psi in psis:
        if psi >= len(y):
            continue
        s = VoCE(psi=psi, t=t, random_state=1000 + seed).fit(views).decision_function(views)
        rows.append((ds, ratio, seed, psi, roc_auc_score(y, s), average_precision_score(y, s)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    ap.add_argument('--ratios', nargs='+', default=RATIOS, choices=RATIOS)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    seeds, t = range(1, 11), 200

    done = {}
    if os.path.exists(CACHE):
        for r in csv.reader(open(CACHE)):
            done[(r[0], r[1], int(r[2]), int(r[3]))] = (float(r[4]), float(r[5]))
    todo = [(ds, ra, sd) for ds in a.datasets for ra in a.ratios for sd in seeds
            if not any((ds, ra, sd, p) in done for p in PSIS)]
    if todo:
        print(f'{len(todo)} data files to run with {a.jobs} processes; results are appended to {CACHE}', flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(a.jobs) as ex, open(CACHE, 'a', newline='') as fh:
            w = csv.writer(fh)
            futs = {ex.submit(run_one, ds, ra, sd, PSIS, t): (ds, ra, sd) for ds, ra, sd in todo}
            for k, f in enumerate(as_completed(futs), 1):
                for r in f.result():
                    w.writerow(r); done[r[:4]] = r[4:]
                fh.flush()
                ds, ra, sd = futs[f]
                print(f'  [{k}/{len(todo)}] {ds} {ra} seed {sd}   {time.time() - t0:.0f}s', flush=True)

    print(f'\n{"Dataset":14s}{"ratio":>6s}{"psi":>5s}{"AUC-ROC":>9s}{"AUC-PR":>8s}   paper (psi, AUC-ROC, AUC-PR)')
    rocs, prs, same, n = [], [], 0, 0
    for ds in a.datasets:
        for ra in a.ratios:
            res = {p: np.array([done[(ds, ra, sd, p)] for sd in seeds]) for p in PSIS if (ds, ra, 1, p) in done}
            best = max(res, key=lambda p: res[p][:, 0].mean())      # first psi with the highest mean AUC-ROC
            roc, pr = f'{res[best][:, 0].mean():.3f}', f'{res[best][:, 1].mean():.3f}'
            rocs.append(res[best][:, 0].mean()); prs.append(res[best][:, 1].mean())
            ref = PAPER[(ds, ra)]
            ok = (best, roc, pr) == ref
            same += ok; n += 1
            print(f'{NAMES[ds]:14s}{ra:>6s}{best:5d}{roc:>9s}{pr:>8s}   {ref[0]:4d} {ref[1]} {ref[2]}   {"same" if ok else "DIFFERENT"}')
    print(f'\n{same} of {n} settings equal the paper.')
    if n == 30:
        m = (f'{np.mean(rocs):.3f}', f'{np.mean(prs):.3f}')
        print(f'Average over the 30 settings: AUC-ROC {m[0]}, AUC-PR {m[1]}   paper {PAPER_MEAN[0]}, {PAPER_MEAN[1]}   '
              f'{"same" if m == PAPER_MEAN else "DIFFERENT"}')


if __name__ == '__main__':
    main()
