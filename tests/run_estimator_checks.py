"""Run sklearn's complete estimator checks and write a machine-readable report.

Invocation: python tests/run_estimator_checks.py [output.json]
This is separate from pytest because sklearn-version-specific check inventories
change independently of the project's numerical regression tests.
"""
import json
import sys
import warnings
from pathlib import Path
from sklearn.utils.estimator_checks import check_estimator
from fastsvm import SVC, SVR, OneClassSVM, LinearSVC, LinearSVR, RBFSampler, Nystroem, KernelCenterer

estimators = [SVC(tol=1e-10, gamma=.4), SVR(tol=1e-10, gamma=.4), OneClassSVM(tol=1e-11, gamma=.4),
              LinearSVC(tol=1e-10, max_iter=100000, random_state=0),
              LinearSVR(tol=1e-10, max_iter=100000, random_state=0),
              RBFSampler(n_components=20, random_state=0),
              Nystroem(n_components=5, random_state=0), KernelCenterer()]
report = {}
for estimator in estimators:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        results = check_estimator(estimator, on_fail=None, on_skip=None)
    items = []
    for r in results:
        items.append({k: (str(v) if k in ('exception', 'estimator') else v)
                      for k, v in r.items()})
    name = type(estimator).__name__
    report[name] = {"checks": items, "warnings": sorted(set(str(w.message) for w in caught))}
    failures = [r for r in items if r['status'] == 'failed']
    print(name, len(items), 'checks', len(failures), 'failures', flush=True)
    for r in failures:
        print(' ', r['check_name'], r.get('exception'), flush=True)
path = Path(sys.argv[1] if len(sys.argv)>1 else 'docs/estimator_checks.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2, default=str)+'\n')
if any(r['status']=='failed' for x in report.values() for r in x['checks']):
    sys.exit(1)
