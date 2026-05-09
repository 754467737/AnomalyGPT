import argparse, json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics_glob', required=True, help='e.g. outputs/.../run_*/eval/metrics.json')
    args = ap.parse_args()
    files = sorted(Path('.').glob(args.metrics_glob))
    vals = {}
    for fp in files:
        m = json.loads(fp.read_text())
        for k,v in m.items():
            if isinstance(v,(int,float)) and v==v:
                vals.setdefault(k,[]).append(v)
    out = {k:{'mean':float(np.mean(v)),'std':float(np.std(v)),'n':len(v)} for k,v in vals.items()}
    print(json.dumps(out, indent=2))
