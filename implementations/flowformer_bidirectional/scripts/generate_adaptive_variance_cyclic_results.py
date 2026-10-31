#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--num_ode_steps', type=int, default=5)
    p.add_argument('--num_transitions', type=int, default=20)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--variance_samples', type=int, default=4)
    p.add_argument('--lookahead_min', type=int, default=2)
    p.add_argument('--lookahead_max', type=int, default=6)
    p.add_argument('--backward_repeats', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--device_idx', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--out_dir', default='results/adaptive_variance_cyclic')
    p.add_argument('--traj_indices', type=int, nargs='*', default=None)
    p.add_argument('--all_trajs', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def resolve(x):
    p = Path(x).expanduser()
    if not p.is_absolute(): p = ROOT / p
    p = p.resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    return p


@torch.inference_mode()
def sample_step(model, x, steps, direction):
    return model.sample(x_cond=x, num_ode_steps=steps, c=direction)[:, -1].detach().cpu()


@torch.inference_mode()
def rollout(model, x0, n, steps, direction):
    states = [x0.detach().cpu()]
    x = states[0]
    for _ in range(n):
        x = sample_step(model, x, steps, direction)
        states.append(x)
    return torch.stack(states, dim=1)


def gather(cands, idx):
    return cands[idx, torch.arange(cands.shape[1])]


def minmax(v, eps):
    lo, hi = v.min(0).values, v.max(0).values
    return (v - lo.unsqueeze(0)) / (hi - lo).clamp_min(eps).unsqueeze(0)


@torch.inference_mode()
def predictive_variance(model, candidate, m, steps, c_fwd):
    futures = torch.stack([sample_step(model, candidate, steps, c_fwd) for _ in range(m)], 0)
    return futures.var(0, unbiased=False).mean((1,2,3))


def adaptive_horizon(var_kb, lmin, lmax, remaining):
    q1 = torch.quantile(var_kb, 0.33, dim=0)
    q2 = torch.quantile(var_kb, 0.67, dim=0)
    z = ((var_kb - q1.unsqueeze(0)) / (q2 - q1).clamp_min(1e-8).unsqueeze(0)).clamp(0,1)
    h = torch.round(lmin + z * (lmax - lmin)).long()
    return h.clamp(1, min(lmax, remaining))


def path_score(fwd, bwd, eps):
    rev = torch.flip(fwd, [1])
    raw = ((bwd - rev) ** 2).mean((2,3,4))[:,1:].mean(1)
    energy = ((fwd[:,1:] - fwd[:,:-1]) ** 2).mean((2,3,4)).mean(1)
    return raw / energy.clamp_min(eps), raw, energy


@torch.inference_mode()
def run_selector(model, real, a, c_fwd, c_bwd):
    b, _, c, h, w = real.shape
    out = torch.empty(b, a.num_transitions+1, c, h, w)
    out[:,0] = real[:,0]
    current = out[:,0]

    selected_h = torch.empty(b, a.num_transitions, dtype=torch.long)
    hit = torch.empty(b, a.num_transitions, dtype=torch.bool)
    gain = torch.empty(b, a.num_transitions)
    chosen = torch.empty(b, a.num_transitions, dtype=torch.long)
    cycle_sel = torch.empty(b, a.num_transitions)
    var_sel = torch.empty(b, a.num_transitions)

    for t in range(a.num_transitions):
        remaining = a.num_transitions - t
        print(f'adaptive greedy {t+1}/{a.num_transitions}', flush=True)

        candidates, variances = [], []
        for _ in range(a.K):
            cand = sample_step(model, current, a.num_ode_steps, c_fwd)
            candidates.append(cand)
            variances.append(predictive_variance(model, cand, a.variance_samples, a.num_ode_steps, c_fwd))

        cands = torch.stack(candidates, 0)          # [K,B,C,H,W]
        vars_ = torch.stack(variances, 0)           # [K,B]
        horizons = adaptive_horizon(vars_, a.lookahead_min, a.lookahead_max, remaining)

        cycle_scores = []
        for k in range(a.K):
            per_b = []
            for bi in range(b):
                L = int(horizons[k,bi])
                cur = current[bi:bi+1]
                first = cands[k,bi:bi+1]
                if L == 1:
                    fwd = torch.stack([cur, first], 1)
                else:
                    tail = rollout(model, first, L-1, a.num_ode_steps, c_fwd)
                    fwd = torch.cat([cur.unsqueeze(1), tail], 1)
                reps = []
                for _ in range(a.backward_repeats):
                    bwd = rollout(model, fwd[:,-1], L, a.num_ode_steps, c_bwd)
                    reps.append(path_score(fwd, bwd, a.eps)[0])
                per_b.append(torch.stack(reps).mean())
            cycle_scores.append(torch.stack(per_b))
        cyc = torch.stack(cycle_scores, 0)

        score = a.alpha * minmax(cyc, a.eps) + (1-a.alpha) * minmax(vars_, a.eps)
        best = torch.argmin(score, 0)
        selected = gather(cands, best)

        true_next = real[:,t+1]
        true_err = ((cands - true_next.unsqueeze(0)) ** 2).mean((2,3,4))
        oracle = torch.argmin(true_err, 0)
        bi = torch.arange(b)
        sel_true = true_err[best, bi]
        avg_true = true_err.mean(0)

        out[:,t+1] = selected
        chosen[:,t] = best
        selected_h[:,t] = horizons[best,bi]
        cycle_sel[:,t] = cyc[best,bi]
        var_sel[:,t] = vars_[best,bi]
        hit[:,t] = best.eq(oracle)
        gain[:,t] = (avg_true - sel_true) / avg_true.clamp_min(a.eps)
        current = selected

        print(f'  mean horizon={selected_h[:,t].float().mean().item():.2f}, '
              f'oracle hit={hit[:,t].float().mean().item():.3f}, '
              f'gain={100*gain[:,t].mean().item():.2f}%', flush=True)

    return {'rollout':out, 'selected_indices':chosen, 'selected_horizons':selected_h,
            'selected_cycle_scores':cycle_sel, 'selected_variances':var_sel,
            'oracle_hit':hit, 'local_gain':gain}


def metrics(pred, true):
    e = ((pred-true)**2).mean((2,3,4))
    mid = (pred.shape[1]-1)//2
    return {'tau0':e[:,0], 'tau_mid':e[:,mid], 'tau_final':e[:,-1], 'rollout_mean':e.mean(1)}


def agg(v):
    return float(v.mean()), float(v.std(unbiased=False))


def main():
    a = args_parser()
    if not 0 <= a.alpha <= 1: raise ValueError('--alpha must be in [0,1]')
    if a.lookahead_min < 1 or a.lookahead_max < a.lookahead_min: raise ValueError('invalid lookahead range')
    if a.variance_samples < 2: raise ValueError('--variance_samples must be >=2')

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ckpt_p, data_p = resolve(a.ckpt), resolve(a.dataset)
    outdir = Path(a.out_dir).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_p, map_location='cpu')
    cfg = ckpt['args']; ms, ds = cfg['ms'], cfg['ds']
    vids = np.load(data_p, mmap_mode='r')
    n, t, _, H, W = vids.shape
    need = a.num_transitions + 1
    if t < need: raise ValueError(f'dataset has {t} states; need {need}')

    ids = list(range(n)) if a.all_trajs else (a.traj_indices if a.traj_indices is not None else [0,n//2,n-1])
    device = DeviceManager.list_devices()[a.device_idx]
    model = FlowFormerInference.from_pretrained(checkpoint_path=str(ckpt_p), img_size=(H,W), device=device)
    c_fwd = torch.tensor([1.,0.]); c_bwd = torch.tensor([0.,1.])

    rows, details = [], []
    for s in range(0, len(ids), a.batch_size):
        batch_ids = ids[s:s+a.batch_size]
        print(f'Batch trajectories: {batch_ids}', flush=True)
        real = torch.from_numpy(np.asarray(vids[batch_ids,:need])).float()
        res = run_selector(model, real, a, c_fwd, c_bwd)
        m = metrics(res['rollout'], real)
        for j, tid in enumerate(batch_ids):
            rows.append({'model':f'flowformer_{ms}_{ds}','dataset':ds,'method':'adaptive_variance_greedy_cyclic',
                         'trajectory_index':tid,'tau0':float(m['tau0'][j]),'tau_mid':float(m['tau_mid'][j]),
                         'tau_final':float(m['tau_final'][j]),'rollout_mean':float(m['rollout_mean'][j]),
                         'mean_selected_horizon':float(res['selected_horizons'][j].float().mean()),
                         'oracle_hit_rate':float(res['oracle_hit'][j].float().mean()),
                         'mean_local_gain':float(res['local_gain'][j].mean()),
                         'K':a.K,'variance_samples':a.variance_samples,'lookahead_min':a.lookahead_min,
                         'lookahead_max':a.lookahead_max,'backward_repeats':a.backward_repeats,'alpha':a.alpha})
        details.append({'trajectory_indices':batch_ids, **{k:v.detach().cpu() for k,v in res.items() if k!='rollout'}})

    names = ['tau0','tau_mid','tau_final','rollout_mean','mean_selected_horizon','oracle_hit_rate','mean_local_gain']
    summary = {}
    print('\nDATASET-LEVEL SUMMARY')
    for name in names:
        v = torch.tensor([r[name] for r in rows], dtype=torch.float64)
        mu, sd = agg(v); summary[name] = {'mean':mu,'std':sd}
        print(f'{name}: {mu:.8g} +/- {sd:.8g}')

    suffix = 'all_trajs' if a.all_trajs else 'selected_trajs'
    base = f'adaptive_variance_flowformer_{ms}_{ds}_K{a.K}_M{a.variance_samples}_L{a.lookahead_min}-{a.lookahead_max}_A{a.alpha:.2f}_T{a.num_transitions}_{suffix}'
    csv_p, pt_p, txt_p = outdir/f'{base}.csv', outdir/f'{base}.pt', outdir/f'{base}.txt'

    with csv_p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    torch.save({'checkpoint':str(ckpt_p),'dataset':str(data_p),'args':vars(a),'rows':rows,'summary':summary,'details':details}, pt_p)
    with txt_p.open('w') as f:
        for name in names:
            f.write(f"{name}: {summary[name]['mean']:.8g} +/- {summary[name]['std']:.8g}\n")
        f.write('\nLaTeX row\n')
        f.write(f"& Adaptive variance cyclic, $K={a.K}$\n")
        for name in ['tau0','tau_mid','tau_final','rollout_mean']:
            f.write(f"& {summary[name]['mean']:.6f} $\\pm$ {summary[name]['std']:.6f}\n")
        f.write('\\\\\n')
    print('\nSaved:'); print(csv_p); print(pt_p); print(txt_p)


if __name__ == '__main__':
    main()
