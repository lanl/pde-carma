#!/usr/bin/env python3
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--num_ode_steps', type=int, default=5)
    p.add_argument('--num_transitions', type=int, default=20)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--device_idx', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--out_dir', default='results/greedy_oracle')
    p.add_argument('--traj_indices', type=int, nargs='*', default=None)
    p.add_argument('--all_trajs', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def resolve_path(text):
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


@torch.inference_mode()
def sample_step(model, x, num_ode_steps, direction):
    path = model.sample(x_cond=x, num_ode_steps=num_ode_steps, c=direction)
    return path[:, -1].detach().cpu()


def gather(cands, idx):
    b = torch.arange(cands.shape[1])
    return cands[idx, b]


@torch.inference_mode()
def rollout_once(model, x0, nsteps, ode_steps, c_fwd):
    b, c, h, w = x0.shape
    out = torch.empty(b, nsteps + 1, c, h, w, dtype=x0.dtype)
    out[:, 0] = x0.cpu()
    cur = out[:, 0]
    for t in range(nsteps):
        cur = sample_step(model, cur, ode_steps, c_fwd)
        out[:, t + 1] = cur
    return out


@torch.inference_mode()
def forward_best(model, real, K, nsteps, ode_steps, c_fwd):
    rollouts = []
    for k in range(K):
        print(f'  forward-best rollout {k+1}/{K}', flush=True)
        rollouts.append(rollout_once(model, real[:, 0], nsteps, ode_steps, c_fwd))
    rollouts = torch.stack(rollouts)  # [K,B,T,C,H,W]
    errs = ((rollouts - real.unsqueeze(0)) ** 2).mean(dim=(2,3,4,5))
    idx = torch.argmin(errs, dim=0)
    b = torch.arange(real.shape[0])
    return {'rollout': rollouts[idx, b], 'best_idx': idx, 'candidate_errors': errs}


@torch.inference_mode()
def stepwise_oracle(model, real, K, nsteps, ode_steps, c_fwd):
    b, _, c, h, w = real.shape
    out = torch.empty(b, nsteps + 1, c, h, w, dtype=real.dtype)
    out[:, 0] = real[:, 0]
    chosen = torch.empty(b, nsteps, dtype=torch.long)
    chosen_err = torch.empty(b, nsteps)
    cur = out[:, 0]
    for t in range(nsteps):
        print(f'  stepwise oracle {t+1}/{nsteps}', flush=True)
        cands = torch.stack([sample_step(model, cur, ode_steps, c_fwd) for _ in range(K)])
        true_next = real[:, t + 1]
        errs = ((cands - true_next.unsqueeze(0)) ** 2).mean(dim=(2,3,4))
        idx = torch.argmin(errs, dim=0)
        sel = gather(cands, idx)
        bidx = torch.arange(b)
        out[:, t + 1] = sel
        chosen[:, t] = idx
        chosen_err[:, t] = errs[idx, bidx]
        cur = sel
    return {'rollout': out, 'selected_indices': chosen, 'selected_true_errors': chosen_err}


@torch.inference_mode()
def greedy_cyclic(model, real, K, nsteps, ode_steps, c_fwd, c_bwd):
    b, _, c, h, w = real.shape
    out = torch.empty(b, nsteps + 1, c, h, w, dtype=real.dtype)
    out[:, 0] = real[:, 0]
    chosen = torch.empty(b, nsteps, dtype=torch.long)
    cyc_err = torch.empty(b, nsteps)
    true_err = torch.empty(b, nsteps)
    cur = out[:, 0]
    for t in range(nsteps):
        print(f'  greedy cyclic {t+1}/{nsteps}', flush=True)
        fwd, recon = [], []
        for _ in range(K):
            y = sample_step(model, cur, ode_steps, c_fwd)
            xhat = sample_step(model, y, ode_steps, c_bwd)
            fwd.append(y); recon.append(xhat)
        fwd = torch.stack(fwd)
        recon = torch.stack(recon)
        cyc = ((recon - cur.unsqueeze(0)) ** 2).mean(dim=(2,3,4))
        true_next = real[:, t + 1]
        terr = ((fwd - true_next.unsqueeze(0)) ** 2).mean(dim=(2,3,4))
        idx = torch.argmin(cyc, dim=0)
        sel = gather(fwd, idx)
        bidx = torch.arange(b)
        out[:, t + 1] = sel
        chosen[:, t] = idx
        cyc_err[:, t] = cyc[idx, bidx]
        true_err[:, t] = terr[idx, bidx]
        cur = sel
    return {
        'rollout': out,
        'selected_indices': chosen,
        'selected_cycle_errors': cyc_err,
        'selected_true_errors': true_err,
    }


def metrics(pred, true):
    per_t = ((pred - true) ** 2).mean(dim=(2,3,4))
    mid = (pred.shape[1] - 1) // 2
    return {
        'tau0': per_t[:, 0],
        'tau_mid': per_t[:, mid],
        'tau_final': per_t[:, -1],
        'rollout_mean': per_t.mean(dim=1),
    }


def aggregate(x):
    return {'mean': float(x.mean()), 'std': float(x.std(unbiased=False))}


def summarize(method_metrics):
    return {
        m: {k: aggregate(v) for k, v in vals.items()}
        for m, vals in method_metrics.items()
    }


def write_txt(path, tag, K, nsteps, summary):
    labels = {
        'forward_best': 'Forward-best oracle',
        'stepwise_oracle': 'Stepwise oracle',
        'greedy_cyclic': 'Greedy cyclic',
    }
    with open(path, 'w') as f:
        f.write(f'Model: {tag}\nK: {K}\nTransitions: {nsteps}\nStates: {nsteps+1}\n\n')
        for method in ['forward_best','stepwise_oracle','greedy_cyclic']:
            s = summary[method]; label = labels[method]
            f.write(f'{label}, K={K}\n')
            for key, pretty in [('tau0','tau_n=0'),('tau_mid','tau_n=N/2'),('tau_final','tau_n=N'),('rollout_mean','rollout_mean')]:
                f.write(f"{pretty}: {s[key]['mean']:.8g} +/- {s[key]['std']:.8g}\n")
            f.write('\nLaTeX row:\n')
            f.write(
                f"& {label}, $K={K}$\n"
                f"& {s['tau0']['mean']:.6f} $\\pm$ {s['tau0']['std']:.6f}\n"
                f"& {s['tau_mid']['mean']:.6f} $\\pm$ {s['tau_mid']['std']:.6f}\n"
                f"& {s['tau_final']['mean']:.6f} $\\pm$ {s['tau_final']['std']:.6f}\n"
                f"& {s['rollout_mean']['mean']:.6f} $\\pm$ {s['rollout_mean']['std']:.6f} \\\\\n\n"
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt_path = resolve_path(args.ckpt)
    data_path = resolve_path(args.dataset)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt['args']; ms = config['ms']; ds = config['ds']
    vids = np.load(data_path, mmap_mode='r')
    if vids.ndim != 5:
        raise ValueError(f'Expected [N,T,C,H,W], got {vids.shape}')
    n_total, t_total, _, H, W = vids.shape
    required_states = args.num_transitions + 1
    if t_total < required_states:
        raise ValueError(f'Dataset has {t_total} states; need {required_states}')

    if args.all_trajs:
        traj_indices = list(range(n_total))
    elif args.traj_indices is None:
        traj_indices = [0, n_total//2, n_total-1]
    else:
        traj_indices = args.traj_indices

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]
    c_fwd = torch.tensor([1.0, 0.0], dtype=torch.float32)
    c_bwd = torch.tensor([0.0, 1.0], dtype=torch.float32)

    print('='*80)
    print('Checkpoint:', ckpt_path)
    print('Dataset:', data_path)
    print('Model:', ms, 'Dataset name:', ds)
    print('Device:', device)
    print('Trajectories:', len(traj_indices), 'Batch size:', args.batch_size)
    print('K:', args.K, 'Transitions:', args.num_transitions, 'States:', required_states)
    print('Greedy cycles/trajectory:', args.K * args.num_transitions)
    print('Greedy directional samples/trajectory:', 2 * args.K * args.num_transitions)
    print('='*80)

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=str(ckpt_path), img_size=(H, W), device=device
    )

    rows = []
    details = {'forward_best': [], 'stepwise_oracle': [], 'greedy_cyclic': []}

    for start in range(0, len(traj_indices), args.batch_size):
        batch_indices = traj_indices[start:start+args.batch_size]
        print(f'\nBatch trajectories: {batch_indices}', flush=True)
        real = torch.from_numpy(np.asarray(vids[batch_indices, :required_states])).float()

        results = {
            'forward_best': forward_best(model, real, args.K, args.num_transitions, args.num_ode_steps, c_fwd),
            'stepwise_oracle': stepwise_oracle(model, real, args.K, args.num_transitions, args.num_ode_steps, c_fwd),
            'greedy_cyclic': greedy_cyclic(model, real, args.K, args.num_transitions, args.num_ode_steps, c_fwd, c_bwd),
        }

        for method, result in results.items():
            m = metrics(result['rollout'], real)
            for local_i, traj_i in enumerate(batch_indices):
                rows.append({
                    'model': f'flowformer_{ms}_{ds}',
                    'dataset': ds,
                    'method': method,
                    'trajectory_index': traj_i,
                    'tau0': float(m['tau0'][local_i]),
                    'tau_mid': float(m['tau_mid'][local_i]),
                    'tau_final': float(m['tau_final'][local_i]),
                    'rollout_mean': float(m['rollout_mean'][local_i]),
                    'K': args.K,
                    'num_transitions': args.num_transitions,
                })
            details[method].append({
                'trajectory_indices': batch_indices,
                **{k: v.cpu() for k, v in result.items() if torch.is_tensor(v) and k != 'rollout'}
            })

    method_metrics = {}
    for method in ['forward_best','stepwise_oracle','greedy_cyclic']:
        subset = [r for r in rows if r['method'] == method]
        method_metrics[method] = {
            key: torch.tensor([r[key] for r in subset], dtype=torch.float64)
            for key in ['tau0','tau_mid','tau_final','rollout_mean']
        }
    summary = summarize(method_metrics)

    print('\nDATASET-LEVEL SUMMARY')
    for method in ['forward_best','stepwise_oracle','greedy_cyclic']:
        print('\n', method)
        for key in ['tau0','tau_mid','tau_final','rollout_mean']:
            s = summary[method][key]
            print(f"  {key}: {s['mean']:.8g} +/- {s['std']:.8g}")

    tag = f'flowformer_{ms}_{ds}'
    suffix = 'all_trajs' if args.all_trajs else 'selected_trajs'
    base = f'greedy_oracle_{tag}_K{args.K}_T{args.num_transitions}_{suffix}'
    csv_path = out_dir / f'{base}.csv'
    pt_path = out_dir / f'{base}.pt'
    txt_path = out_dir / f'{base}.txt'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    torch.save({
        'checkpoint': str(ckpt_path),
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_args': config,
        'dataset': str(data_path),
        'trajectory_indices': traj_indices,
        'K': args.K,
        'num_transitions': args.num_transitions,
        'num_states': required_states,
        'num_ode_steps': args.num_ode_steps,
        'rows': rows,
        'summary': summary,
        'selection_details': details,
    }, pt_path)

    write_txt(txt_path, tag, args.K, args.num_transitions, summary)
    print('\nSaved:')
    print(csv_path); print(pt_path); print(txt_path)


if __name__ == '__main__':
    main()
