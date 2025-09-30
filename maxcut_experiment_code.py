import os, math, time, random, hashlib
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def er_sparse_graph(n:int, d_avg:float, seed:int):
    rng = random.Random(seed)
    p = d_avg / max(1, n-1)
    if p > 1.0: p = 1.0
    adj = {i: [] for i in range(1, n+1)}
    edges = []
    for u in range(1, n+1):
        for v in range(u+1, n+1):
            if rng.random() <= p:
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u,v))
    return adj, edges

def cut_value_from_side(adj:Dict[int, List[int]], side:List[int]) -> int:
    val = 0
    for u, nbrs in adj.items():
        for v in nbrs:
            if u < v and (side[u] ^ side[v]):
                val += 1
    return val

def local_search_1flip(adj:Dict[int, List[int]], seed:int, max_flips_factor:float=2.0, max_passes:int=2000):
    n = len(adj)
    rng = random.Random(seed)
    side = [0]*(n+1)
    side[1] = 1
    for v in range(2, n+1):
        side[v] = rng.randint(0,1)

    deg = [0]*(n+1)
    for v in range(1, n+1):
        deg[v] = len(adj[v])

    def gain(v:int) -> int:
        opp = 0
        sv = side[v]
        for u in adj[v]:
            if side[u] != sv:
                opp += 1
        same = deg[v] - opp
        return same - opp

    m = sum(deg[1:]) // 2
    max_flips = int(max_flips_factor * max(1, m))

    flips = 0
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for v in range(1, n+1):
            g = gain(v)
            if g > 0:
                side[v] ^= 1
                flips += 1
                improved = True
                if flips >= max_flips:
                    improved = False
                break

    cut = cut_value_from_side(adj, side)
    return cut, flips, passes

def run_experiment(sizes, trials_per_size=15, d_avg=8.0, master_seed=20250813, out_dir="."):
    rows = []
    for n in sizes:
        for t in range(trials_per_size):
            seed_graph = master_seed*1000003 + 7919*n + 31337*int(round(d_avg*1000)) + t
            adj, edges = er_sparse_graph(n, d_avg, seed_graph)
            data = "".join(f"{min(u,v)} {max(u,v)}\n" for (u,v) in sorted(edges)).encode("utf-8")
            h = hashlib.sha256(data).hexdigest()
            seed_ls = int(h[:16], 16) ^ seed_graph

            t0 = time.perf_counter()
            cut, flips, passes = local_search_1flip(adj, seed_ls)
            dt = time.perf_counter() - t0

            rows.append({
                "n": n, "avg_deg_target": d_avg, "m": len(edges), "cut": cut,
                "flips": flips, "passes": passes, "time_sec": dt,
                "seed_graph": seed_graph, "seed_local": seed_ls, "edges_sha256": h,
            })

    df = pd.DataFrame(rows)
    trials_csv = os.path.join(out_dir, "maxcut_perf_trials.csv")
    df.to_csv(trials_csv, index=False)

    summary = df.groupby("n").agg(
        m_mean=("m","mean"),
        time_mean=("time_sec","mean"),
        time_std=("time_sec","std"),
        trials=("time_sec","count")
    ).reset_index()

    z90 = 1.645
    summary["ci_half"] = z90 * summary["time_std"] / np.sqrt(summary["trials"])
    summary["b_over_a"] = summary["ci_half"] / summary["time_mean"]
    summary["ci_low"] = summary["time_mean"] - summary["ci_half"]
    summary["ci_high"] = summary["time_mean"] + summary["ci_half"]

    logn = np.log(summary["n"].values)
    logt = np.log(summary["time_mean"].values)
    A = np.vstack([np.ones_like(logn), logn]).T
    coef, *_ = np.linalg.lstsq(A, logt, rcond=None)
    logA, alpha = coef[0], coef[1]
    A_hat = math.exp(logA)
    summary["fit_time_pred"] = A_hat * (summary["n"].values ** alpha)

    summary_csv = os.path.join(out_dir, "maxcut_perf_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig1 = os.path.join(out_dir, "runtime_vs_n.png")
    fig2 = os.path.join(out_dir, "runtime_loglog.png")

    plt.figure()
    plt.errorbar(summary["n"], summary["time_mean"], yerr=summary["ci_half"], fmt='o', capsize=3)
    plt.plot(summary["n"], summary["fit_time_pred"], marker='x')
    plt.xlabel("n (number of vertices)")
    plt.ylabel("Local-search time (seconds)")
    plt.title("Max-Cut 1-flip local search: runtime vs n (avg deg ≈ 8)")
    plt.savefig(fig1, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.loglog(summary["n"], summary["time_mean"], marker='o', linestyle='')
    plt.loglog(summary["n"], summary["fit_time_pred"], marker='x', linestyle='-')
    plt.xlabel("n (log scale)")
    plt.ylabel("Local-search time (seconds, log scale)")
    plt.title("Log–log fit: time ≈ A * n^alpha")
    plt.savefig(fig2, bbox_inches="tight")
    plt.close()

    return trials_csv, summary_csv, fig1, fig2, A_hat, alpha

if __name__ == "__main__":
    sizes = [100, 200, 400, 800, 1600, 2400]
    run_experiment(sizes, trials_per_size=15, d_avg=8.0, master_seed=20250813, out_dir=".")
