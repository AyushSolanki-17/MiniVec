#!/usr/bin/env python3
"""Run and calibrate the public DARTH reference HNSW implementation."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts import datasets
from scripts.experiments import e2_estimator_v2 as e2


DATASET_CONFIG = {
    "sift100k": {
        "name": "SIFT100M", "source": "sift-128-euclidean", "M": 32,
        "efc": 500, "efs": 500, "ipi": 782, "mpi": 156,
        "model": "SIFT100M_M32_efC500_efS500_s10000_k10_nestim100_li1_all_feats.txt",
        "base_file": "base.100M.fvecs",
    },
    "glove100k": {
        "name": "GLOVE100", "source": "glove-100-angular", "M": 16,
        "efc": 500, "efs": 500, "ipi": 212, "mpi": 42,
        "model": "GLOVE100_M16_efC500_efS500_s10000_k10_nestim100_li1_all_feats.txt",
        "base_file": "base.1183514.fvecs",
    },
}
# Covers DARTH's reference range and E1 v2's contract baseline range.
EF_GRID = [40, 60, 80, 120, 160, 240, 320, 480, 640, 1024]
TARGET = 0.95


def query_splits(rows, seed=42):
    """Return train/calibration/test query ID sets using the E2 v2 split."""
    train_rows, cal_rows, test_rows = e2.split_queries(rows, seed)
    def get_ids(group):
        return {int(row["query_id"]) for row in group}
    train_ids, cal_ids, test_ids = map(get_ids, (train_rows, cal_rows, test_rows))
    if train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids:
        raise ValueError("query partitions overlap")
    return train_ids, cal_ids, test_ids


def write_fvecs(path, vectors):
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    dim = np.full((len(vectors), 1), vectors.shape[1], dtype=np.int32)
    with path.open("wb") as stream:
        np.concatenate((dim.view(np.float32), vectors), axis=1).tofile(stream)


def write_ivecs(path, indices):
    indices = np.asarray(indices, dtype=np.int32, order="C")
    dim = np.full((len(indices), 1), indices.shape[1], dtype=np.int32)
    with path.open("wb") as stream:
        np.concatenate((dim, indices), axis=1).tofile(stream)


def exact_neighbors(vectors, queries, k, batch=64):
    """Compute exact squared-L2 neighbors without importing the local ANN engine."""
    result = np.empty((len(queries), k), dtype=np.int32)
    norms = np.einsum("ij,ij->i", vectors, vectors)
    for start in range(0, len(queries), batch):
        query_batch = queries[start:start + batch]
        distances = (
            np.einsum("ij,ij->i", query_batch, query_batch)[:, None]
            + norms[None, :]
            - 2 * query_batch @ vectors.T
        )
        result[start:start + len(query_batch)] = np.argpartition(
            distances, k - 1, axis=1
        )[:, :k]
    return result


def prepare_reference_data(dataset_key, signal_csv, output_root, limit_train=100_000):
    """Write first-100k E1-v2 collection data in the reference fvecs format."""
    config = DATASET_CONFIG[dataset_key]
    with signal_csv.open(newline="", encoding="utf-8") as stream:
        signal_rows = list(csv.DictReader(stream))
    _, calibration_ids, test_ids = query_splits(signal_rows)
    vectors, queries, _, _ = datasets.load(config["source"])
    vectors = vectors[:limit_train]
    vectors = np.asarray(vectors, dtype=np.float32)
    if config["source"].endswith("angular"):
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        queries = np.asarray(queries, dtype=np.float32)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    else:
        queries = np.asarray(queries, dtype=np.float32)

    shared_dir = output_root / "shared" / config["name"]
    shared_dir.mkdir(parents=True, exist_ok=True)
    write_fvecs(shared_dir / config["base_file"], vectors)
    split_rows = {}
    for split, query_ids in (("calibration", calibration_ids), ("test", test_ids)):
        ids = sorted(query_ids)
        query_vectors = queries[ids]
        truth = np.asarray(exact_neighbors(vectors, query_vectors, 10), dtype=np.int32)
        truth_dist = np.empty_like(truth, dtype=np.float32)
        for row_idx, (query, neighbors) in enumerate(zip(query_vectors, truth)):
            diff = vectors[neighbors] - query
            truth_dist[row_idx] = np.einsum("ij,ij->i", diff, diff)

        split_dir = output_root / split / config["name"]
        split_dir.mkdir(parents=True, exist_ok=True)
        base_link = split_dir / config["base_file"]
        if base_link.exists() or base_link.is_symlink():
            base_link.unlink()
        base_link.symlink_to((shared_dir / config["base_file"]).resolve())
        write_fvecs(split_dir / "query.10K.fvecs", query_vectors)
        write_ivecs(split_dir / "query.groundtruth.10K.k1000.ivecs", truth)
        write_fvecs(split_dir / "query.groundtruth.10K.k1000.fvecs", truth_dist)
        split_rows[split] = len(ids)
    return {
        "dataset": dataset_key,
        "reference_name": config["name"],
        "indexed_vectors": len(vectors),
        "calibration_queries": split_rows["calibration"],
        "test_queries": split_rows["test"],
        "M": config["M"],
        "ef_construction": config["efc"],
        "ef_search": config["efs"],
        "metric": "squared L2; normalized vectors preserve angular ranking",
    }


def read_log(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty DARTH log: {path}")
    return rows


def risk_calibrated_claims(calibration_rows, test_rows, target=TARGET):
    """Apply E2 v2's fixed threshold-selection rule to DARTH's recall scores."""
    cal_scores = [float(row["r_predicted"]) for row in calibration_rows]
    normalized = [
        {**row, "query_id": row["qid"], "ef_search": "1", "recall_at_k": row["r_actual"]}
        for row in calibration_rows
    ]
    risk = e2.choose_threshold(normalized, cal_scores, target)
    threshold = risk["threshold"]
    certified_rows = [
        {**row, "certified_score": float(
            threshold is not None and float(row["r_predicted"]) >= threshold
        )}
        for row in test_rows
    ]
    return risk, certified_rows


def declared_claim_certificate(calibration_rows, target=TARGET):
    """Check the fixed DARTH declaration threshold with E2 v2's risk bound."""
    declared = [row for row in calibration_rows
                if float(row["r_predicted"]) >= target]
    false_claims = sum(float(row["r_actual"]) < target for row in declared)
    upper_bound = e2.false_claim_upper_bound(
        false_claims, len(declared), e2.DELTA / len(e2.THRESHOLDS)
    ) if declared else None
    return {
        "claims": len(declared),
        "false_claims": false_claims,
        "risk_upper": upper_bound,
        "certified": upper_bound is not None and upper_bound <= e2.RISK_LIMIT,
    }


def claim_metrics(rows, score_field, target=TARGET, threshold=TARGET):
    claims = [row for row in rows if float(row[score_field]) >= threshold]
    hits = sum(float(row["r_actual"]) >= target for row in claims)
    low, high = e2.wilson_interval(hits, len(claims))
    return {
        "claims": len(claims),
        "claim_share": len(claims) / len(rows) if rows else float("nan"),
        "precision": hits / len(claims) if claims else float("nan"),
        "precision_ci_low": low,
        "precision_ci_high": high,
        "false_claims": len(claims) - hits,
        "claim_distance_calls_mean": (
            float(np.mean([int(row["dists"]) for row in claims])) if claims else float("nan")
        ),
        "claim_latency_ms_mean": (
            float(np.mean([float(row["elaps_ms"]) for row in claims])) if claims else float("nan")
        ),
    }


def contract_baseline(calibration_by_ef, test_by_ef, target=TARGET, min_share=0.95):
    """Select the smallest fixed ef meeting query coverage on calibration queries."""
    eligible = []
    for ef, rows in sorted(calibration_by_ef.items(), key=lambda pair: int(pair[0])):
        share = float(np.mean([float(row["r"]) >= target for row in rows]))
        if share >= min_share:
            eligible.append((int(ef), share))
    if not eligible:
        return None
    ef, cal_share = eligible[0]
    test_rows = test_by_ef[ef]
    return {
        "ef": ef,
        "calibration_met_share": cal_share,
        "test_met_share": float(np.mean([float(row["r"]) >= target for row in test_rows])),
        "distance_calls": sum(int(row["dists"]) for row in test_rows),
        "distance_calls_mean": float(np.mean([int(row["dists"]) for row in test_rows])),
        "latency_ms_mean": float(np.mean([float(row["elaps_ms"]) for row in test_rows])),
    }


def summarize(dataset_key, calibration_darth, test_darth, calibration_fixed, test_fixed):
    direct = claim_metrics(test_darth, "r_predicted")
    risk, certified_rows = risk_calibrated_claims(calibration_darth, test_darth)
    declared_risk = declared_claim_certificate(calibration_darth)
    certified = claim_metrics(certified_rows, "certified_score", threshold=0.5)
    baseline = contract_baseline(calibration_fixed, test_fixed)
    all_calls = sum(int(row["dists"]) for row in test_darth)
    all_latency = float(np.mean([float(row["elaps_ms"]) for row in test_darth]))
    return {
        "dataset": dataset_key,
        "target": TARGET,
        "darth_claims": direct["claims"],
        "test_queries": len(test_darth),
        "darth_claim_share": direct["claim_share"],
        "darth_precision": direct["precision"],
        "darth_precision_ci_low": direct["precision_ci_low"],
        "darth_precision_ci_high": direct["precision_ci_high"],
        "darth_false_claims": direct["false_claims"],
        "darth_calls_mean": all_calls / len(test_darth),
        "darth_calls_total": all_calls,
        "darth_latency_ms_mean": all_latency,
        "risk_threshold": risk["threshold"],
        "risk_calibration_upper": risk["upper_bound"],
        "risk_calibration_claims": risk["calibration_claims"],
        "risk_calibration_false_claims": risk["calibration_false_claims"],
        "declared_calibration_claims": declared_risk["claims"],
        "declared_calibration_false_claims": declared_risk["false_claims"],
        "declared_calibration_risk_upper": declared_risk["risk_upper"],
        "declared_claim_certified": declared_risk["certified"],
        "certified_claims": certified["claims"],
        "certified_claim_share": certified["claim_share"],
        "certified_precision": certified["precision"],
        "certified_precision_ci_low": certified["precision_ci_low"],
        "certified_precision_ci_high": certified["precision_ci_high"],
        "certified_false_claims": certified["false_claims"],
        "baseline_ef": baseline["ef"] if baseline else "",
        "baseline_calibration_met_share": baseline["calibration_met_share"] if baseline else "",
        "baseline_test_met_share": baseline["test_met_share"] if baseline else "",
        "baseline_calls_mean": baseline["distance_calls_mean"] if baseline else "",
        "baseline_latency_ms_mean": baseline["latency_ms_mean"] if baseline else "",
        "baseline_over_darth_calls_ratio": (
            baseline["distance_calls"] / all_calls if baseline and all_calls else float("nan")
        ),
        "baseline_over_darth_latency_ratio": (
            baseline["latency_ms_mean"] / all_latency if baseline and all_latency else float("nan")
        ),
    }


def run_driver(driver, config, dataset_root, output_path, mode, query_count,
               index_path, ef_search, model_path=None, save_index=False):
    command = [
        str(driver), "--dataset", config["name"], "--M", str(config["M"]),
        "--efConstruction", str(config["efc"]), "--efSearch", str(ef_search),
        "--query-num", str(query_count), "--k", "10", "--output", str(output_path),
        "--mode", mode, "--index-filepath", str(index_path),
        "--dataset-dir-prefix", f"{dataset_root}/", "--query-type", "testing",
    ]
    if save_index:
        command.append("--save-index")
    if model_path is not None:
        command.extend([
            "--target-recall", str(TARGET), "--initial-prediction-interval", str(config["ipi"]),
            "--min-prediction-interval", str(config["mpi"]), "--predictor-model-path", str(model_path),
        ])
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def run_dataset(dataset_key, args):
    config = DATASET_CONFIG[dataset_key]
    result_dir = args.output_dir / dataset_key
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_reference_data(
        dataset_key, args.signals_dir / f"{dataset_key}.csv", args.data_root,
    )
    config["model_path"] = args.darth_repo / "predictor_models" / "darth" / config["model"]
    index_path = result_dir / f"{config['name']}.index"
    test_count = metadata["test_queries"]
    run_driver(
        args.driver, config, args.data_root / "test", result_dir / "index-build.csv",
        "no-early-stop", test_count, index_path, config["efs"],
        save_index=True,
    )

    darth_logs = {}
    fixed_logs = {"calibration": {}, "test": {}}
    for split in ("calibration", "test"):
        split_root = args.data_root / split
        count = metadata[f"{split}_queries"]
        darth_path = result_dir / f"darth-{split}.csv"
        run_driver(
            args.driver, config, split_root, darth_path, "early-stop-testing",
            count, index_path, config["efs"], config["model_path"],
        )
        darth_logs[split] = read_log(darth_path)
        for ef in EF_GRID:
            baseline_path = result_dir / f"fixed-ef{ef}-{split}.csv"
            run_driver(
                args.driver, config, split_root, baseline_path, "no-early-stop",
                count, index_path, ef,
            )
            fixed_logs[split][ef] = read_log(baseline_path)
    result = summarize(
        dataset_key,
        darth_logs["calibration"], darth_logs["test"],
        fixed_logs["calibration"], fixed_logs["test"],
    )
    result.update(metadata)
    (result_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--darth-repo", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--signals-dir", type=Path, default=Path("data/signals/v2"))
    parser.add_argument("--data-root", type=Path, default=Path("data/darth-reference"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench/darth-reference"))
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), nargs="+",
                        default=["sift100k", "glove100k"])
    args = parser.parse_args()
    summaries = [run_dataset(dataset, args) for dataset in args.dataset]
    summary_path = args.output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for result in summaries:
        print(
            f"{result['dataset']} DARTH claims={result['darth_claims']}/{result['test_queries']} "
            f"precision={result['darth_precision']:.3f} "
            f"CI=[{result['darth_precision_ci_low']:.3f},{result['darth_precision_ci_high']:.3f}] "
            f"risk-certified={result['certified_claims']}/{result['test_queries']} "
            f"fixed-ef={result['baseline_ef'] or 'n/a'} "
            f"baseline-over-DARTH-calls={result['baseline_over_darth_calls_ratio']:.3f} "
            f"baseline-over-DARTH-latency={result['baseline_over_darth_latency_ratio']:.3f}"
        )
    print(f"wrote {len(summaries)} summaries to {summary_path}")


if __name__ == "__main__":
    main()
