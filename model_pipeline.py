"""Train a spatial CNN to forecast the next hour of NYC complaint counts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

DATASET = "ehong/nypd_complaint_data_historic_ds"
MANIFEST = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"
COLUMNS = ["CMPLNT_FR_DT", "CMPLNT_FR_TM", "Latitude", "Longitude"]
BOUNDS = {"west": -74.27, "east": -73.69, "south": 40.48, "north": 40.93}
SIZE = 12
HOURS_PER_WEEK = 168
DASHBOARD_DAYS = 28


def fetch_shards(directory: Path) -> list[Path]:
    """Reuse complete local files or download the Hub's Parquet conversion."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        with urlopen(MANIFEST, timeout=30) as response:
            items = json.load(response)["parquet_files"]
        manifest.write_text(json.dumps(items), encoding="utf-8")
    else:
        items = json.loads(manifest.read_text(encoding="utf-8"))
    items = [item for item in items if item["split"] == "train"]
    if not items:
        raise ValueError("Hugging Face did not return any train shards")

    def fetch(item: dict) -> Path:
        path = directory / item["filename"]
        if path.exists() and path.stat().st_size == item["size"]:
            return path
        partial = path.with_suffix(".download")
        print(f"Downloading {path.name}", flush=True)
        with urlopen(item["url"] + "?download=true", timeout=60) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        if partial.stat().st_size != item["size"]:
            raise OSError(f"Incomplete download: {path.name}")
        partial.replace(path)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, items))


def aggregate(shards: list[Path], start: str, end: str) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
    index = pd.date_range(start, end, freq="h", inclusive="left")
    if len(index) < 4 * HOURS_PER_WEEK:
        raise ValueError("Choose a date range of at least four weeks")
    counts = np.zeros((len(index), SIZE, SIZE), dtype=np.int32)
    source_rows = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(columns=COLUMNS, batch_size=100_000):
            frame = batch.to_pandas()
            source_rows += len(frame)
            dates = pd.to_datetime(frame["CMPLNT_FR_DT"], format="mixed", errors="coerce")
            times = frame["CMPLNT_FR_TM"].str.extract(
                r"^\s*([01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\s*$", expand=False
            )
            lat, lon = frame["Latitude"], frame["Longitude"]
            valid = (
                dates.ge(index[0]) & dates.le(index[-1]) & times.notna()
                & lat.gt(BOUNDS["south"]) & lat.le(BOUNDS["north"])
                & lon.ge(BOUNDS["west"]) & lon.lt(BOUNDS["east"])
            )
            hours = ((dates[valid] - index[0]) // pd.Timedelta(hours=1)).to_numpy()
            hours = hours + times[valid].to_numpy(dtype=np.int64)
            rows = ((BOUNDS["north"] - lat[valid].to_numpy(dtype=np.float64))
                    * SIZE / (BOUNDS["north"] - BOUNDS["south"])).astype(int)
            cols = ((lon[valid].to_numpy(dtype=np.float64) - BOUNDS["west"])
                    * SIZE / (BOUNDS["east"] - BOUNDS["west"])).astype(int)
            np.add.at(counts, (hours, rows, cols), 1)
        print(f"Imported {shard.name} ({source_rows:,} rows read)", flush=True)
    if not counts.any():
        raise ValueError("No usable complaints in the selected dates and map bounds")
    # Trim empty time before the first and after the last populated calendar day.
    occupied = np.flatnonzero(counts.sum(axis=(1, 2)))
    first = occupied[0] // 24 * 24
    last = (occupied[-1] // 24 + 1) * 24
    return counts[first:last], index[first:last], source_rows


class ForecastCNN(nn.Module):
    def __init__(self, steps: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(steps + 7, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, history: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        calendar = calendar[:, :, None, None].expand(-1, -1, *history.shape[-2:])
        inputs = torch.cat((history, calendar), dim=1).contiguous(memory_format=torch.channels_last)
        return F.softplus(self.layers(inputs)[:, 0])


def calendar_features(index: pd.DatetimeIndex) -> torch.Tensor:
    hour = index.hour.to_numpy()
    week = index.day_of_week.to_numpy()
    daily = 2 * np.pi * hour / 24
    weekly = 2 * np.pi * week / 7
    values = np.stack(
        (np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly), hour == 0, hour == 12),
        axis=1,
    )
    return torch.tensor(values, dtype=torch.float32)


def train_and_predict(counts, index, steps, epochs, batch_size, evaluate_from, patience):
    cutoff = int(len(counts) * .8)
    first = max(steps, HOURS_PER_WEEK)
    if cutoff <= first or cutoff >= len(counts):
        raise ValueError("Too little date coverage for training and a chronological holdout")
    mask = torch.from_numpy(counts[:cutoff].sum(axis=0) > 0)
    raw = torch.from_numpy(counts).to(torch.float32)
    logged = raw.log1p()
    calendar = calendar_features(index)
    offsets = torch.arange(-steps, 0)

    def batches(start, end, shuffle):
        return DataLoader(torch.arange(start, end), batch_size=batch_size, shuffle=shuffle)

    def predict(model, targets):
        history = logged[targets[:, None] + offsets]
        previous_week = logged[targets - HOURS_PER_WEEK, None]
        return model(torch.cat((history, previous_week), dim=1), calendar[targets])

    def fit(end, max_epochs, validation_mask=None):
        torch.manual_seed(42)
        model = ForecastCNN(steps).to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        train_mask = mask if validation_mask is None else validation_mask
        train_batches = batches(first, end, True)
        best_epoch, best_mae = max_epochs, float("inf")
        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for targets in train_batches:
                optimizer.zero_grad(set_to_none=True)
                forecast = predict(model, targets)[:, train_mask]
                loss = F.poisson_nll_loss(forecast, raw[targets][:, train_mask], log_input=False)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)

            message = f"Epoch {epoch}/{max_epochs} · training loss {total_loss / (end - first):.4f}"
            if validation_mask is None:
                print(message, flush=True)
                continue

            model.eval()
            validation_error = 0.0
            with torch.inference_mode():
                for targets in batches(end, cutoff, False):
                    difference = (predict(model, targets) - raw[targets])[:, validation_mask]
                    validation_error += difference.abs().sum().item()
            mae = validation_error / ((cutoff - end) * int(validation_mask.sum()))
            print(f"{message} · validation MAE {mae:.5f}", flush=True)
            if mae < best_mae - .0001:
                best_epoch, best_mae = epoch, mae
            elif epoch - best_epoch >= patience:
                break
        return model, best_epoch

    chosen_epochs = epochs
    if patience:
        validation_start = int(len(counts) * .7)
        if validation_start <= first or validation_start >= cutoff:
            raise ValueError("Too little date coverage for a chronological validation period")
        validation_mask = torch.from_numpy(counts[:validation_start].sum(axis=0) > 0)
        print(f"Choosing epochs on {index[validation_start]} to {index[cutoff - 1]}", flush=True)
        _, chosen_epochs = fit(validation_start, epochs, validation_mask)
        print(f"Selected {chosen_epochs} epochs. Refitting on the first 80% of hours.", flush=True)
    model, _ = fit(cutoff, chosen_epochs)

    model.eval()
    absolute = squared = baseline_absolute = city_absolute = baseline_city_absolute = 0.0
    shown = [[], [], []]
    display_start = max(cutoff, len(counts) - evaluate_from * 24)
    with torch.inference_mode():
        for targets in batches(cutoff, len(counts), False):
            actual = raw[targets]
            baseline = raw[targets - HOURS_PER_WEEK]
            forecast = predict(model, targets).masked_fill(~mask, 0)
            delta = forecast - actual
            naive_delta = baseline - actual
            absolute += delta[:, mask].abs().sum().item()
            squared += delta[:, mask].square().sum().item()
            baseline_absolute += naive_delta[:, mask].abs().sum().item()
            city_absolute += delta.sum(dim=(1, 2)).abs().sum().item()
            baseline_city_absolute += naive_delta.sum(dim=(1, 2)).abs().sum().item()
            keep = targets >= display_start
            for series, values in zip(shown, (actual, forecast, baseline)):
                series.append(values[keep].numpy())
    hours = len(counts) - cutoff
    active_cells = int(mask.sum())
    cells = hours * active_cells
    metrics = dict(
        mae=absolute / cells,
        rmse=(squared / cells) ** .5,
        baseline_mae=baseline_absolute / cells,
        city_mae=city_absolute / hours,
        baseline_city_mae=baseline_city_absolute / hours,
        hours=hours,
        active_cells=active_cells,
    )
    return model, cutoff, mask, display_start, [np.concatenate(series) for series in shown], metrics, chosen_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/hf"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2024-01-01", help="End date, exclusive")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs, or a maximum with --patience")
    parser.add_argument("--patience", type=int, default=0, help="Stop epoch selection after this many epochs without validation improvement (0 disables)")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch_size, args.threads) < 1:
        parser.error("--epochs, --steps, --batch-size, and --threads must be positive")

    if args.patience < 0:
        parser.error("--patience must be nonnegative")
    torch.set_num_threads(args.threads)
    counts, index, source_rows = aggregate(fetch_shards(args.data), args.start, args.end)
    print(
        f"{int(counts.sum()):,} geocoded complaints · {index[0]} to {index[-1]} NYC local time",
        flush=True,
    )
    model, cutoff, mask, display_start, shown, metrics, trained_epochs = train_and_predict(
        counts, index, args.steps, args.epochs, args.batch_size, DASHBOARD_DAYS, args.patience
    )
    print(
        f"Holdout MAE per active cell/hour · CNN {metrics['mae']:.3f} · "
        f"same hour last week {metrics['baseline_mae']:.3f}",
        flush=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "hourly_counts.npz", counts=counts, timestamps=index.astype(str).to_numpy(dtype=str))
    torch.save(
        {"state_dict": model.state_dict(), "steps": args.steps, "epochs": trained_epochs,
         "bounds": BOUNDS, "grid_size": SIZE, "active_cells": mask},
        args.output / "forecast_cnn.pt",
    )
    actual, forecast, baseline = shown
    payload = dict(
        source=DATASET, source_rows=source_rows, complaints=int(counts.sum()), size=SIZE,
        bounds=BOUNDS, period_start=index[0].isoformat(), period_end=index[-1].isoformat(),
        holdout_start=index[cutoff].isoformat(), timestamps=index[display_start:].strftime("%Y-%m-%dT%H:%M").tolist(),
        steps=args.steps, epochs=trained_epochs, metrics=metrics,
        actual=actual.astype(int).reshape(-1, SIZE * SIZE).tolist(),
        forecast=forecast.round(2).astype(np.float64).round(2).reshape(-1, SIZE * SIZE).tolist(),
        baseline=baseline.astype(int).reshape(-1, SIZE * SIZE).tolist(),
    )
    temp = args.output / "dashboard_data.tmp"
    temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temp.replace(args.output / "dashboard_data.json")
    print(f"Saved model, hourly counts and dashboard data in {args.output.resolve()}/", flush=True)


if __name__ == "__main__":
    main()
