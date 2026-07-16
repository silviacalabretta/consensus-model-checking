#!/usr/bin/env python3

import csv
import math
from dataclasses import dataclass
from pathlib import Path


CONFIG_KEY_FIELDS = ("model", "N", "Za", "Zb", "C", "t", "h", "qa", "qb")


@dataclass(frozen=True)
class GridConfig:
    model: str
    N: int
    Za: int
    Zb: int
    C: int
    t: int
    h: int
    qa: float = 1.05
    qb: float = 0.95

    @property
    def disruptor_count(self) -> int:
        return self.Za + self.Zb if self.model.endswith("_zealots") else self.C

    @property
    def disruptor_fraction(self) -> float:
        return self.disruptor_count / self.N

    def key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, field)) for field in CONFIG_KEY_FIELDS)

    def to_model_params(self):
        from src.types import ModelParams
        return ModelParams(
            model_name=self.model,
            N=self.N,
            Za=self.Za,
            Zb=self.Zb,
            C=self.C,
            t=self.t,
            h=self.h,
            qa=self.qa,
            qb=self.qb,
        )


def even_disruptor_counts(N: int, d_fractions: list[float]) -> list[int]:
    return [2 * math.floor(d * N / 2 + 0.5) for d in d_fractions]


def generate_configs(
    models: list[str],
    N_values: list[int],
    D_fractions: list[float],
    t: int,
    h: int,
    qa: float = 1.05,
    qb: float = 0.95,
) -> list[GridConfig]:
    configs: list[GridConfig] = []
    for N in sorted(set(N_values)):
        for disruptors in even_disruptor_counts(N, D_fractions):
            for model in models:
                if model.endswith("_zealots"):
                    Za = Zb = disruptors // 2
                    C = 0
                else:
                    Za = Zb = 0
                    C = disruptors
                configs.append(
                    GridConfig(
                        model=model, N=N, Za=Za, Zb=Zb, C=C,
                        t=t, h=h, qa=qa, qb=qb,
                    )
                )
    return configs


def read_existing_csv(output_path: Path) -> tuple[list[str] | None, set[tuple[str, ...]]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None, set()
    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames) if reader.fieldnames else None
        keys = {
            tuple(row[field] for field in CONFIG_KEY_FIELDS) for row in reader
            if row.get("status") == "ok"
        }
    return fields, keys
