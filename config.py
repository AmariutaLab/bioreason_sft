"""Config loading. Configs live in YAML; source code never changes for an experiment.

    configs/<stage>/<name>.yaml     e.g. configs/sft/qwen8b.yaml
    prompts/<kind>/<name>.yaml      e.g. prompts/teacher/terse.yaml

Inheritance: a config may declare `extends: default` and override only the keys
it changes. Deep-merged, so `lora: {r: 64}` keeps the rest of the lora block.

    # configs/sft/bigger_lora.yaml
    extends: default
    lora:
      r: 64          # everything else inherited from default.yaml

Dot access for readability: cfg.lora.r  (and cfg["lora"]["r"] still works).
"""
from __future__ import annotations

import copy
import re
import json
from pathlib import Path

import yaml

import paths


class Cfg(dict):
    """dict with attribute access; nested dicts wrapped on the way out."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(
                f"no config key '{k}'. available: {sorted(self.keys())}")
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get_path(self, dotted, default=None):
        """cfg.get_path('lora.r', 16)"""
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def pretty(self) -> str:
        return json.dumps(self, indent=2, default=str, sort_keys=True)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(p: Path) -> dict:
    paths.require(p, f"expected a YAML config at {p}")
    return yaml.safe_load(p.read_text()) or {}


def _resolve(directory: Path, name: str, _seen=None) -> dict:
    """Load name.yaml, applying `extends` chains (depth-first)."""
    _seen = _seen or []
    if name in _seen:
        raise ValueError(f"circular extends: {' -> '.join(_seen + [name])}")
    raw = _load_yaml(directory / f"{name}.yaml")
    parent = raw.pop("extends", None)
    if parent:
        base = _resolve(directory, parent, _seen + [name])
        return _deep_merge(base, raw)
    return raw


def load_config(stage: str, name: str = "default", overrides: dict | None = None) -> Cfg:
    """configs/<stage>/<name>.yaml, plus optional CLI overrides."""
    cfg = _resolve(paths.CONFIGS / stage, name)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_stage"], cfg["_name"] = stage, name
    return Cfg(cfg)


def load_prompts(spec: str) -> Cfg:
    """prompts/<kind>/<name>.yaml, given spec 'kind/name' (e.g. 'teacher/default')."""
    kind, _, name = spec.partition("/")
    if not name:
        raise ValueError(f"prompt spec must be 'kind/name', got '{spec}'")
    return Cfg(_resolve(paths.PROMPTS / kind, name))


_SCI_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce(val: str):
    """YAML-parse a CLI value, with a fix for YAML 1.1 scientific notation.

    YAML 1.1 (what PyYAML implements) requires a decimal point AND a signed
    exponent for floats, so `2.0e-4` parses but `1e-5` comes back as the STRING
    '1e-5' — which would silently reach the optimizer as a string. Coerce it.
    """
    parsed = yaml.safe_load(val)
    if isinstance(parsed, str) and _SCI_RE.match(parsed.strip()):
        return float(parsed)
    return parsed


def parse_overrides(pairs) -> dict:
    """--set lora.r=64 --set train.lr=1e-5  ->  nested dict.

    Values are YAML-parsed, so ints/floats/bools/lists all work:
        --set train.epochs=5
        --set train.lr=1e-5                 (coerced to float, see _coerce)
        --set lora.target_modules='[q_proj, v_proj]'
        --set model.load_in_4bit=false
    """
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, _, val = pair.partition("=")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return out


def add_config_args(ap, stage: str):
    """Standard config flags for every script."""
    ap.add_argument("--config", default="default",
                    help=f"configs/{stage}/<name>.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a config key, e.g. --set lora.r=64")


def resolve(ap_args, stage: str) -> Cfg:
    cfg = load_config(stage, ap_args.config, parse_overrides(ap_args.set))
    print(f"[config] {stage}/{ap_args.config}"
          + (f" + {ap_args.set}" if ap_args.set else ""))
    return cfg


def snapshot(cfg: Cfg, out_dir: Path, extra: dict | None = None):
    """Write the FULLY RESOLVED config next to the run's outputs.

    This is what makes a run reproducible: the yaml on disk may change later,
    but this snapshot records exactly what produced these artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    if extra:
        payload["_runtime"] = extra
    (out_dir / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, default=str))
