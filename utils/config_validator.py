from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

# Dipendenza esterna: jsonschema
from jsonschema import Draft7Validator


class ConfigError(Exception):
    """Errore custom per configurazioni non valide."""
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e


def _validate_with_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)

    if errors:
        msg_lines = ["Config validation failed (schema errors):"]
        for err in errors:
            # Costruisce un path leggibile tipo: training.batch_size
            loc = ".".join([str(x) for x in err.path]) if err.path else "(root)"
            msg_lines.append(f" - {loc}: {err.message}")
        raise ConfigError("\n".join(msg_lines))


def _validate_split_sum(config: Dict[str, Any], tol: float = 1e-6) -> None:
    split = config["dataset"]["split"]
    s = float(split["train"]) + float(split["val"]) + float(split["test"])
    if abs(s - 1.0) > tol:
        raise ConfigError(
            f"Split values must sum to 1.0, got {s:.6f}. "
            f"(train={split['train']}, val={split['val']}, test={split['test']})"
        )


def _validate_classes(config: Dict[str, Any]) -> None:
    classes = config["dataset"]["classes"]
    num_classes = config["dataset"]["num_classes"]
    if len(classes) != num_classes:
        raise ConfigError(
            f"dataset.num_classes={num_classes} but dataset.classes has length {len(classes)}"
        )


def _validate_paths_exist(config: Dict[str, Any], base_dir: Path) -> None:
    """
    Controllo opzionale che le cartelle dataset esistano.
    Utile per scoprire subito errori di path nel config.
    """
    train_root = base_dir / config["dataset"]["paths"]["train_root"]
    test_root = base_dir / config["dataset"]["paths"]["test_root"]

    if not train_root.exists():
        raise ConfigError(f"Dataset train_root does not exist: {train_root}")
    if not test_root.exists():
        raise ConfigError(f"Dataset test_root does not exist: {test_root}")


def load_and_validate_config(
    config_path: str | Path,
    schema_path: str | Path,
    *,
    base_dir: Path | None = None,
    check_paths: bool = True
) -> Tuple[Dict[str, Any], Path]:
    """
    Carica e valida la configurazione.

    Args:
        config_path: path al config.json
        schema_path: path al config_schema.json
        base_dir: root del progetto (se None: calcolata come parent del config)
        check_paths: se True, verifica esistenza cartelle train/test

    Returns:
        (config_dict, base_dir)
    """
    config_path = Path(config_path).resolve()
    schema_path = Path(schema_path).resolve()

    if base_dir is None:
        # assume config è in <root>/configs/config.json
        base_dir = config_path.parent.parent

    config = _read_json(config_path)
    schema = _read_json(schema_path)

    # 1) validazione schema
    _validate_with_schema(config, schema)

    # 2) validazioni logiche extra
    _validate_split_sum(config)
    _validate_classes(config)

    # 3) validazione path (opzionale)
    if check_paths:
        _validate_paths_exist(config, base_dir)

    return config, base_dir
