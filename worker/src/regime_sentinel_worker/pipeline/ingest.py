"""Load and validate the frozen thesis-v1 price snapshot.

The loader is deliberately offline: it accepts one checked-in JSON snapshot and
never downloads, imputes, forward-fills or silently drops observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


INSTRUMENTS = ("BTC-USD", "ETH-USD")
SNAPSHOT_COLUMNS = ("date_utc", *INSTRUMENTS)
MODEL_SNAPSHOT_SCHEMA_VERSION = 1
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SnapshotValidationError(ValueError):
    """Raised when a snapshot violates the data contract."""


@dataclass(frozen=True)
class PriceSnapshot:
    """Validated shared-date close prices from one immutable snapshot."""

    path: Path
    snapshot_id: str
    dates: tuple[str, ...]
    prices: dict[str, tuple[float, ...]]
    file_sha256: str
    content_sha256: str
    provenance: dict[str, Any]

    @property
    def row_count(self) -> int:
        return len(self.dates)

    @property
    def observed_start(self) -> str:
        return self.dates[0]

    @property
    def observed_end(self) -> str:
        return self.dates[-1]


@dataclass(frozen=True)
class InputManifest:
    """Minimal reproducibility manifest for the validated input snapshot."""

    snapshot_id: str
    snapshot_path: str
    snapshot_file_sha256: str
    snapshot_content_sha256: str
    instruments: tuple[str, str]
    timezone: str
    observed_start: str
    observed_end: str
    price_observations_per_instrument: int
    log_return_observations_per_instrument: int
    manifest_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "snapshot_id": self.snapshot_id,
            "snapshot_path": self.snapshot_path,
            "snapshot_file_sha256": self.snapshot_file_sha256,
            "snapshot_content_sha256": self.snapshot_content_sha256,
            "instruments": list(self.instruments),
            "timezone": self.timezone,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "price_observations_per_instrument": self.price_observations_per_instrument,
            "log_return_observations_per_instrument": self.log_return_observations_per_instrument,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_content_sha256(rows: list[list[Any]]) -> str:
    canonical = "".join(
        f"{row[0]},{float(row[1]):.6f},{float(row[2]):.6f}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotValidationError(message)


def load_price_snapshot(
    path: Path,
    *,
    expected_snapshot_id: str | None = None,
    expected_row_count: int | None = None,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
) -> PriceSnapshot:
    """Read and validate one snapshot without changing its observations."""

    path = path.resolve()
    if not path.exists():
        raise SnapshotValidationError(f"snapshot does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"cannot read snapshot {path}: {exc}") from exc

    _require(isinstance(payload, dict), "snapshot must be a JSON object")
    _require(
        payload.get("schemaVersion") == MODEL_SNAPSHOT_SCHEMA_VERSION,
        "snapshot schemaVersion must equal 1",
    )
    columns = tuple(payload.get("columns", ()))
    _require(columns == SNAPSHOT_COLUMNS, f"snapshot columns are not canonical: {columns!r}")
    snapshot_id = payload.get("snapshotId")
    _require(isinstance(snapshot_id, str) and snapshot_id, "snapshotId must be non-empty")
    if expected_snapshot_id is not None:
        _require(snapshot_id == expected_snapshot_id, "snapshotId does not match protocol")

    rows = payload.get("rows")
    _require(isinstance(rows, list) and rows, "snapshot rows must be a non-empty array")
    if expected_row_count is not None:
        _require(len(rows) == expected_row_count, "snapshot row count does not match protocol")

    parsed_dates: list[date] = []
    dates: list[str] = []
    prices = {instrument: [] for instrument in INSTRUMENTS}
    seen_dates: set[date] = set()
    for index, row in enumerate(rows):
        _require(
            isinstance(row, list) and len(row) == len(SNAPSHOT_COLUMNS),
            f"snapshot row {index} must contain date, BTC close and ETH close",
        )
        raw_date = row[0]
        _require(
            isinstance(raw_date, str) and _ISO_DATE_RE.fullmatch(raw_date) is not None,
            f"snapshot row {index} date must be a canonical ISO date",
        )
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise SnapshotValidationError(
                f"snapshot row {index} date is not a real ISO date: {raw_date!r}"
            ) from exc
        _require(parsed_date not in seen_dates, f"snapshot contains duplicate date {raw_date}")
        if parsed_dates:
            _require(
                parsed_date - parsed_dates[-1] == timedelta(days=1),
                f"snapshot UTC calendar is not daily before {raw_date}",
            )
        seen_dates.add(parsed_date)
        parsed_dates.append(parsed_date)
        dates.append(raw_date)

        for column_index, instrument in enumerate(INSTRUMENTS, start=1):
            value = row[column_index]
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0,
                f"snapshot row {index} has invalid {instrument} close: {value!r}",
            )
            prices[instrument].append(float(value))

    provenance = payload.get("provenance")
    _require(isinstance(provenance, dict), "snapshot provenance must be an object")
    for field, expected in (
        ("frequency", "daily"),
        ("timezone", "UTC"),
        ("field", "unadjusted close"),
    ):
        _require(
            provenance.get(field) == expected,
            f"snapshot provenance.{field} must equal {expected!r}",
        )
    _require(
        provenance.get("observedStart") == dates[0],
        "snapshot provenance.observedStart does not match rows",
    )
    _require(
        provenance.get("observedEnd") == dates[-1],
        "snapshot provenance.observedEnd does not match rows",
    )

    quality = payload.get("quality", {})
    _require(isinstance(quality, dict), "snapshot quality must be an object")
    _require(quality.get("rowCount") == len(rows), "snapshot quality.rowCount mismatch")
    _require(
        quality.get("missingSharedCloseCount") == 0,
        "snapshot contains missing shared closes",
    )

    content_sha256 = _canonical_content_sha256(rows)
    _require(
        quality.get("contentSha256") == content_sha256,
        "snapshot embedded contentSha256 mismatch",
    )
    if expected_content_sha256 is not None:
        _require(
            content_sha256 == expected_content_sha256,
            "snapshot contentSha256 does not match protocol",
        )

    file_sha256 = sha256_file(path)
    if expected_file_sha256 is not None:
        _require(
            file_sha256 == expected_file_sha256,
            "snapshot fileSha256 does not match protocol",
        )

    return PriceSnapshot(
        path=path,
        snapshot_id=snapshot_id,
        dates=tuple(dates),
        prices={instrument: tuple(values) for instrument, values in prices.items()},
        file_sha256=file_sha256,
        content_sha256=content_sha256,
        provenance=dict(provenance),
    )


def load_frozen_snapshot(
    protocol_path: Path,
    *,
    repo_root: Path,
) -> PriceSnapshot:
    """Load the local snapshot referenced by a validated thesis-v1 protocol."""

    from regime_sentinel_worker.experiment_protocol import validate_protocol_file

    protocol = validate_protocol_file(protocol_path.resolve(), repo_root.resolve())
    contract = protocol["data"]["snapshot"]
    snapshot_path = (repo_root / contract["path"]).resolve()
    try:
        snapshot_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SnapshotValidationError("snapshot path escapes the repository") from exc
    return load_price_snapshot(
        snapshot_path,
        expected_snapshot_id=contract["snapshotId"],
        expected_row_count=contract["rowCount"],
        expected_file_sha256=contract["fileSha256"],
        expected_content_sha256=contract["contentSha256"],
    )


def build_input_manifest(
    snapshot: PriceSnapshot,
    *,
    snapshot_path: str | None = None,
) -> InputManifest:
    """Build the small input manifest for a validated snapshot."""

    return InputManifest(
        snapshot_id=snapshot.snapshot_id,
        snapshot_path=snapshot_path or snapshot.path.as_posix(),
        snapshot_file_sha256=snapshot.file_sha256,
        snapshot_content_sha256=snapshot.content_sha256,
        instruments=INSTRUMENTS,
        timezone=snapshot.provenance["timezone"],
        observed_start=snapshot.observed_start,
        observed_end=snapshot.observed_end,
        price_observations_per_instrument=snapshot.row_count,
        log_return_observations_per_instrument=max(snapshot.row_count - 1, 0),
    )


def write_input_manifest(manifest: InputManifest, path: Path) -> None:
    """Write a deterministic JSON manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
