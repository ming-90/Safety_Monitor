from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from safety_monitor.models import HazardZone

APP_NAME = "제조 현장 안전 모니터"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", _project_root()))
    return _project_root()


def default_config_path() -> Path:
    if getattr(sys, "frozen", False):
        if platform.system() == "Darwin":
            return Path.home() / "Library" / "Application Support" / APP_NAME / "config.json"
        return Path.home() / ".safety-monitor" / "config.json"
    return _project_root() / "config.json"


def bundled_config_path() -> Path:
    return bundled_resource_root() / "config.json"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is not None:
        p = path
    else:
        p = default_config_path()
        if not p.is_file():
            bundled = bundled_config_path()
            if bundled.is_file():
                p = bundled
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_writable_path(path_value: str) -> str:
    p = Path((path_value or "").strip())
    if not p:
        return ""
    if p.is_absolute():
        return str(p)
    return str(default_config_path().parent / p)


def parse_zones(raw: List[dict]) -> List[HazardZone]:
    return [HazardZone.from_dict(z) for z in raw]


def zones_to_json(zones: List[HazardZone]) -> List[dict]:
    return [z.to_dict() for z in zones]


def resolve_model_path(model_path: str, config_path: Optional[Path] = None) -> str:
    """상대 경로는 설정 파일 디렉터리·프로젝트 루트 기준으로 탐색합니다."""
    p = (model_path or "").strip()
    if not p:
        return ""
    po = Path(p)
    if po.is_file():
        return str(po.resolve())
    roots: List[Path] = []
    if config_path is not None:
        roots.append(config_path.parent)
    roots.append(default_config_path().parent)
    roots.append(bundled_resource_root())
    roots.append(_project_root())
    for root in roots:
        cand = (root / po).resolve()
        if cand.is_file():
            return str(cand)
    return p
