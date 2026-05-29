from __future__ import annotations

import html
import json
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

APP_NAME = "MASTER DASHBOARD"
STYLE_MARKER = "ORIGIN_STYLE_MASTER_V1"
TIMEOUT = 0.45
LOCAL_TZ = ZoneInfo("America/New_York")
JARVIS_VOICE_URL = "http://127.0.0.1:8110"
QDRANT_SEARCH_URL = "http://127.0.0.1:8100/search"
DQ_SALES_URL = "http://127.0.0.1:8090/api/dq/sales"
DEFAULT_AUTO_SEARCH_QUERY = "SALES TODAY FOR DQS"
SOCIAL_BOT_STALE_HOURS = 48
SOCIAL_BOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "gcbw-legacy-social-bot",
        "title": "GCBW Legacy Social Bot",
        "description": "Production GCBW Facebook and Instagram poster. Currently disabled by timer/env flags.",
        "store": "GCBW",
        "path": "/home/msr8109/apex/gcbw_social_bot.py",
        "service": "GCBW_Social_Post_Bot_Daily.service",
        "timer": "GCBW_Social_Post_Bot_Schedule.timer",
        "log": "/home/msr8109/apex/logs/gcbw_social_bot.log",
        "posted_json": "/home/msr8109/apex/credentials/gcbw_posted_files.json",
        "ledger": "/home/msr8109/apex/credentials/gcbw_posted_images.sqlite3",
        "stale_hours": 48,
    },
    {
        "id": "gcbw-social-framework",
        "title": "GCBW Social Bot Framework",
        "description": "Refactored multi-store social bot path for GCBW cutover testing.",
        "store": "GCBW",
        "path": "/home/msr8109/projects/social_bot",
        "service": "GCBW_Social_Bot_Framework.service",
        "timer": "GCBW_Social_Bot_Framework.timer",
        "log": "/home/msr8109/projects/social_bot/logs/gcbw_social_bot.log",
        "posted_json": "/home/msr8109/projects/social_bot/state/gcbw_posted_files.json",
        "ledger": "/home/msr8109/projects/social_bot/state/gcbw_ledger.sqlite3",
        "stale_hours": 48,
    },
    {
        "id": "dq-social-bot",
        "title": "DQ Social Bot",
        "description": "Dairy Queen social bot configuration and compliance scaffold.",
        "store": "DQ",
        "path": "/home/msr8109/projects/social_bot/configs/dq.env",
        "service": None,
        "timer": None,
        "log": "/home/msr8109/projects/social_bot/logs/dq_social_bot.log",
        "posted_json": "/home/msr8109/projects/social_bot/state/dq_posted_files.json",
        "ledger": "/home/msr8109/projects/social_bot/state/dq_ledger.sqlite3",
        "stale_hours": 48,
    },
    {
        "id": "gcbw-weekly-photo-intake",
        "title": "GCBW Weekly Photo Intake",
        "description": "Weekly GCBW photo request, follow-up, and post-generation intake workflow.",
        "store": "GCBW",
        "path": "/home/msr8109/gcbw_social",
        "service": None,
        "timer": None,
        "log": "/home/msr8109/gcbw_social/logs/gcbw_social.log",
        "posted_json": None,
        "ledger": None,
        "stale_hours": 168,
    },
)


@dataclass(frozen=True)
class ServiceCard:
    id: str
    group: str
    title: str
    description: str
    href: str
    probe_url: str | None
    source: str = "PGX"
    badge: str = "live"
    read_only: bool = True


SERVICE_CATALOG: tuple[ServiceCard, ...] = (
    ServiceCard("cfo-dashboard", "Finance", "CFO Dashboard", "Banking, loans, entities, mapping, events, and AI search.", "https://cfo.luxqozp.com/", "http://127.0.0.1:8098/healthz", badge="primary"),
    ServiceCard("cfo-loans", "Finance", "CFO Loans", "Origin-backed loan view from the CFO system.", "https://cfo.luxqozp.com/loans", "http://127.0.0.1:8098/loans", badge="finance"),
    ServiceCard("gcbw-cash-recon", "Finance", "GCBW Cash Recon", "Month-to-date sales and bank deposit reconciliation.", "http://10.0.0.190:8099/mtd", "http://127.0.0.1:8099/mtd", badge="ops"),
    ServiceCard("dq-foodsafe-logs", "Operations", "DQ FoodSafe Logs", "Store food safety logs and shared operating records.", "https://dq-foodsafe-logs.netlify.app/", "https://dq-foodsafe-logs.netlify.app/", source="Netlify", badge="ops"),
    ServiceCard("gcbw-public-site", "Operations", "GCBW Public Site", "Live public storefront. 4,900+ products. Count updated May 29, 2026.", "https://goldcoastbeerwineliquor.com/", "https://goldcoastbeerwineliquor.com/", source="Public", badge="site"),
    ServiceCard("gcbw-legacy-social-bot", "Social Bots", "GCBW Legacy Social Bot", "Production Facebook and Instagram poster: alive state, last run, and last posted asset.", "#social-bots", None, badge="bot"),
    ServiceCard("gcbw-social-framework", "Social Bots", "GCBW Social Bot Framework", "Refactored multi-store bot status and post ledger.", "#social-bots", None, badge="bot"),
    ServiceCard("dq-social-bot", "Social Bots", "DQ Social Bot", "Dairy Queen social bot scaffold status and post ledger.", "#social-bots", None, badge="bot"),
    ServiceCard("gcbw-weekly-photo-intake", "Social Bots", "GCBW Weekly Photo Intake", "Photo request and follow-up automation state.", "#social-bots", None, badge="bot"),
    ServiceCard("apex-dashboard", "Trading", "APEX Dashboard", "APEX intelligence dashboard and status surface.", "https://apex.luxqozp.com/", "http://127.0.0.1:8095/", badge="halted"),
    ServiceCard("bb-terminal", "Trading", "BB Terminal", "OpenBB terminal UI for market research.", "https://bb.luxqozp.com/", "http://127.0.0.1:5173/", badge="research"),
    ServiceCard("qdrant", "AI Search", "Qdrant", "Vector database dashboard and collection health.", "http://10.0.0.190:6333/dashboard", "http://127.0.0.1:6333/", badge="data"),
    ServiceCard("qdrant-master-search", "AI Search", "Qdrant Master Search", "Unified semantic search API across indexed PGX knowledge.", "http://10.0.0.190:8100/", "http://127.0.0.1:8100/health", badge="search"),
    ServiceCard("master-file-lookup", "AI Search", "Master File Lookup", "MASTER FILE lookup API status for entity and tax records.", "#master-file-lookup", "http://127.0.0.1:8102/health", badge="local"),
    ServiceCard("l-drive-search", "AI Search", "L-Drive Search", "Read-only search across the indexed bookkeeper share.", "http://10.0.0.190:8097/search?q=gcbw&limit=5", "http://127.0.0.1:8097/health", badge="search"),
    ServiceCard("jarvis-voice", "AI Search", "Jarvis Voice", "Active Executive Baritone voice profile service for the Jarvis layer.", "http://10.0.0.190:8110/api/voices", "http://127.0.0.1:8110/healthz", badge="ai"),
    ServiceCard("n8n", "Infrastructure", "n8n", "Automation workflow console.", "http://10.0.0.190:5678/", "http://127.0.0.1:5678/healthz", badge="workflow"),
    ServiceCard("dify", "Infrastructure", "Dify", "Local AI application console.", "http://10.0.0.190:3000/", "http://127.0.0.1:3000/", badge="ai"),
    ServiceCard("pgx-command-center", "Infrastructure", "PGX Command Center", "PGX service overview and system dashboard.", "http://10.0.0.190:8090/", "http://127.0.0.1:8090/", badge="ops"),
    ServiceCard("pgx-status-api", "Infrastructure", "PGX Status API", "Build and host status endpoint for PGX services.", "http://10.0.0.190:8103/health", "http://127.0.0.1:8103/health", badge="api"),
    ServiceCard("terminal", "Infrastructure", "Terminal", "Cloudflare-routed PGX browser terminal.", "https://terminal.luxqozp.com/", "http://127.0.0.1:7681/", badge="shell"),
)

GROUPS = ("Finance", "Operations", "Social Bots", "Trading", "AI Search", "Infrastructure")
app = FastAPI(title=APP_NAME, version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)


def probe_url(url: str | None, timeout: float = TIMEOUT) -> dict[str, Any]:
    if not url:
        return {"status": "link", "latency_ms": None, "checked": False}
    started = time.perf_counter()
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "master-dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = int(getattr(response, "status", 0) or 0)
            state = "up" if 200 <= code < 500 else "down"
            return {"status": state, "latency_ms": int((time.perf_counter() - started) * 1000), "checked": True, "code": code}
    except urllib.error.HTTPError as exc:
        state = "up" if 200 <= exc.code < 500 else "down"
        return {"status": state, "latency_ms": int((time.perf_counter() - started) * 1000), "checked": True, "code": exc.code}
    except Exception as exc:  # noqa: BLE001
        return {"status": "down", "latency_ms": int((time.perf_counter() - started) * 1000), "checked": True, "error": exc.__class__.__name__}


def collect_status() -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    for card in SERVICE_CATALOG:
        item = asdict(card)
        item["probe"] = probe_url(card.probe_url)
        cards.append(item)
    summary = {
        "total": len(cards),
        "up": sum(1 for card in cards if card["probe"]["status"] == "up"),
        "down": sum(1 for card in cards if card["probe"]["status"] == "down"),
        "link": sum(1 for card in cards if card["probe"]["status"] == "link"),
    }
    return {"name": APP_NAME, "read_only": True, "generated_at": datetime.now(timezone.utc).isoformat(), "summary": summary, "cards": cards}


def _tone(badge: str) -> str:
    if badge in {"primary", "finance", "ops", "site"}:
        return "green"
    if badge in {"halted", "research"}:
        return "amber"
    if badge in {"search", "data", "ai", "bot"}:
        return "blue"
    return "neutral"


def _initials(title: str) -> str:
    return "".join(part[0].upper() for part in title.replace("-", " ").split()[:2]) or "MD"


def _part_of_day(now: datetime | None = None) -> str:
    current = now or datetime.now(LOCAL_TZ)
    hour = current.astimezone(LOCAL_TZ).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    return "evening"


def build_greeting(status: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    status = status or collect_status()
    summary = status.get("summary", {})
    total = int(summary.get("total") or 0)
    up = int(summary.get("up") or 0)
    down = int(summary.get("down") or 0)
    links = int(summary.get("link") or 0)
    part = _part_of_day(now)
    if total and down == 0:
        health = f"{up} of {total} surfaces are online"
    elif total:
        health = f"{up} of {total} surfaces are online, with {down} needing attention"
    else:
        health = "the dashboard is loading"
    if links:
        health += f", and {links} reference links are available"
    jarvis_up = any(
        str(card.get("title", "")).lower() == "jarvis voice" and card.get("probe", {}).get("status") == "up"
        for card in status.get("cards", [])
    )
    jarvis_text = "Jarvis Voice is online" if jarvis_up else "Jarvis Voice is checking in"
    text = f"Good {part}, Mike. MASTER DASHBOARD is online. {health}. {jarvis_text}."
    return {"part_of_day": part, "text": text, "summary": summary}


def synthesize_jarvis_audio(text: str) -> tuple[bytes, dict[str, str]]:
    payload = json.dumps({"text": text, "voice_id": "jarvis"}).encode("utf-8")
    req = urllib.request.Request(
        f"{JARVIS_VOICE_URL}/api/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=420) as response:
        data = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    if not data.startswith(b"RIFF"):
        raise RuntimeError("Jarvis Voice did not return WAV audio")
    return data, headers


def search_qdrant_master(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "k": 3, "total": 6})
    with urllib.request.urlopen(f"{QDRANT_SEARCH_URL}?{params}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_dq_sales() -> dict[str, Any]:
    with urllib.request.urlopen(DQ_SALES_URL, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _compact_snippet(value: Any, limit: int = 220) -> str:
    snippet = " ".join(str(value or "").split())
    if len(snippet) > limit:
        return snippet[: limit - 1].rstrip() + "..."
    return snippet


def _money(value: Any) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return f"${amount:,.2f}"


def _pct(value: Any) -> str:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return "no year-over-year comparison"
    direction = "up" if pct >= 0 else "down"
    return f"{direction} {abs(pct):.1f}% versus last year"


def _sales_timestamp(payload: dict[str, Any]) -> str:
    raw = str(payload.get("scraped_at") or "").strip()
    if not raw:
        return "the latest PAR POS scrape"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt.astimezone(LOCAL_TZ).strftime("%I:%M %p ET").lstrip("0")
    except ValueError:
        return raw


def _is_dq_sales_today_query(query: str) -> bool:
    q = query.lower()
    has_store = any(token in q for token in ("dqs", "dq", "dairy queen"))
    return has_store and "sales" in q and "today" in q


def build_dq_sales_answer(query: str, sales_payload: dict[str, Any]) -> dict[str, Any]:
    stores = []
    for store in sales_payload.get("stores") or []:
        today = store.get("today") or {}
        current = float(today.get("current") or 0)
        stores.append(
            {
                "store_id": str(store.get("store_id") or ""),
                "store_name": str(store.get("store_name") or "Unknown DQ"),
                "today_sales": current,
                "vs_ly_pct": today.get("vs_ly_pct"),
            }
        )
    stores.sort(key=lambda item: item["today_sales"], reverse=True)
    total = round(sum(item["today_sales"] for item in stores), 2)
    as_of = _sales_timestamp(sales_payload)
    if stores:
        store_sentence = "; ".join(
            f"{item['store_name']} {_money(item['today_sales'])}, {_pct(item['vs_ly_pct'])}"
            for item in stores
        )
        answer = f"DQS sales today are {_money(total)} total as of {as_of}. {store_sentence}."
    else:
        answer = f"I checked DQS sales today, but the PAR POS sales file has no store rows as of {as_of}."
    return {
        "query": query,
        "source": "dq_sales",
        "answer": answer,
        "total_sales_today": total,
        "as_of": as_of,
        "stores": stores,
        "results": [
            {
                "collection": "PAR POS DQ sales",
                "score": None,
                "snippet": f"{item['store_name']}: {_money(item['today_sales'])}, {_pct(item['vs_ly_pct'])}",
            }
            for item in stores
        ],
        "speakable": True,
    }


def build_search_chat_answer(query: str, search_payload: dict[str, Any]) -> dict[str, Any]:
    raw_results = search_payload.get("results") or []
    results = []
    for item in raw_results[:4]:
        results.append(
            {
                "collection": str(item.get("collection") or "unknown"),
                "score": item.get("score"),
                "snippet": _compact_snippet(item.get("snippet") or item.get("id") or "No snippet available."),
            }
        )
    total_hits = int(search_payload.get("total_hits") or len(results))
    collections = int(search_payload.get("collections_searched") or 0)
    if results:
        top = results[0]
        answer = (
            f"I searched PGX for {query}. I found {total_hits} matches across {collections} collections. "
            f"Top result from {top['collection']}: {top['snippet']}"
        )
    else:
        answer = f"I searched PGX for {query}. I did not find a strong match."
    return {"query": query, "answer": answer, "total_hits": total_hits, "collections_searched": collections, "results": results}



def _systemctl_user_state(unit: str | None) -> str:
    if not unit:
        return "none"
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.7,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    state = (result.stdout or result.stderr or "unknown").strip().splitlines()
    return state[0] if state else "unknown"


def _file_mtime_iso(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, LOCAL_TZ).isoformat()
    except OSError:
        return None


def _read_tail_lines(path: str | None, limit: int = 80) -> list[str]:
    if not path:
        return []
    p = Path(path)
    try:
        with p.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line.strip() for line in data.splitlines() if line.strip()][-limit:]


def _parse_log_time(line: str) -> datetime | None:
    match = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d{3})?", line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return None


def _latest_log_event(path: str | None) -> dict[str, Any]:
    lines = _read_tail_lines(path)
    latest_line = lines[-1] if lines else None
    latest_at = None
    post_line = None
    post_at = None
    error_count_24h = 0
    cutoff = datetime.now(LOCAL_TZ) - timedelta(hours=24)
    for line in lines:
        ts = _parse_log_time(line)
        if ts:
            latest_at = ts
            if " ERROR" in line and ts >= cutoff:
                error_count_24h += 1
        lower = line.lower()
        if " posted" in lower and "posted this run" not in lower and "total files ever posted" not in lower:
            post_line = line
            post_at = ts or post_at
    return {
        "last_run_at": latest_at.isoformat() if latest_at else _file_mtime_iso(path),
        "latest_log_line": latest_line,
        "last_post_log_line": post_line,
        "last_post_log_at": post_at.isoformat() if post_at else None,
        "errors_24h": error_count_24h,
    }


def _load_posted_json(path: str | None) -> tuple[int, dict[str, Any] | None]:
    if not path:
        return 0, None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, None
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    posts = [row for row in rows if isinstance(row, dict)]
    def key(row: dict[str, Any]) -> str:
        return str(row.get("posted_at") or "")
    latest = max(posts, key=key, default=None)
    return len(posts), latest


def _ledger_post_count(path: str | None) -> int | None:
    if not path or not Path(path).exists():
        return None
    try:
        with sqlite3.connect(path, timeout=0.2) as conn:
            row = conn.execute("SELECT COUNT(*) FROM posted_images WHERE status='posted'").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0] or 0) if row else None


def _normalize_last_post(post: dict[str, Any] | None, log_event: dict[str, Any]) -> dict[str, Any] | None:
    if post:
        return {
            "filename": post.get("filename") or "unknown",
            "posted_at": post.get("posted_at"),
            "fb_post_id": post.get("fb_post_id"),
            "ig_post_id": post.get("ig_post_id"),
        }
    if log_event.get("last_post_log_line"):
        return {
            "filename": None,
            "posted_at": log_event.get("last_post_log_at"),
            "fb_post_id": None,
            "ig_post_id": None,
            "log_line": log_event.get("last_post_log_line"),
        }
    return None


def _log_indicates_disabled(latest_log_line: str | None) -> bool:
    line = (latest_log_line or "").lower()
    return "disabled by configuration" in line


def _is_scaffold_ready(spec: dict[str, Any], path_exists: bool, log_event: dict[str, Any], posted_count: int) -> bool:
    if not path_exists or posted_count > 0 or log_event.get("last_run_at"):
        return False
    return str(spec.get("path") or "").endswith(".env")


def _recent_run_within_hours(log_event: dict[str, Any], hours: int) -> bool:
    if not log_event.get("last_run_at"):
        return False
    try:
        last_run = datetime.fromisoformat(str(log_event["last_run_at"]))
    except ValueError:
        return False
    return datetime.now(LOCAL_TZ) - last_run.astimezone(LOCAL_TZ) <= timedelta(hours=hours)


def _latest_run_has_error(log_path: str | None) -> bool:
    lines = _read_tail_lines(log_path, limit=40)
    if not lines:
        return False
    start_idx = 0
    for i, line in enumerate(lines):
        if "Starting" in line or "Social Bot Starting" in line:
            start_idx = i
    return any(" ERROR" in line for line in lines[start_idx:])


def _classify_social_bot(
    spec: dict[str, Any],
    *,
    path_exists: bool,
    service_state: str,
    schedule_state: str,
    log_event: dict[str, Any],
    posted_count: int,
) -> dict[str, Any]:
    stale_hours = int(spec.get("stale_hours") or SOCIAL_BOT_STALE_HOURS)
    latest_log_line = log_event.get("latest_log_line")
    errors_24h = int(log_event.get("errors_24h") or 0)
    timer_or_service_active = service_state == "active" or schedule_state == "active"
    recent_run = _recent_run_within_hours(log_event, stale_hours)

    if timer_or_service_active:
        if _log_indicates_disabled(latest_log_line):
            return {
                "state": "disabled",
                "badge_label": "Disabled",
                "alive": True,
                "needs_attention": False,
            }
        return {
            "state": "alive",
            "badge_label": "Alive",
            "alive": True,
            "needs_attention": False,
        }
    if _log_indicates_disabled(latest_log_line):
        return {
            "state": "disabled",
            "badge_label": "Disabled",
            "alive": False,
            "needs_attention": False,
        }
    if _is_scaffold_ready(spec, path_exists, log_event, posted_count):
        return {
            "state": "ready",
            "badge_label": "Ready",
            "alive": True,
            "needs_attention": False,
        }
    if errors_24h > 0 or _latest_run_has_error(spec.get("log")):
        return {
            "state": "error",
            "badge_label": "Error",
            "alive": False,
            "needs_attention": True,
        }
    if timer_or_service_active or recent_run:
        return {
            "state": "alive",
            "badge_label": "Alive",
            "alive": True,
            "needs_attention": False,
        }
    return {
        "state": "check",
        "badge_label": "Check",
        "alive": False,
        "needs_attention": True,
    }


def build_social_bot_status() -> dict[str, Any]:
    bots: list[dict[str, Any]] = []
    for spec in SOCIAL_BOT_SPECS:
        service_state = _systemctl_user_state(spec.get("service"))
        schedule_state = _systemctl_user_state(spec.get("timer"))
        log_event = _latest_log_event(spec.get("log"))
        json_count, json_latest = _load_posted_json(spec.get("posted_json"))
        ledger_count = _ledger_post_count(spec.get("ledger"))
        posted_count = max(json_count, ledger_count or 0)
        path_exists = Path(str(spec.get("path") or "")).exists()
        classification = _classify_social_bot(
            spec,
            path_exists=path_exists,
            service_state=service_state,
            schedule_state=schedule_state,
            log_event=log_event,
            posted_count=posted_count,
        )
        last_post = _normalize_last_post(json_latest, log_event)
        bots.append({
            "id": spec["id"],
            "title": spec["title"],
            "description": spec["description"],
            "store": spec["store"],
            "path": spec["path"],
            "path_exists": path_exists,
            "service": spec.get("service"),
            "timer": spec.get("timer"),
            "service_state": service_state,
            "schedule_state": schedule_state,
            "state": classification["state"],
            "badge_label": classification["badge_label"],
            "alive": classification["alive"],
            "needs_attention": classification["needs_attention"],
            "posted_count": posted_count,
            "last_post": last_post,
            "last_posted_at": (last_post or {}).get("posted_at"),
            "last_run_at": log_event.get("last_run_at"),
            "latest_log_line": log_event.get("latest_log_line"),
            "errors_24h": log_event.get("errors_24h", 0),
        })
    summary = {
        "total": len(bots),
        "alive": sum(1 for bot in bots if bot["alive"]),
        "needs_check": sum(1 for bot in bots if bot["needs_attention"]),
        "needs_attention": sum(1 for bot in bots if bot["needs_attention"]),
        "posted_total": sum(int(bot.get("posted_count") or 0) for bot in bots),
    }
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "summary": summary, "bots": bots}

def render_dashboard() -> str:
    greeting_label = html.escape(f"Good {_part_of_day()}, Mike")
    default_query = html.escape(DEFAULT_AUTO_SEARCH_QUERY, quote=True)
    default_query_js = json.dumps(DEFAULT_AUTO_SEARCH_QUERY)
    sections: list[str] = []
    for group in GROUPS:
        cards = [card for card in SERVICE_CATALOG if card.group == group]
        body = []
        for card in cards:
            cid = html.escape(card.id)
            body.append(f'''<article class="card" id="{cid}" data-card-id="{cid}">
<div class="cardTop"><div class="mark">{html.escape(_initials(card.title))}</div><div class="pill" data-status-pill="{cid}">Checking</div></div>
<h3>{html.escape(card.title)}</h3><p>{html.escape(card.description)}</p>
<div class="cardFoot"><span class="badge {_tone(card.badge)}">{html.escape(card.badge.upper())}</span><span>{html.escape(card.source)}</span></div>
<a class="open" href="{html.escape(card.href, quote=True)}" target="_blank" rel="noopener" aria-label="Open {html.escape(card.title)}">Open</a>
</article>''')
        sections.append(f'''<section class="section" id="{html.escape(group.lower().replace(' ', '-'))}"><div class="sectionHead"><span>{html.escape(group)}</span><strong>{len(cards)}</strong></div><div class="grid">{''.join(body)}</div></section>''')

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{APP_NAME}</title><meta name="dashboard-style" content="{STYLE_MARKER}"><style>
:root{{--bg:#f6f5f2;--surface:#fff;--soft:#efede7;--ink:#111;--muted:#696969;--line:#dedbd4;--green:#0f8f5f;--blue:#1f78d1;--amber:#a16207;--shadow:0 16px 40px rgba(17,17,17,.06)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}.layout{{display:grid;grid-template-columns:236px minmax(0,1fr);min-height:100vh}}.side{{border-right:1px solid var(--line);background:rgba(255,255,255,.72);padding:22px 14px;position:sticky;top:0;height:100vh}}.brand{{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;margin:0 6px 34px}}.brandMark{{width:24px;height:24px;border:2px solid var(--ink);border-radius:7px;display:grid;place-items:center;font-size:11px}}.label{{color:#9a978f;font-size:11px;text-transform:uppercase;margin:20px 10px 8px}}.nav{{height:34px;display:flex;align-items:center;padding:0 10px;border-radius:7px;color:#56534d;text-decoration:none;font-size:14px}}.nav.active,.nav:hover{{background:#e9e7e1;color:var(--ink)}}main{{padding:34px 42px 60px;min-width:0}}.top{{display:flex;justify-content:space-between;gap:22px;margin-bottom:26px}}h1{{font-size:30px;line-height:1.15;margin:0 0 8px;font-weight:650}}.sub{{margin:0;color:var(--muted);font-size:14px}}.round{{width:36px;height:36px;border-radius:50%;border:1px solid var(--line);background:var(--surface);display:grid;place-items:center;font-size:13px}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;margin-bottom:18px}}.metric{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:15px 16px;min-height:82px}}.metric span{{display:block;color:var(--muted);font-size:12px;margin-bottom:8px}}.metric strong{{font-size:24px;font-weight:600}}.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);margin-bottom:20px;overflow:hidden}}.panelHead{{padding:18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:14px}}.panelHead span{{color:var(--muted);font-size:12px;text-transform:uppercase}}.pulse{{display:grid;grid-template-columns:1.1fr .9fr}}.copy{{padding:26px 18px}}.copy h2{{margin:0 0 8px;font-size:22px;font-weight:620}}.copy p{{margin:0;color:var(--muted);max-width:620px;font-size:14px;line-height:1.55}}.bars{{min-height:164px;border-left:1px solid var(--line);background:linear-gradient(180deg,#fbfbfa,#f3f7f3);display:flex;align-items:end;padding:22px;gap:9px}}.bar{{flex:1;min-width:10px;border-radius:5px 5px 0 0;background:#99d8bd}}.bar:nth-child(2n){{background:#82c4ef}}.bar:nth-child(3n){{background:#e8c36b}}.section{{margin-top:16px}}.sectionHead{{height:46px;display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:12px;text-transform:uppercase}}.sectionHead strong{{color:var(--ink);font-size:13px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}.card{{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:8px;min-height:202px;padding:14px;display:flex;flex-direction:column;gap:14px}}.card:hover{{box-shadow:0 10px 26px rgba(17,17,17,.05)}}.cardTop,.cardFoot{{display:flex;align-items:center;justify-content:space-between;gap:10px}}.mark{{width:38px;height:38px;border-radius:8px;background:#f0f0ec;display:grid;place-items:center;font-weight:700;font-size:13px}}.pill{{border:1px solid var(--line);border-radius:999px;color:var(--muted);background:var(--soft);padding:4px 8px;font-size:12px}}.pill.up{{color:var(--green);background:#e5f4ec;border-color:#b7ddca}}.pill.down{{color:#9f1239;background:#fde8ee;border-color:#f4b8c9}}h3{{margin:2px 0 8px;font-size:18px;line-height:1.25;font-weight:650}}p{{margin:0;color:var(--muted);font-size:13px;line-height:1.45}}.cardFoot{{margin-top:auto;color:var(--muted);font-size:12px}}.badge{{border-radius:999px;padding:4px 8px;font-size:11px;border:1px solid var(--line)}}.green{{color:var(--green);background:#e5f4ec;border-color:#b7ddca}}.blue{{color:var(--blue);background:#e8f1fb;border-color:#bbd8f5}}.amber{{color:var(--amber);background:#fbf0d7;border-color:#ebcf91}}.neutral{{color:#4b5563;background:#eeeeea}}.open{{position:absolute;inset:0;color:transparent;text-decoration:none}}.note{{margin-top:26px;color:var(--muted);font-size:12px}}.jarvisPanel{{padding:18px;display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,.9fr);gap:18px;align-items:stretch}}.jarvisStatus{{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;margin-top:14px}}.dot{{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(15,143,95,.12)}}.jarvisControls{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}}.btn{{border:1px solid var(--ink);background:var(--ink);color:#fff;border-radius:7px;min-height:38px;padding:0 14px;font:inherit;font-size:14px;cursor:pointer}}.btn.secondary{{background:#fff;color:var(--ink)}}.btn:disabled{{opacity:.55;cursor:wait}}.voiceAudio{{width:100%;margin-top:14px}}.searchBox{{border:1px solid var(--line);border-radius:8px;background:#fbfbfa;padding:14px}}.searchBox h3{{font-size:17px;margin:0 0 12px}}.searchForm{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px}}.searchInput{{height:40px;border:1px solid var(--line);border-radius:7px;background:#fff;padding:0 12px;font:inherit;font-size:14px;min-width:0}}.searchAnswer{{margin-top:12px;color:var(--ink);font-size:14px;line-height:1.5}}.searchResults{{margin:12px 0 0;padding:0;list-style:none;display:grid;gap:8px}}.searchResults li{{border-top:1px solid var(--line);padding-top:8px;color:var(--muted);font-size:12px}}.searchResults strong{{display:block;color:var(--ink);font-size:13px;margin-bottom:3px}}.socialBots{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;padding:18px}}.socialBot{{border:1px solid var(--line);border-radius:8px;background:#fbfbfa;padding:14px;display:grid;gap:10px}}.socialBotTop{{display:flex;align-items:start;justify-content:space-between;gap:12px}}.socialBot h3{{font-size:17px;margin:0}}.socialBotMeta{{display:grid;grid-template-columns:1fr 1fr;gap:8px;color:var(--muted);font-size:12px}}.socialBotMeta strong{{display:block;color:var(--ink);font-size:13px;margin-top:2px}}.socialBotLog{{border-top:1px solid var(--line);padding-top:8px;color:var(--muted);font-size:12px;line-height:1.35;word-break:break-word}}.socialBotEmpty{{color:var(--muted);font-size:13px}}@media(max-width:980px){{.layout{{grid-template-columns:1fr}}.side{{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}}main{{padding:24px 16px 44px}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.pulse{{grid-template-columns:1fr}}.bars{{border-left:0;border-top:1px solid var(--line)}}}}@media(max-width:580px){{.top{{flex-direction:column}}.metrics,.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="layout" data-style="{STYLE_MARKER}"><aside class="side"><div class="brand"><span class="brandMark">MD</span><span>MASTER DASHBOARD</span></div><div class="label">Command</div><a class="nav active" href="#overview">Overview</a><a class="nav" href="#finance">Finance</a><a class="nav" href="#operations">Operations</a><a class="nav" href="#social-bots">Social Bots</a><a class="nav" href="#trading">Trading</a><a class="nav" href="#ai-search">AI Search</a><a class="nav" href="#infrastructure">Infrastructure</a></aside><main id="overview"><div class="top"><div><h1 id="greetingTitle">{greeting_label}</h1><p class="sub">PGX command surface for finance, operations, research, search, and infrastructure.</p></div><div class="round">PGX</div></div><div class="metrics"><div class="metric"><span>Total surfaces</span><strong id="metric-total">{len(SERVICE_CATALOG)}</strong></div><div class="metric"><span>Online</span><strong id="metric-up">--</strong></div><div class="metric"><span>Needs check</span><strong id="metric-down">--</strong></div><div class="metric"><span>Mode</span><strong>Read-only</strong></div></div><section class="panel"><div class="panelHead"><span>Net operations</span><strong id="last-updated">Checking status</strong></div><div class="pulse"><div class="copy"><h2>Everything important in one place.</h2><p>Finance, store operations, market research, AI search, and PGX infrastructure are grouped into one command surface with live health probes and direct links.</p></div><div class="bars"><div class="bar" style="height:35%"></div><div class="bar" style="height:54%"></div><div class="bar" style="height:42%"></div><div class="bar" style="height:72%"></div><div class="bar" style="height:64%"></div><div class="bar" style="height:82%"></div><div class="bar" style="height:76%"></div><div class="bar" style="height:90%"></div></div></div></section><section class="panel" id="jarvis-loading"><div class="panelHead"><span>Jarvis Voice</span><strong id="jarvisVoiceLabel">Loading voice</strong></div><div class="jarvisPanel"><div class="copy"><h2 id="jarvisGreetingText">{greeting_label}</h2><p>Jarvis speaks the dashboard load greeting and reads search chat answers using the active JARVIS Voicebox profile.</p><div class="jarvisStatus"><span class="dot"></span><span id="jarvisVoiceStatus">Preparing greeting.</span></div><div class="jarvisControls"><button class="btn" id="jarvisEnableBtn" type="button">Start Jarvis</button><button class="btn secondary" id="jarvisReplayBtn" type="button">Replay greeting</button></div><audio class="voiceAudio" id="jarvisBootAudio" preload="auto" playsinline></audio></div><div class="searchBox"><h3>Search Chat</h3><form class="searchForm" id="searchChatForm"><input class="searchInput" id="searchChatInput" name="q" autocomplete="off" placeholder="Ask PGX search..." value="{default_query}" aria-label="Search PGX"><button class="btn" type="submit">Ask</button></form><div class="searchAnswer" id="searchChatAnswer">Ask a question and Jarvis will read the answer.</div><ul class="searchResults" id="searchChatResults"></ul><audio class="voiceAudio" id="jarvisSearchAudio" preload="none" playsinline controls></audio></div></div></section><section class="panel" id="socialBotsPanel"><div class="panelHead"><span>Social Bots</span><strong id="socialBotsSummary">Checking bots</strong></div><div class="socialBots" id="socialBotsList"><div class="socialBotEmpty">Loading bot status...</div></div></section>{''.join(sections)}<p class="note">MASTER DASHBOARD runs on PGX port 8101. Public hostname routing remains a separate Cloudflare dashboard action.</p></main></div><script>
const jarvisBootAudio=document.getElementById('jarvisBootAudio');
const jarvisSearchAudio=document.getElementById('jarvisSearchAudio');
const jarvisEnableBtn=document.getElementById('jarvisEnableBtn');
const jarvisReplayBtn=document.getElementById('jarvisReplayBtn');
const jarvisVoiceStatus=document.getElementById('jarvisVoiceStatus');
const jarvisGreetingText=document.getElementById('jarvisGreetingText');
const jarvisVoiceLabel=document.getElementById('jarvisVoiceLabel');
let pendingBootUrl=null;
const DEFAULT_AUTO_SEARCH_QUERY={default_query_js};

function fmtWhen(value){{if(!value)return 'never';try{{return new Date(value).toLocaleString();}}catch(e){{return value;}}}}
function shortLog(value){{if(!value)return 'No log line found';return value.length>180?value.slice(0,179)+'...':value;}}
function botPillClass(bot){{if(bot.state==='error'||bot.state==='check')return 'down';return 'up';}}
async function refreshSocialBots(){{const panel=document.getElementById('socialBotsList');const summary=document.getElementById('socialBotsSummary');if(!panel||!summary)return;try{{const r=await fetch('/api/social-bots',{{cache:'no-store'}});const p=await r.json();const attention=p.summary.needs_attention??p.summary.needs_check??0;summary.textContent=`${{p.summary.alive}} alive · ${{attention}} need attention · ${{p.summary.posted_total}} posted`;panel.innerHTML=(p.bots||[]).map(bot=>`<article class="socialBot"><div class="socialBotTop"><h3>${{bot.title}}</h3><span class="pill ${{botPillClass(bot)}}">${{bot.badge_label||'Check'}}</span></div><p>${{bot.description}}</p><div class="socialBotMeta"><span>Schedule<strong>${{bot.schedule_state||'none'}}</strong></span><span>Service<strong>${{bot.service_state||'none'}}</strong></span><span>Last run<strong>${{fmtWhen(bot.last_run_at)}}</strong></span><span>Last posted<strong>${{fmtWhen(bot.last_posted_at)}}</strong></span><span>Posted total<strong>${{bot.posted_count}}</strong></span><span>24h errors<strong>${{bot.errors_24h}}</strong></span></div><div class="socialBotLog">${{shortLog(bot.latest_log_line)}}</div></article>`).join('');}}catch(e){{summary.textContent='Bot status unavailable';panel.innerHTML='<div class="socialBotEmpty">Social bot status unavailable.</div>';}}}}
async function refreshStatus(){{try{{const r=await fetch('/api/status',{{cache:'no-store'}});const p=await r.json();document.getElementById('metric-total').textContent=p.summary.total;document.getElementById('metric-up').textContent=p.summary.up;document.getElementById('metric-down').textContent=p.summary.down;document.getElementById('last-updated').textContent=new Date(p.generated_at).toLocaleString();for(const c of p.cards){{const pill=document.querySelector(`[data-status-pill="${{c.id}}"]`);if(!pill)continue;pill.classList.remove('up','down');if(c.probe.status==='up'){{pill.classList.add('up');pill.textContent=c.probe.latency_ms===null?'Online':`Online ${{c.probe.latency_ms}}ms`;}}else if(c.probe.status==='down'){{pill.classList.add('down');pill.textContent='Check';}}else{{pill.textContent='Link';}}}}}}catch(e){{document.getElementById('last-updated').textContent='Status unavailable';}}}}
async function loadGreetingText(){{const r=await fetch('/api/jarvis/greeting',{{cache:'no-store'}});if(!r.ok)throw new Error('Greeting unavailable');const p=await r.json();jarvisGreetingText.textContent=p.text;document.getElementById('greetingTitle').textContent=`Good ${{p.part_of_day}}, Mike`;return p.text;}}
async function fetchVoice(endpoint,body){{const r=await fetch(endpoint,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:body?JSON.stringify(body):undefined}});if(!r.ok)throw new Error('Voice request failed: '+r.status);return URL.createObjectURL(await r.blob());}}
async function playUrl(audio,url){{audio.src=url;await audio.play();}}
async function playGreeting(){{jarvisEnableBtn.disabled=true;jarvisVoiceStatus.textContent='Generating Jarvis greeting...';try{{await loadGreetingText();pendingBootUrl=await fetchVoice('/api/jarvis/greeting/speak');try{{await playUrl(jarvisBootAudio,pendingBootUrl);jarvisVoiceStatus.textContent='Jarvis greeting played.';jarvisVoiceLabel.textContent='Voice active';}}catch(e){{jarvisVoiceStatus.textContent='Voice is ready. Click Start Jarvis to play.';jarvisVoiceLabel.textContent='Click to start';}}}}catch(e){{jarvisVoiceStatus.textContent=e.message;jarvisVoiceLabel.textContent='Voice check failed';}}finally{{jarvisEnableBtn.disabled=false;}}}}
jarvisEnableBtn.addEventListener('click',async()=>{{if(pendingBootUrl){{try{{await playUrl(jarvisBootAudio,pendingBootUrl);jarvisVoiceStatus.textContent='Jarvis greeting played.';jarvisVoiceLabel.textContent='Voice active';return;}}catch(e){{jarvisVoiceStatus.textContent=e.message;}}}}await playGreeting();}});
jarvisReplayBtn.addEventListener('click',playGreeting);
document.addEventListener('click',async()=>{{if(pendingBootUrl&&jarvisBootAudio.paused){{try{{await playUrl(jarvisBootAudio,pendingBootUrl);jarvisVoiceStatus.textContent='Jarvis greeting played.';jarvisVoiceLabel.textContent='Voice active';pendingBootUrl=null;}}catch(e){{}}}}}},{{once:true}});
async function runSearch(q){{const input=document.getElementById('searchChatInput');const answer=document.getElementById('searchChatAnswer');const results=document.getElementById('searchChatResults');input.value=q;if(!q)return;answer.textContent='Searching PGX...';results.innerHTML='';try{{const r=await fetch('/api/search-chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{q}})}});if(!r.ok)throw new Error('Search failed: '+r.status);const p=await r.json();answer.textContent=p.answer;results.innerHTML=(p.results||[]).map(item=>`<li><strong>${{item.collection}}</strong>${{item.snippet}}</li>`).join('');const voiceUrl=await fetchVoice('/api/jarvis/speak',{{text:p.answer}});try{{await playUrl(jarvisSearchAudio,voiceUrl);}}catch(e){{jarvisSearchAudio.src=voiceUrl;answer.textContent=p.answer+' Voice is ready. Click play if Chrome blocked autoplay.';}}}}catch(e){{answer.textContent=e.message;}}}}
document.getElementById('searchChatForm').addEventListener('submit',async(event)=>{{event.preventDefault();await runSearch(document.getElementById('searchChatInput').value.trim());}});
refreshStatus();refreshSocialBots();window.setInterval(refreshStatus,30000);window.setInterval(refreshSocialBots,30000);window.setTimeout(playGreeting,450);window.setTimeout(()=>runSearch(DEFAULT_AUTO_SEARCH_QUERY),1300);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def homepage() -> str:
    return render_dashboard()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "name": APP_NAME, "read_only": True, "cards": len(SERVICE_CATALOG)}


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(collect_status())


@app.get("/api/social-bots")
def api_social_bots() -> JSONResponse:
    return JSONResponse(build_social_bot_status())


@app.get("/api/jarvis/greeting")
def api_jarvis_greeting() -> JSONResponse:
    return JSONResponse({"ok": True, **build_greeting(collect_status())})


@app.post("/api/jarvis/greeting/speak")
def api_jarvis_greeting_speak() -> Response:
    greeting = build_greeting(collect_status())
    try:
        audio, headers = synthesize_jarvis_audio(greeting["text"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Jarvis Voice unavailable") from exc
    return Response(audio, media_type="audio/wav", headers={
        "Cache-Control": "no-store",
        "Content-Disposition": 'inline; filename="jarvis_dashboard_greeting.wav"',
        "X-Jarvis-TTS-Provider": headers.get("x-jarvis-tts-provider", "voicebox"),
        "X-Jarvis-TTS-Model": headers.get("x-jarvis-tts-model", "luxtts"),
        "X-Jarvis-TTS-Voice": headers.get("x-jarvis-tts-voice", "JARVIS"),
    })


@app.post("/api/jarvis/speak")
async def api_jarvis_speak(request: Request) -> Response:
    payload = await request.json()
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 4096:
        raise HTTPException(status_code=400, detail="text exceeds 4096 characters")
    try:
        audio, headers = synthesize_jarvis_audio(text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Jarvis Voice unavailable") from exc
    return Response(audio, media_type="audio/wav", headers={
        "Cache-Control": "no-store",
        "Content-Disposition": 'inline; filename="jarvis_search_chat.wav"',
        "X-Jarvis-TTS-Provider": headers.get("x-jarvis-tts-provider", "voicebox"),
        "X-Jarvis-TTS-Model": headers.get("x-jarvis-tts-model", "luxtts"),
        "X-Jarvis-TTS-Voice": headers.get("x-jarvis-tts-voice", "JARVIS"),
    })


@app.post("/api/search-chat")
async def api_search_chat(request: Request) -> JSONResponse:
    payload = await request.json()
    query = str(payload.get("q") or payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    try:
        if _is_dq_sales_today_query(query):
            answer = build_dq_sales_answer(query, fetch_dq_sales())
        else:
            answer = build_search_chat_answer(query, search_qdrant_master(query))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="PGX search unavailable") from exc
    return JSONResponse({"ok": True, "speakable": True, **answer})


@app.get("/favicon.ico")
def favicon() -> JSONResponse:
    return JSONResponse({"ok": True})
