#!/usr/bin/env python3
"""GitHub new-repo star-rise monitor.

Discovers repos created within a rolling window, reconstructs their daily star
series from the official stargazers/history endpoint, and emits a ranked
Markdown leaderboard plus a JSON snapshot.

Zero third-party dependencies: Python 3.9+ standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UTC = dt.timezone.utc
DAY_SECONDS = 86400
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2026-03-10",
    "User-Agent": "star-rise-monitor/1.0",
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def api_get(path: str, params: dict | None = None, token: str | None = None):
    """GET a GitHub API path with retry/backoff on rate limits and 5xx."""
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
    last_error = None
    for attempt in range(4):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            if exc.code in (403, 429) and attempt < 3:
                reset = exc.headers.get("x-ratelimit-reset")
                wait = 60
                if reset:
                    try:
                        wait = max(5, int(reset) - int(time.time()) + 3)
                    except ValueError:
                        pass
                log(f"rate limited; sleeping {wait}s ({body})")
                time.sleep(min(wait, 300))
                last_error = exc
                continue
            if 500 <= exc.code < 600 and attempt < 3:
                time.sleep(2 ** attempt)
                last_error = exc
                continue
            raise RuntimeError(f"HTTP {exc.code} on {path}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"network error on {path}: {exc}") from exc
    raise RuntimeError(f"gave up on {path}: {last_error}")


def search_candidates(token: str | None, lookback_days: int, min_stars: int,
                      max_candidates: int, now: dt.datetime) -> list[dict]:
    """Find new repos by stars. Uses the A..B range qualifier (NOT >A <B)."""
    end = now.date()
    start = end - dt.timedelta(days=lookback_days)
    query = (
        f"created:{start.isoformat()}..{end.isoformat()} "
        f"stars:>={min_stars} fork:false archived:false"
    )
    found: list[dict] = []
    seen: set[int] = set()
    for page in range(1, 11):  # search API caps at 1000 results
        data, _ = api_get(
            "/search/repositories",
            {"q": query, "sort": "stars", "order": "desc",
             "per_page": 100, "page": page},
            token,
        )
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                found.append(item)
        log(f"search page {page}: +{len(items)} (total {len(found)}/{data.get('total_count', '?')})")
        if len(found) >= max_candidates or len(items) < 100:
            break
        time.sleep(1)
    return found[:max_candidates]


def get_star_history(full_name: str, token: str | None) -> list[dict]:
    owner, repo = full_name.split("/", 1)
    out: list[dict] = []
    for page in range(1, 6):
        data, _ = api_get(
            f"/repos/{owner}/{repo}/stargazers/history",
            {"per_page": 30, "page": page},
            token,
        )
        if not data:
            break
        out.extend(data)
        if len(data) < 30:
            break
    return out


def ols_slope(values: list[float]) -> float:
    """Least-squares slope of values against 0..n-1. Not enough points -> 0."""
    n = len(values)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2
    my = sum(values) / n
    num = sum((i - mx) * (v - my) for i, v in enumerate(values))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def build_daily(history: list[dict], created: dt.datetime,
                today: dt.date, repo_age_days: int) -> tuple[dict[str, int], int]:
    """Expand weekly buckets into a date->daily-new map.

    Star-history weeks start on Sunday (UTC). `days[0]` = Sunday. The current
    (incomplete) day is kept but flagged by the caller via DATE_TODAY.
    """
    daily: dict[str, int] = {}
    for week in history:
        week_start = week["week"]
        for i, count in enumerate(week.get("days", [])):
            d = dt.datetime.fromtimestamp(week_start + i * DAY_SECONDS, UTC).date()
            if d < created.date() or d > today:
                continue
            daily[d.isoformat()] = daily.get(d.isoformat(), 0) + count
    cumulative = sum(daily.values())
    return daily, cumulative


def analyze(repo: dict, history: list[dict], now: dt.datetime,
            window_days: int) -> dict | None:
    created = dt.datetime.strptime(repo["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    today = now.date()
    age_days = max(1, (now - created).days)
    daily, cumulative = build_daily(history, created, today, age_days)
    if not daily:
        return None

    dates = sorted(daily)
    # Complete days only (exclude today, which is still accumulating).
    complete = [d for d in dates if d != today.isoformat()]
    values = [daily[d] for d in complete]
    if len(values) < 2:
        return None

    cumulative_curve: list[float] = []
    running = 0
    for d in complete:
        running += daily[d]
        cumulative_curve.append(float(running))

    window = min(window_days, len(cumulative_curve))
    slope_recent = ols_slope(cumulative_curve[-window:])

    # Trend = recent slope vs the equally-sized window before it.
    trend = "样本不足"
    ratio = None
    if len(cumulative_curve) >= 2 * window:
        slope_prior = ols_slope(cumulative_curve[-2 * window:-window])
        # A near-zero prior slope makes the ratio explode (e.g. 111x) and
        # misleads readers. Require the denominator to be materially sized.
        if abs(slope_prior) >= max(1.0, abs(slope_recent) * 0.05):
            ratio = slope_recent / slope_prior
            trend = "加速" if ratio > 1.2 else "减速" if ratio < 0.8 else "平稳"
        else:
            trend = "新爆发" if slope_recent > 0 else "平稳"

    today_str = today.isoformat()
    return {
        "full_name": repo["full_name"],
        "url": repo["html_url"],
        "description": (repo.get("description") or "").replace("|", "/").strip(),
        "created_at": repo["created_at"],
        "age_days": age_days,
        "stars_now": repo["stargazers_count"],
        "cumulative": cumulative,
        "daily_latest": daily[complete[-1]],
        "daily_today_partial": daily.get(today_str),
        "daily_avg": cumulative / age_days,
        "slope_recent": slope_recent,
        "slope_window": window,
        "trend": trend,
        "trend_ratio": ratio,
        "series": daily,
    }


def render_markdown(rows: list[dict], meta: dict, top_n: int) -> str:
    lines = [
        "# GitHub 新项目涨星榜",
        "",
        f"**统计时间**：{meta['generated_at']}　|　**更新频率**：每日一次　|　**窗口**：创建 ≤ {meta['lookback_days']} 天",
        "",
        f"> 共扫描 {meta['candidates']} 个候选项目，有效 {meta['valid']} 个，下列为前 {min(top_n, len(rows))} 名。",
        "",
        f"| # | 项目 | 创建时间 | 当前 Star | 累计增长 | 最新完整日新增 | 今日(未完) | 日均新增 | 近{meta['window_days']}日斜率 | 趋势 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(rows[:top_n], 1):
        name = f"[{r['full_name']}]({r['url']})"
        today_partial = "—" if r["daily_today_partial"] is None else f"{r['daily_today_partial']:,}"
        ratio = "" if r["trend_ratio"] is None else f" ({r['trend_ratio']:.2f}x)"
        lines.append(
            f"| {i} | {name} | {r['created_at'][:10]} | {r['stars_now']:,} | "
            f"{r['cumulative']:,} | {r['daily_latest']:,} | {today_partial} | "
            f"{r['daily_avg']:.1f} | {r['slope_recent']:.1f} | {r['trend']}{ratio} |"
        )

    lines += [
        "",
        "## 统计口径",
        "",
        f"- **对象**：创建时间在 {meta['window_start']} 至 {meta['window_end']}（滚动 {meta['lookback_days']} 天）的公开仓库；排除 fork、归档库；星数门槛 ≥ {meta['min_stars']}。搜索使用 `created:A..B` 区间语法（分开写 `>A <B` 会被 GitHub 静默忽略）。",
        "- **数据源**：GitHub 官方 Search API 发现候选，`GET /repos/{owner}/{repo}/stargazers/history` 取历史。该端点按日历周返回，每周含 7 个日增量（`days[0]` 为周日，UTC）；旧的 `stargazers` 列表端点已于 2026-07 起限制访问。",
        "- **累计增长**：`history` 全量日增量之和，即从 0 Star 起累计；等于当前 Star 的近似值（取消星会被回溯剔除，可能与当前总数有小幅偏差）。",
        "- **每日新增**：`days[]` 展开到具体日期；今日数据尚未结束，单独标注为「今日(未完)」，不参与斜率计算。",
        "- **日均新增**：累计增长 ÷ 存活天数（向下取整，至少 1 天）。",
        f"- **涨星速度（主排序）**：最近 `min({meta['window_days']}, 可用完整日数)` 日的**累计曲线最小二乘斜率**（Star/天），反映当下速度而非历史总量。",
        "- **趋势**：近窗口斜率 ÷ 前一等长窗口斜率；>1.2 加速，<0.8 减速，否则平稳；不足两个窗口记「样本不足」；前窗口斜率接近 0 时不除，改记「新爆发」，避免出现失真比值。",
        "- **时效**：每日 UTC 08:00（北京时间 16:00）自动更新一次；榜单为无状态重算，每次重写，历史快照另存 `output/history/`。",
        "",
        "## 已知局限",
        "",
        "- Search API 单查询上限 1000 条且按现有星数排序，会漏掉尚未达到星数门槛的极早期项目；这是用「信号真实性」换「覆盖率」的有意取舍。",
        "- Star 历史为净增口径（用户取消星会回溯影响历史值），与 GH Archive 的 WatchEvent 毛增不同。",
        "- GitHub Actions 的 cron 在整点后可能有数分钟到数十分钟的排队延迟；统计时间以脚本实际运行时刻为准。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--min-age-days", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=150)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--dry-run", action="store_true", help="print table only, write nothing")
    args = parser.parse_args()

    now = dt.datetime.now(UTC)
    token = args.token or None
    log(f"run at {now.isoformat()} | token={'yes' if token else 'no'}")
    if not token:
        log("WARNING: no token; unauthenticated core limit is 60 req/hour")

    candidates = search_candidates(token, args.lookback_days, args.min_stars,
                                   args.candidates, now)
    log(f"{len(candidates)} candidates; fetching star history...")

    results: list[dict] = []
    for idx, repo in enumerate(candidates, 1):
        created = dt.datetime.strptime(repo["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        age = (now - created).total_seconds() / DAY_SECONDS
        if age < args.min_age_days or age > args.lookback_days:
            continue
        try:
            history = get_star_history(repo["full_name"], token)
        except RuntimeError as exc:
            log(f"  [{idx}/{len(candidates)}] skip {repo['full_name']}: {exc}")
            continue
        row = analyze(repo, history, now, args.window_days)
        if row:
            results.append(row)
        if idx % 25 == 0:
            log(f"  [{idx}/{len(candidates)}] {len(results)} valid")

    results.sort(key=lambda r: r["slope_recent"], reverse=True)
    meta = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "generated_at_iso": now.isoformat(),
        "lookback_days": args.lookback_days,
        "min_stars": args.min_stars,
        "window_days": args.window_days,
        "window_start": (now.date() - dt.timedelta(days=args.lookback_days)).isoformat(),
        "window_end": now.date().isoformat(),
        "candidates": len(candidates),
        "valid": len(results),
    }

    markdown = render_markdown(results, meta, args.top)
    if args.dry_run:
        print(markdown)
        return 0

    out_dir = Path(args.output_dir)
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(markdown, encoding="utf-8")

    snapshot = {
        "meta": meta,
        "ranking": [{k: v for k, v in r.items() if k != "series"} for r in results[:args.top]],
        "series": {r["full_name"]: r["series"] for r in results[:args.top]},
    }
    date_str = now.date().isoformat()
    (out_dir / "history" / f"{date_str}.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {out_dir / 'README.md'} and {out_dir / 'history' / (date_str + '.json')}")
    log(f"top 3: {[r['full_name'] for r in results[:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
