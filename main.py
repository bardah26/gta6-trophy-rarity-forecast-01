#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads a case file /input/<id>.in (schema makefun/gta6-trophies-case/v1):
every Exophase read of the GTA VI PS5 trophy page before the case's
checkpoint, and forecasts each trophy's earn rate at the case's target as
five quantiles (5%, 25%, 50%, 75%, 95%), written to /output/<id>.out.

Model, per trophy, in logit space with time u = ln(1 + hours since the
2026-11-19T00:00:00Z launch):
  * a saturating logistic curve  r(u) = R / (1 + exp(-(a + b*u)))  fitted
    by a grid over the ceiling R plus a linear regression in the
    transformed variable ln(R/r - 1), with a weak prior pulling R toward
    3x the last observed rate -- but only when the series visibly
    decelerates in the window (its second-half slope under 75% of its
    first-half slope). A logistic fitted to still-accelerating data
    invents a ceiling the data never showed and extrapolates worse
    than the trend, so without visible deceleration the trend is used;
  * a straight-line logit trend, fitted over the last 48h of reads
    and extrapolated undamped (when the data shows no deceleration,
    the trend is the honest extrapolation; damping it cost a quarter
    of the growth in worlds where the rise continued). A single
    extrapolation is clamped to 3.5 logits from the last rate.
  * a second, structural view of the same trophy: earners E(u) and
    players-tracked N(u) are each fitted with the same
    deceleration-gated machinery, and the rate is E(u)/N(u). This
    matters because a rate is a ratio: when the tracked base grows
    faster than the earners, the rate falls even while every count
    rises (the launch-night practice data shows exactly this dilution
    pattern once Exophase refreshes). When the tracked base moved a
    lot in the window (2x or more), the structural view is the
    informative one and is used alone; otherwise the final median is
    the logit-average of the rate view and the earners/tracked view.
The forecast spread is an honest one: the house random-walk term
(0.5 * sqrt(horizon in days)) combined with the fit residual and a
model-uncertainty term that grows with the horizon, because Exophase
refreshes its page in jumps -- the practice cases show flat history
followed by a jump, so a narrow band is dishonest.

Deterministic: no randomness, no wall-clock reads, no network.
Standard library only. One output file per input case, trophies in the
case's own order, values rounded to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Standard normal quantiles at the five required levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Fit rates are clamped a little tighter than the scorer's 0.01..99.99 so
# the logistic ceiling grid always has room above the largest rate.
FIT_LO = 0.01
FIT_HI = 99.9
OUT_LO = 0.01
OUT_HI = 99.99


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """logit of a rate given in percent, with the scorer's clamping."""
    p = clamp(rate, FIT_LO, FIT_HI) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment, launch):
    return max(0.0, (moment - launch).total_seconds() / 3600.0)


def linfit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, sse, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, sum((y - my) ** 2 for y in ys), n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse, n


def logistic_at(ceil, a, b, u):
    """ceil / (1 + exp(-(a + b*u))), with the exponent clamped so a
    degenerate fit can saturate instead of overflowing."""
    lp = clamp(a + b * u, -50.0, 50.0)
    return ceil / (1.0 + math.exp(-lp))


def _decelerating(us, ys):
    """True when a series' log-slope is clearly falling: the second half
    of the window grows at under 85% of the first half's rate, from a
    positive start. A series still accelerating (or flat) shows no
    saturation yet, and a logistic fitted to it is a guess -- the trend
    extrapolates it better. (The 85% line is deliberately loose: a
    base that is only mildly bending in the window still bends a lot
    over a 4.5-day horizon.)"""
    n = len(us)
    if n < 6:
        return False
    mid = us[n // 2]
    early = [(u, y) for u, y in zip(us, ys) if u <= mid]
    late = [(u, y) for u, y in zip(us, ys) if u > mid]
    if len(early) < 3 or len(late) < 3:
        return False
    _a, b_early, _s, _n = linfit([p[0] for p in early],
                                 [p[1] for p in early])
    _a, b_late, _s, _n = linfit([p[0] for p in late],
                                [p[1] for p in late])
    return b_early > 0.05 and b_late < 0.85 * b_early


def fit_growth(us, vals, prior_mult, hi_mult, u_target):
    """Fit a positive count series (earners, or players tracked) as a
    function of u. A saturating logistic is used only when the series
    visibly decelerates in the window (it is approaching a ceiling);
    otherwise the log-linear trend extrapolates -- a logistic fitted to
    still-accelerating data invents a ceiling the data never showed.
    Returns (predicted value at u_target, residual sd in log space)."""
    logs = [math.log(max(v, 1e-9)) for v in vals]
    a, b, sse_tr, n = linfit(us, logs)
    v_max = max(vals)
    v_last = vals[-1]
    c_prior = prior_mult * v_last
    best = None  # (score, ceil, a_log, b_log, sse)
    if n >= 3 and _decelerating(us, logs):
        lo = min(max(v_max * 1.05, v_max + 0.5), v_max * hi_mult)
        grid = []
        v = lo
        while v < v_max * hi_mult:
            grid.append(v)
            v *= 1.4
        grid.append(v_max * hi_mult)
        for ceil in grid:
            if ceil <= v_max:
                continue
            ws = []
            ok = True
            for val in vals:
                ratio = ceil / val - 1.0
                if ratio <= 0.0:
                    ok = False
                    break
                ws.append(math.log(ratio))
            if not ok:
                continue
            c_w, d_w, _sse_w, _ = linfit(us, ws)
            b_log = -d_w
            if b_log < 0.0:
                b_log = 0.0
                a_log = -sum(ws) / len(ws)
            else:
                a_log = -c_w
            sse = 0.0
            for u, lv in zip(us, logs):
                pred = logistic_at(ceil, a_log, b_log, u)
                sse += (lv - math.log(max(pred, 1e-9))) ** 2
            # Weak tie-breaker only: when the data cannot tell ceilings
            # apart (flat history) the prior wins; when the data has real
            # curvature the fit's SSE must dominate, so this must NOT
            # scale with the number of reads.
            penalty = 0.05 * (math.log(ceil / c_prior)) ** 2
            score = sse + penalty
            if best is None or score < best[0]:
                best = (score, ceil, a_log, b_log, sse)
    if best is not None and best[4] < 0.9 * sse_tr:
        _score, ceil, a_log, b_log, sse = best
        pred = logistic_at(ceil, a_log, b_log, u_target)
        return pred, clamp(math.sqrt(sse / max(n, 1)), 0.12, 0.70)
    # Log-linear trend, undamped: when the series shows no deceleration
    # the trend is the honest extrapolation, and damping it is an
    # unjustified haircut (it cost a quarter of the growth in worlds
    # where the rise continued). Saturation is the logistic's job,
    # above, when the data shows it. The clamp is the only guardrail.
    pred_log = a + b * u_target
    pred_log = clamp(pred_log, logs[-1] - 4.0, logs[-1] + 4.0)
    return math.exp(pred_log), clamp(math.sqrt(sse_tr / max(n, 1)),
                                     0.12, 0.70)


def fit_rate(us, ys, rates, u_target):
    """Fit one trophy's rate series. Returns (median percent at target,
    residual sd in logit space). The saturating logistic is used only
    when the rate visibly decelerates toward a ceiling; otherwise the
    damped linear logit trend (a rising rate with no visible ceiling
    must not be capped by an invented one)."""
    n = len(us)
    r_last = rates[-1] if rates else 1.0
    y_last = ys[-1] if ys else logit_pct(r_last)
    a_lin, b_lin, sse_lin, _ = linfit(us, ys)
    r_max = max(rates) if rates else r_last
    r_prior = clamp(3.0 * r_last, 0.05, 0.95)  # percent
    best = None  # (score, R, a, b, sse)
    if n >= 3 and _decelerating(us, ys):
        lo_r = min(max(r_max * 1.05, r_max + 0.05), 99.0)
        grid = []
        v = lo_r
        while v < 99.0:
            grid.append(v)
            v *= 1.35
        grid.append(99.0)
        for r_ceil in grid:
            if r_ceil <= r_max:
                continue
            ws = []
            ok = True
            for r in rates:
                ratio = r_ceil / clamp(r, FIT_LO, FIT_HI) - 1.0
                if ratio <= 0.0:
                    ok = False
                    break
                ws.append(math.log(ratio))
            if not ok:
                continue
            c_w, d_w, _sse_w, _ = linfit(us, ws)
            b_log = -d_w
            if b_log < 0.0:
                b_log = 0.0
                a_log = -sum(ws) / len(ws)
            else:
                a_log = -c_w
            sse = 0.0
            for u, y in zip(us, ys):
                p_pred = logistic_at(r_ceil, a_log, b_log, u) / 100.0
                p_pred = clamp(p_pred, 0.0001, 0.9999)
                pred_y = math.log(p_pred / (1.0 - p_pred))
                sse += (y - pred_y) ** 2
            # Weak tie-breaker only: when the data cannot tell ceilings
            # apart (flat history) the prior wins; when the data has real
            # curvature the fit's SSE must dominate, so this must NOT
            # scale with the number of reads.
            penalty = 0.05 * (math.log(r_ceil / r_prior)) ** 2
            score = sse + penalty
            if best is None or score < best[0]:
                best = (score, r_ceil, a_log, b_log, sse)
    if best is not None and best[4] < 0.9 * sse_lin:
        _score, r_ceil, a_log, b_log, sse = best
        med = logistic_at(r_ceil, a_log, b_log, u_target)
        return clamp(med, OUT_LO, OUT_HI), math.sqrt(sse / max(n, 1))
    med_logit = a_lin + b_lin * u_target
    # Never let a single extrapolation run more than 3.5 logits from
    # the last observed rate.
    med_logit = clamp(med_logit, y_last - 3.5, y_last + 3.5)
    med = 100.0 * sigmoid(med_logit)
    return clamp(med, OUT_LO, OUT_HI), math.sqrt(sse_lin / max(n, 1))


def forecast_trophy(us, ys, rates, earners, n_pred, n_growth,
                    u_target, horizon_days):
    """Forecast one trophy. us/ys/rates: the rate series; earners: the
    earner counts per read; n_pred: players tracked at the target;
    n_growth: players-tracked growth across the fit window
    (last / first). Returns [q05..q95] in percent."""
    med_rate, resid_sd = fit_rate(us, ys, rates, u_target)
    med = med_rate
    if not us:
        # No reads at all for this trophy: a flat 1% with a wide band.
        med = 1.0
        resid_sd = 0.70
    if earners and n_pred and n_pred > 0.0:
        e_pred, _e_sd = fit_growth(us, earners, 3.0, 50.0, u_target)
        if e_pred > 0.0:
            med_en = clamp(100.0 * e_pred / n_pred, OUT_LO, OUT_HI)
            if n_growth and n_growth >= 2.0:
                # The tracked base moved a lot in the window, so the
                # rate is a moving ratio, not a series of its own: the
                # structural earners/tracked view is the informative
                # one, and the flat rate view is blind to the coming
                # divergence between the two counts.
                med = med_en
            else:
                # Logit-average of the two views: each is a reasonable
                # model, and the average hedges either one's failure.
                med = 100.0 * sigmoid((logit_pct(med_rate)
                                       + logit_pct(med_en)) / 2.0)
    med = clamp(med, OUT_LO, OUT_HI)

    # Honest spread: random walk + fit residual + a modest model
    # term. The random-walk term is the house's calibrated scale; the
    # model term covers refresh jumps and model error, but it must stay
    # modest -- an over-wide band pays at every trophy when the median
    # is right, and the median now carries the forecast.
    s_rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    s_resid = clamp(resid_sd, 0.12, 0.70)
    s_model = 0.15 + 0.15 * math.sqrt(max(horizon_days, 0.0))
    spread = math.sqrt(s_rw * s_rw + s_resid * s_resid
                       + s_model * s_model)
    spread = clamp(spread, 0.12, 2.0)

    m_logit = logit_pct(med)
    out = []
    prev = None
    for z in Z:
        q = 100.0 * sigmoid(m_logit + z * spread)
        q = clamp(round(q, 4), OUT_LO, OUT_HI)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    """Return the output object for one parsed case dict."""
    trophies = case["trophies"]
    history = case.get("history") or []
    launch = LAUNCH
    release = (case.get("game") or {}).get("release_date")
    if release:
        try:
            launch = datetime.fromisoformat(release).replace(
                tzinfo=timezone.utc)
        except ValueError:
            launch = LAUNCH

    target = parse_time(case["target"])
    at = parse_time(case["at"]) if case.get("at") else None
    horizon_days = (target - at).total_seconds() / 86400.0 if at else 0.0
    if horizon_days <= 0.0 and history:
        last_slot = parse_time(history[-1].get("slot") or history[-1]["at"])
        horizon_days = (target - last_slot).total_seconds() / 86400.0
    horizon_days = max(horizon_days, 0.0)
    u_target = math.log1p(hours_since_launch(target, launch))

    # Reads inside the last 48h before the checkpoint carry the signal;
    # older reads are stale for a launch-week growth curve.
    cutoff = None
    if at is not None:
        cutoff = at.timestamp() - 48.0 * 3600.0
    rows = []
    for read in history:
        moment = parse_time(read.get("slot") or read["at"])
        if cutoff is not None and moment.timestamp() < cutoff:
            continue
        rows.append((moment, read.get("rates") or [],
                     read.get("players_tracked")))
    if not rows:
        rows = [(parse_time(r.get("slot") or r["at"]),
                 r.get("rates") or [], r.get("players_tracked"))
                for r in history]

    # Players tracked at the target, fitted once for the whole case.
    n_pred = None
    n_growth = None
    n_us, n_vals = [], []
    for moment, _rates, n_tracked in rows:
        if n_tracked:
            n_us.append(math.log1p(hours_since_launch(moment, launch)))
            n_vals.append(float(n_tracked))
    if len(n_us) >= 2:
        n_pred, _n_sd = fit_growth(n_us, n_vals, 3.0, 20.0, u_target)
        if n_vals[0] > 0.0:
            n_growth = n_vals[-1] / n_vals[0]

    out_trophies = []
    for idx, trophy in enumerate(trophies):
        us, ys, rates, earners = [], [], [], []
        for moment, all_rates, n_tracked in rows:
            if idx >= len(all_rates):
                continue
            r = all_rates[idx]
            if r is None:
                continue
            us.append(math.log1p(hours_since_launch(moment, launch)))
            ys.append(logit_pct(r))
            rates.append(float(r))
            if n_tracked:
                earners.append(max(float(r) / 100.0 * float(n_tracked),
                                   0.5))
        if not us:
            # No data at all for this trophy: a flat 1% with a wide band.
            out_trophies.append({"id": trophy["id"],
                                 "q": forecast_trophy([], [], [1.0], [],
                                                      None, None, u_target,
                                                      horizon_days)})
        else:
            out_trophies.append({"id": trophy["id"],
                                 "q": forecast_trophy(us, ys, rates,
                                                      earners, n_pred,
                                                      n_growth,
                                                      u_target,
                                                      horizon_days)})
    return {"trophies": out_trophies}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"))
        with open(os.path.join(out_dir, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
