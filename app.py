"""汎用 XRD 解析・AI診断アプリ。起動: streamlit run app.py"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

from materials import (
    CRYSTAL_SYSTEMS,
    CUSTOM_PRESET_ID,
    KNOWN_IMPURITIES,
    MATERIAL_PRESETS,
    constrain_by_system,
    crystal_system_defaults,
    dummy_structure,
    fit_axes,
    impurity_structure,
    scale_structure,
    structure_from_params,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_XRD_PATH = APP_DIR / "sample_data" / "nb2alc_dummy.csv"


def get_openai_api_key(user_input: str = "") -> str:
    typed = (user_input or "").strip()
    if typed:
        return typed
    try:
        return str(st.secrets.get("OPENAI_API_KEY") or "").strip()
    except Exception:
        return ""


def load_xy(file) -> pd.DataFrame:
    if isinstance(file, (str, Path)):
        raw = Path(file).read_bytes()
    else:
        raw = file.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    text = raw.decode("utf-8-sig", errors="ignore")
    if text.count("\n") < 2 and "\\n" in text:
        text = text.replace("\\n", "\n")
    buf = io.StringIO(text)
    last_err: Exception | None = None
    for sep in [None, ",", "\t", ";", r"\s+"]:
        buf.seek(0)
        try:
            if sep is None:
                df = pd.read_csv(buf, comment="#", header=None, engine="python")
            else:
                df = pd.read_csv(buf, sep=sep, comment="#", header=None, engine="python")
            numeric = df.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.dropna(axis=1, how="all")
            if numeric.shape[1] < 2:
                continue
            xy = numeric.iloc[:, :2].copy()
            xy.columns = ["two_theta", "intensity"]
            xy = xy.dropna()
            if len(xy) >= 10:
                return xy.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise ValueError(f"2θ と Intensity の2列を読めませんでした: {last_err}")


def structure_from_cif(upload) -> Structure:
    suffix = os.path.splitext(upload.name)[1] or ".cif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        path = tmp.name
    try:
        return Structure.from_file(path)
    finally:
        os.unlink(path)


def describe_structure(struct: Structure) -> dict[str, Any]:
    lat = struct.lattice
    a, b, c = lat.abc
    alpha, beta, gamma = lat.angles
    sg_symbol, sg_number, crystal = "?", 0, "unknown"
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = int(sga.get_space_group_number())
        crystal = sga.get_crystal_system() or "unknown"
    except Exception:
        pass
    return {
        "formula": struct.composition.reduced_formula,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "space_group": sg_symbol,
        "space_group_number": sg_number,
        "crystal_system": crystal,
        "n_sites": len(struct),
    }


def xrd_pattern(structure: Structure, tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray, list]:
    calc = XRDCalculator(wavelength="CuKa")
    pat = calc.get_pattern(structure, two_theta_range=(tmin, tmax))
    return np.asarray(pat.x, dtype=float), np.asarray(pat.y, dtype=float), list(pat.hkls)


def sticks_to_curve(grid: np.ndarray, px: np.ndarray, py: np.ndarray, fwhm: float) -> np.ndarray:
    y = np.zeros_like(grid, dtype=float)
    if len(px) == 0:
        return y
    sigma = max(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-4)
    for x0, i0 in zip(px, py):
        y += float(i0) * np.exp(-0.5 * ((grid - x0) / sigma) ** 2)
    return y


def r_metrics(yobs: np.ndarray, ycalc: np.ndarray) -> dict[str, float]:
    denom = float(np.dot(ycalc, ycalc) + 1e-18)
    scale = float(np.dot(yobs, ycalc) / denom)
    y = scale * ycalc
    r_p = float(np.sum(np.abs(yobs - y)) / (np.sum(np.abs(yobs)) + 1e-18))
    r_wp = float(np.sqrt(np.sum((yobs - y) ** 2) / (np.sum(yobs**2) + 1e-18)))
    n = len(yobs)
    s_like = float(np.sqrt(np.sum((yobs - y) ** 2) / max(n - 3, 1)))
    return {"scale": scale, "R": r_p, "Rwp": r_wp, "S": s_like}


def fit_lattice(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    proto: Structure,
    crystal_system: str,
    params: dict[str, float],
    fwhm: float,
) -> dict[str, Any]:
    tmin, tmax = float(two_theta.min()), float(two_theta.max())
    yobs = intensity.astype(float)
    yobs = yobs / (yobs.max() + 1e-18)
    axes = fit_axes(crystal_system)
    x0 = [float(params[k]) for k in axes]

    def unpack(vec: np.ndarray) -> tuple[float, float, float, float, float, float]:
        d = dict(params)
        for key, val in zip(axes, vec):
            d[key] = float(val)
        return constrain_by_system(
            crystal_system, d["a"], d["b"], d["c"], d["alpha"], d["beta"], d["gamma"]
        )

    def make_struct(vec: np.ndarray) -> Structure:
        a, b, c, alpha, beta, gamma = unpack(vec)
        return scale_structure(proto, a, b, c, alpha, beta, gamma)

    def objective(vec: np.ndarray) -> float:
        try:
            px, py, _ = xrd_pattern(make_struct(vec), tmin, tmax)
        except Exception:
            return 1e6
        ycalc = sticks_to_curve(two_theta, px, py, fwhm)
        ycalc = ycalc / (ycalc.max() + 1e-18)
        return r_metrics(yobs, ycalc)["R"]

    bounds = []
    for key, val in zip(axes, x0):
        if key in {"alpha", "beta", "gamma"}:
            bounds.append((max(val - 4.0, 60.0), min(val + 4.0, 130.0)))
        else:
            bounds.append((val * 0.97, val * 1.03))

    result = differential_evolution(objective, bounds, maxiter=16, popsize=8, seed=0, polish=True)
    a, b, c, alpha, beta, gamma = unpack(result.x)
    fitted = make_struct(result.x)
    px, py, hkls = xrd_pattern(fitted, tmin, tmax)
    ycalc = sticks_to_curve(two_theta, px, py, fwhm)
    ycalc = ycalc / (ycalc.max() + 1e-18)
    metrics = r_metrics(yobs, ycalc)
    return {
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "success": bool(result.success),
        **metrics,
        "px": px,
        "py": py,
        "hkls": hkls,
        "structure": fitted,
        "axes": axes,
    }


def experimental_peaks(two_theta: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = max(float(np.percentile(intensity, 70)), 1e-9)
    distance = max(int(len(two_theta) / 400), 3)
    idx, _ = find_peaks(intensity, height=height, distance=distance)
    if len(idx) == 0:
        idx, _ = find_peaks(intensity, prominence=float(np.std(intensity) * 0.5))
    return two_theta[idx], intensity[idx]


def unexplained_peaks(
    peak_x: np.ndarray,
    peak_y: np.ndarray,
    main_x: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(peak_x) == 0 or len(main_x) == 0:
        return peak_x, peak_y
    mask = np.array([np.min(np.abs(main_x - x)) > tol for x in peak_x], dtype=bool)
    return peak_x[mask], peak_y[mask]


def match_score(un_x: np.ndarray, un_y: np.ndarray, imp_x: np.ndarray, tol: float) -> dict[str, float]:
    if len(un_x) == 0:
        return {"score": 0.0, "n_match": 0, "n_unexplained": 0, "intensity_frac": 0.0}
    hits = 0
    hit_i = 0.0
    matched_positions: list[float] = []
    for x, y in zip(un_x, un_y):
        if len(imp_x) and np.min(np.abs(imp_x - x)) <= tol:
            hits += 1
            hit_i += float(y)
            matched_positions.append(float(x))
    intensity_frac = hit_i / (float(np.sum(un_y)) + 1e-18)
    coverage = hits / len(un_x)
    return {
        "score": 0.6 * intensity_frac + 0.4 * coverage,
        "n_match": hits,
        "n_unexplained": int(len(un_x)),
        "intensity_frac": intensity_frac,
        "matched_two_theta": matched_positions,
    }


def peak_shift_summary(peak_x: np.ndarray, peak_y: np.ndarray, main_x: np.ndarray) -> str:
    if len(peak_x) == 0 or len(main_x) == 0:
        return "比較できるピークが不足しています。"
    i = int(np.argmax(peak_y))
    x = float(peak_x[i])
    nearest = float(main_x[np.argmin(np.abs(main_x - x))])
    return (
        f"最強実験ピーク 2θ={x:.3f}°、最近傍の主相理論ピーク {nearest:.3f}°、"
        f"差 {x - nearest:+.3f}°"
    )


def data_trend_summary(two_theta: np.ndarray, intensity: np.ndarray, peak_x: np.ndarray) -> str:
    return (
        f"測定範囲 2θ={two_theta.min():.2f}–{two_theta.max():.2f}°、"
        f"点数 {len(two_theta)}、検出ピーク {len(peak_x)} 本、"
        f"最大強度 {float(np.max(intensity)):.3g}"
    )


def diagnose_with_llm(
    *,
    material_name: str,
    formula: str,
    crystal_system: str,
    space_group: str,
    lattice_text: str,
    synthesis: str,
    trends: str,
    shift_note: str,
    unexplained: np.ndarray,
    candidates: pd.DataFrame,
    api_key: str,
) -> str:
    if not api_key:
        raise RuntimeError("OpenAI APIキーがありません。")

    rows = []
    for _, row in candidates.iterrows():
        rows.append(
            f"- {row['phase']}: score={row['score']:.3f}, "
            f"一致ピーク数={int(row['n_match'])}/{int(row['n_unexplained'])}, "
            f"強度寄与={row['intensity_frac']:.3f}"
        )
    impurity_block = "\n".join(rows) if rows else "（有意な不純物一致なし）"
    un_txt = ", ".join(f"{x:.2f}" for x in unexplained[:20]) if len(unexplained) else "なし"

    prompt = (
        f"解析対象物質: {material_name}（組成の目安: {formula}）\n"
        f"結晶系: {crystal_system} / 空間群: {space_group}\n"
        f"使用した格子定数: {lattice_text}\n"
        f"合成・プロセス情報: {synthesis}\n"
        f"実験XRDの概況: {trends}\n"
        f"ピークシフトの目安: {shift_note}\n"
        f"主相で説明しにくいピーク 2θ (°): {un_txt}\n"
        f"不純物スクリーニング（簡易一致）:\n{impurity_block}\n\n"
        "あなたは粉末XRDと無機合成に詳しい材料科学者です。"
        "特定の物質に固定せず、上記の対象物質・結晶系・データ傾向に基づいて日本語で診断してください。\n"
        "必ず次を含めてください。\n"
        f"1. 見出しで「{material_name} の診断」と明記する。\n"
        "2. 未帰属ピークやシフトから考えられる不純物・格子歪み・固溶・配向の可能性。\n"
        "3. その物質の合成化学・熱力学の観点から、次回改善すべきパラメータ"
        "（モル比、温度、雰囲気、冷却、前駆体、溶媒/塩など）。\n"
        "4. 本スクリーニングは簡易一致であり確定相同定ではないこと。"
    )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "汎用のXRD診断アシスタント。ユーザーが指定した物質と結晶系に合わせて考察する。"
                    "Nb2AlC専用の前提は使わない。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


def make_figure(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    overlays: list[dict[str, Any]],
    unexplained_x: np.ndarray | None = None,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=two_theta,
            y=intensity,
            mode="lines",
            name="実験データ",
            line=dict(color="#1f77b4", width=1.4),
        )
    )
    ymax = float(np.max(intensity)) if len(intensity) else 1.0
    for item in overlays:
        px, py = item["x"], item["y"]
        if len(px) == 0:
            continue
        scale = ymax / (float(np.max(py)) + 1e-18) * item.get("scale", 0.85)
        if item.get("mode") == "sticks":
            xs, ys = [], []
            for x, y in zip(px, py):
                xs.extend([x, x, None])
                ys.extend([0.0, float(y) * scale, None])
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=item["color"], width=1.6),
                    name=item["name"],
                    hovertemplate=f"{item['name']}<br>2θ=%{{x:.2f}}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=item["grid"],
                    y=item["curve"] * (ymax / (item["curve"].max() + 1e-18)) * item.get("scale", 0.7),
                    mode="lines",
                    name=item["name"],
                    line=dict(color=item["color"], width=1.2, dash="dot"),
                )
            )
    if unexplained_x is not None and len(unexplained_x):
        fig.add_trace(
            go.Scatter(
                x=unexplained_x,
                y=np.full(len(unexplained_x), ymax * 1.02),
                mode="markers",
                marker=dict(symbol="x", size=9, color="#d62728"),
                name="未帰属ピーク",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="2θ (°)",
        yaxis_title="Intensity (a.u.)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=520,
    )
    return fig


def lattice_form(system: str, defaults: dict[str, float], key_prefix: str) -> dict[str, float]:
    axes = fit_axes(system)
    cols = st.columns(min(len(axes), 3))
    values = dict(defaults)
    for i, ax in enumerate(axes):
        with cols[i % len(cols)]:
            fmt = "%.2f" if ax in {"alpha", "beta", "gamma"} else "%.4f"
            values[ax] = float(
                st.number_input(
                    f"{ax} ({'°' if ax in {'alpha', 'beta', 'gamma'} else 'Å'})",
                    value=float(defaults[ax]),
                    format=fmt,
                    key=f"{key_prefix}_{ax}",
                )
            )
    a, b, c, alpha, beta, gamma = constrain_by_system(
        system, values["a"], values["b"], values["c"], values["alpha"], values["beta"], values["gamma"]
    )
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def parse_extra_impurities(text: str) -> list[str]:
    names = []
    for part in text.replace("、", ",").split(","):
        name = part.strip()
        if name and name in KNOWN_IMPURITIES:
            names.append(name)
    return names


def main() -> None:
    st.set_page_config(page_title="汎用 XRD 解析・AI診断", layout="wide")

    with st.sidebar:
        st.header("1. 解析対象")
        preset_labels = {p.id: p.name for p in MATERIAL_PRESETS.values()}
        preset_labels[CUSTOM_PRESET_ID] = "その他（カスタム）"
        preset_id = st.selectbox(
            "物質プリセット",
            list(preset_labels.keys()),
            format_func=lambda k: preset_labels[k],
        )
        if preset_id == CUSTOM_PRESET_ID:
            material_name = st.text_input("物質名", value="Custom phase").strip() or "Custom phase"
            formula = st.text_input("組成式（任意）", value="").strip() or material_name
            crystal_system = st.selectbox("結晶系", CRYSTAL_SYSTEMS, index=1)
            space_group = st.text_input("空間群（任意）", value="")
            space_group_number = int(st.number_input("空間群番号（任意）", value=0, step=1))
            defaults = crystal_system_defaults(crystal_system)
            st.caption("新しい物質はプリセット未登録でも、格子定数またはCIFで解析できます。")
        else:
            preset = MATERIAL_PRESETS[preset_id]
            material_name = preset.name
            formula = preset.formula
            crystal_system = preset.crystal_system
            space_group = preset.space_group
            space_group_number = preset.space_group_number
            defaults = preset.lattice_params()
            st.caption(preset.notes)

        st.header("2. 結晶構造")
        cif_file = st.file_uploader("主相CIF（推奨・自動読込）", type=["cif"])
        extra_cifs = st.file_uploader("候補相CIF（任意・複数）", type=["cif"], accept_multiple_files=True)
        cif_struct: Structure | None = None
        cif_meta: dict[str, Any] | None = None
        if cif_file is not None:
            try:
                cif_struct = structure_from_cif(cif_file)
                cif_meta = describe_structure(cif_struct)
                st.success(
                    f"CIF: {cif_meta['formula']} / {cif_meta['crystal_system']} / "
                    f"{cif_meta['space_group']} ({cif_meta['space_group_number']})"
                )
                use_cif_lattice = st.checkbox("CIFの格子定数を初期値にする", value=True)
                if use_cif_lattice:
                    defaults = {
                        "a": cif_meta["a"],
                        "b": cif_meta["b"],
                        "c": cif_meta["c"],
                        "alpha": cif_meta["alpha"],
                        "beta": cif_meta["beta"],
                        "gamma": cif_meta["gamma"],
                    }
                    crystal_system = cif_meta["crystal_system"] or crystal_system
                    space_group = cif_meta["space_group"] or space_group
                    space_group_number = cif_meta["space_group_number"] or space_group_number
                    formula = cif_meta["formula"] or formula
            except Exception as exc:  # noqa: BLE001
                st.error(f"CIF読込失敗: {exc}")
                cif_struct = None

        params = lattice_form(crystal_system, defaults, key_prefix=f"lat_{preset_id}")
        sg_label = space_group or (f"#{space_group_number}" if space_group_number else "未指定")

        st.header("3. 実験データ")
        exp_file = st.file_uploader("実験XRD（.txt / .csv）", type=["txt", "csv"])
        if SAMPLE_XRD_PATH.is_file() and st.button("サンプルデータを読み込む"):
            st.session_state["use_sample_xrd"] = True
        fwhm = st.slider("擬似ピーク半値幅 (°)", 0.05, 0.50, 0.18, 0.01)
        peak_tol = st.slider("ピーク一致許容 (°2θ)", 0.10, 0.80, 0.25, 0.05)
        run_fit = st.checkbox("格子定数を簡易最適化する", value=True)
        show_sim = st.checkbox("理論パターンを曲線でも重ねる", value=True)

        st.header("4. 合成・不純物")
        if preset_id in MATERIAL_PRESETS and MATERIAL_PRESETS[preset_id].synthesis_routes:
            routes = MATERIAL_PRESETS[preset_id].synthesis_routes
            synthesis = st.selectbox("合成方法", list(routes.keys()))
            impurity_names = list(routes[synthesis])
        else:
            synthesis = st.text_input("合成方法（任意）", value="未指定")
            impurity_names = []
        extra_imp_text = st.text_input(
            "追加スクリーニング相（カンマ区切り）",
            value="",
            help=f"登録済み: {', '.join(KNOWN_IMPURITIES)}",
        )
        impurity_names = list(dict.fromkeys(impurity_names + parse_extra_impurities(extra_imp_text)))

        st.header("5. AI診断")
        user_api_key = st.text_input(
            "OpenAI APIキー",
            type="password",
            help="入力したキーを優先。空なら Streamlit Secrets の OPENAI_API_KEY。",
        )

    st.title(f"{material_name} の XRD 解析")
    st.caption(
        f"{formula}　|　結晶系 {crystal_system}　|　空間群 {sg_label}　|　"
        "理論ピーク重ね合わせ・格子フィット・不純物スクリーニング・AI診断"
    )

    if exp_file is not None:
        st.session_state["use_sample_xrd"] = False

    if exp_file is None and not st.session_state.get("use_sample_xrd"):
        st.info("サイドバーから実験XRDをアップロードするか、サンプルデータを読み込んでください。")
        st.stop()

    try:
        data = load_xy(exp_file if exp_file is not None else SAMPLE_XRD_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    two_theta = data["two_theta"].to_numpy(dtype=float)
    intensity = data["intensity"].to_numpy(dtype=float)
    order = np.argsort(two_theta)
    two_theta, intensity = two_theta[order], intensity[order]
    tmin, tmax = float(two_theta.min()), float(two_theta.max())

    proto = cif_struct
    if proto is None:
        proto = structure_from_params(
            preset_id if preset_id in MATERIAL_PRESETS else CUSTOM_PRESET_ID,
            params["a"],
            params["b"],
            params["c"],
            params["alpha"],
            params["beta"],
            params["gamma"],
        )
        if preset_id == CUSTOM_PRESET_ID:
            proto = dummy_structure(
                params["a"], params["b"], params["c"], params["alpha"], params["beta"], params["gamma"]
            )

    main_struct = scale_structure(
        proto, params["a"], params["b"], params["c"], params["alpha"], params["beta"], params["gamma"]
    )

    fit_info: dict[str, Any] | None = None
    if run_fit:
        with st.spinner(f"{material_name} の格子定数を探索しています…"):
            fit_info = fit_lattice(two_theta, intensity, proto, crystal_system, params, fwhm)
            main_struct = fit_info["structure"]
            params = {k: fit_info[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")}

    main_x, main_y, main_hkls = xrd_pattern(main_struct, tmin, tmax)
    main_curve = sticks_to_curve(two_theta, main_x, main_y, fwhm)
    peak_x, peak_y = experimental_peaks(two_theta, intensity)
    un_x, un_y = unexplained_peaks(peak_x, peak_y, main_x, peak_tol)
    shift_note = peak_shift_summary(peak_x, peak_y, main_x)
    trends = data_trend_summary(two_theta, intensity, peak_x)

    overlays = [
        {
            "name": f"{material_name} 理論ピーク",
            "x": main_x,
            "y": main_y,
            "color": "#ff7f0e",
            "mode": "sticks",
            "scale": 0.9,
        }
    ]
    if show_sim:
        overlays.append(
            {
                "name": f"{material_name} シミュレーション",
                "x": main_x,
                "y": main_y,
                "grid": two_theta,
                "curve": main_curve,
                "color": "#ffbb78",
                "mode": "curve",
                "scale": 0.65,
            }
        )

    extra_structs: dict[str, Structure] = {}
    for extra in extra_cifs or []:
        try:
            extra_structs[extra.name] = structure_from_cif(extra)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{extra.name} を読めませんでした: {exc}")

    rows = []
    imp_patterns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in impurity_names:
        try:
            st_imp = impurity_structure(name)
            ix, iy, _ = xrd_pattern(st_imp, tmin, tmax)
            imp_patterns[name] = (ix, iy)
            metrics = match_score(un_x, un_y, ix, peak_tol)
            rows.append({"phase": name, "source": "プロセス推定", **metrics})
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{name} の理論XRD計算に失敗: {exc}")

    for fname, struct in extra_structs.items():
        try:
            ix, iy, _ = xrd_pattern(struct, tmin, tmax)
            metrics = match_score(un_x, un_y, ix, peak_tol)
            rows.append({"phase": fname, "source": "アップロードCIF", **metrics})
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{fname} の理論XRD計算に失敗: {exc}")

    cand_df = pd.DataFrame(rows)
    if not cand_df.empty:
        cand_df = cand_df.sort_values("score", ascending=False).reset_index(drop=True)
        top_name = str(cand_df.iloc[0]["phase"])
        if top_name in imp_patterns:
            ix, iy = imp_patterns[top_name]
            overlays.append(
                {
                    "name": f"最有力不純物 {top_name}",
                    "x": ix,
                    "y": iy,
                    "color": "#9467bd",
                    "mode": "sticks",
                    "scale": 0.55,
                }
            )

    fig = make_figure(
        two_theta,
        intensity,
        overlays,
        unexplained_x=un_x,
        title=f"{material_name}：実験XRDと理論パターン",
    )
    st.plotly_chart(fig, use_container_width=True)

    lattice_text = (
        f"a={params['a']:.4f} Å, b={params['b']:.4f} Å, c={params['c']:.4f} Å, "
        f"α={params['alpha']:.2f}°, β={params['beta']:.2f}°, γ={params['gamma']:.2f}°"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{material_name} の格子定数")
        st.write(lattice_text)
        if fit_info:
            st.write(
                f"R = {fit_info['R']:.4f}　Rwp = {fit_info['Rwp']:.4f}　"
                f"S目安 = {fit_info['S']:.4f}　最適化軸: {', '.join(fit_info['axes'])}"
            )
            st.caption("R は Σ|Iobs−sIcalc|/ΣIobs。結晶系に応じて独立な軸だけ動かします。")
        st.caption(shift_note)
    with col2:
        st.subheader(f"{material_name} の主相ピーク")
        if len(main_x):
            labels = []
            for h in main_hkls:
                if h and isinstance(h[0], dict) and "hkl" in h[0]:
                    labels.append(str(h[0]["hkl"]))
                else:
                    labels.append("")
            st.dataframe(
                pd.DataFrame({"2theta": main_x, "I_rel": main_y, "hkl": labels}),
                hide_index=True,
                height=240,
            )

    st.subheader(f"{material_name} の不純物スクリーニング")
    st.write(
        f"合成情報 **{synthesis}**。候補相: {', '.join(impurity_names) if impurity_names else '（リストなし・CIFのみ）'}。"
        f" 実験ピーク {len(peak_x)} 本のうち未帰属は {len(un_x)} 本。"
    )
    if cand_df.empty:
        st.info("比較できる不純物パターンがありません。候補CIFか登録相名を追加してください。")
    else:
        show = cand_df[["phase", "source", "score", "n_match", "n_unexplained", "intensity_frac"]].copy()
        show["score"] = show["score"].map(lambda v: f"{v:.3f}")
        show["intensity_frac"] = show["intensity_frac"].map(lambda v: f"{v:.3f}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        best = cand_df.iloc[0]
        st.success(
            f"{material_name} で最も怪しい候補: **{best['phase']}**"
            f"（score={best['score']:.3f}、一致 {int(best['n_match'])}/{int(best['n_unexplained'])}）"
        )

    st.subheader(f"{material_name} のAI診断")
    st.write("選択中の物質名・結晶系・XRD傾向・スクリーニング結果から、プロンプトをその場で組み立てます。")
    api_key = get_openai_api_key(user_api_key)
    if not api_key:
        st.warning(
            "OpenAI APIキーが未設定です。サイドバーにキーを入力するか、"
            "Streamlit Cloud の Secrets に OPENAI_API_KEY を設定してください。"
            " キーがない状態では診断APIは呼び出しません。"
        )
    elif st.button(f"{material_name} を診断する", type="primary"):
        try:
            with st.spinner(f"{material_name} についてLLMに問い合わせています…"):
                text = diagnose_with_llm(
                    material_name=material_name,
                    formula=formula,
                    crystal_system=crystal_system,
                    space_group=sg_label,
                    lattice_text=lattice_text,
                    synthesis=synthesis,
                    trends=trends,
                    shift_note=shift_note,
                    unexplained=un_x,
                    candidates=cand_df if not cand_df.empty else pd.DataFrame(),
                    api_key=api_key,
                )
            st.markdown(text)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


if __name__ == "__main__":
    main()
