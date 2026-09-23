"""Nb2AlC XRD 解析の最小アプリ。起動: streamlit run app.py"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

CU_KA = 1.5406
DEFAULT_A = 3.107
DEFAULT_C = 13.888

SYNTHESIS_IMPURITIES: dict[str, list[str]] = {
    "HFエッチング法": ["NbC", "Al2O3", "Nb", "Al", "C"],
    "溶融塩法": ["NbC", "Al2O3", "Nb", "Al", "NaCl"],
    "反応焼結法": ["NbC", "Al2O3", "Nb", "Al", "NbAl3"],
}


def nb2alc_structure(a: float = DEFAULT_A, c: float = DEFAULT_C) -> Structure:
    """P63/mmc の Nb2AlC（211 MAX）簡易構造。"""
    return Structure.from_spacegroup(
        194,
        Lattice.hexagonal(a, c),
        ["Nb", "Al", "C"],
        [[1.0 / 3.0, 2.0 / 3.0, 0.088], [1.0 / 3.0, 2.0 / 3.0, 0.75], [0.0, 0.0, 0.0]],
    )


def impurity_structure(name: str) -> Structure:
    """スクリーニング用の簡易結晶構造（文献値に近い格子定数）。"""
    builders = {
        "NbC": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(4.470), ["Nb", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "Al2O3": lambda: Structure.from_spacegroup(
            167,
            Lattice.hexagonal(4.759, 12.991),
            ["Al", "O"],
            [[0.0, 0.0, 0.3523], [0.3064, 0.0, 0.25]],
        ),
        "Nb": lambda: Structure.from_spacegroup(229, Lattice.cubic(3.300), ["Nb"], [[0, 0, 0]]),
        "Al": lambda: Structure.from_spacegroup(225, Lattice.cubic(4.050), ["Al"], [[0, 0, 0]]),
        "C": lambda: Structure.from_spacegroup(227, Lattice.cubic(3.567), ["C"], [[0, 0, 0]]),
        "NaCl": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "NbAl3": lambda: Structure.from_spacegroup(
            139,
            Lattice.tetragonal(3.845, 8.601),
            ["Nb", "Al", "Al"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.25]],
        ),
    }
    if name not in builders:
        raise KeyError(f"未登録の不純物: {name}")
    return builders[name]()


def load_xy(file) -> pd.DataFrame:
    raw = file.read()
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
    a0: float,
    c0: float,
    fwhm: float,
) -> dict[str, Any]:
    tmin, tmax = float(two_theta.min()), float(two_theta.max())
    yobs = intensity.astype(float)
    yobs = yobs / (yobs.max() + 1e-18)

    def objective(vec: np.ndarray) -> float:
        a, c = float(vec[0]), float(vec[1])
        try:
            px, py, _ = xrd_pattern(nb2alc_structure(a, c), tmin, tmax)
        except Exception:
            return 1e6
        ycalc = sticks_to_curve(two_theta, px, py, fwhm)
        ycalc = ycalc / (ycalc.max() + 1e-18)
        return r_metrics(yobs, ycalc)["R"]

    bounds = [(a0 * 0.97, a0 * 1.03), (c0 * 0.97, c0 * 1.03)]
    result = differential_evolution(objective, bounds, maxiter=18, popsize=8, seed=0, polish=True)
    a_fit, c_fit = float(result.x[0]), float(result.x[1])
    px, py, hkls = xrd_pattern(nb2alc_structure(a_fit, c_fit), tmin, tmax)
    ycalc = sticks_to_curve(two_theta, px, py, fwhm)
    ycalc = ycalc / (ycalc.max() + 1e-18)
    metrics = r_metrics(yobs, ycalc)
    return {"a": a_fit, "c": c_fit, "success": bool(result.success), **metrics, "px": px, "py": py, "hkls": hkls}


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
    if len(peak_x) == 0:
        return peak_x, peak_y
    if len(main_x) == 0:
        return peak_x, peak_y
    keep = []
    for x in peak_x:
        keep.append(np.min(np.abs(main_x - x)) > tol)
    mask = np.asarray(keep, dtype=bool)
    return peak_x[mask], peak_y[mask]


def match_score(un_x: np.ndarray, un_y: np.ndarray, imp_x: np.ndarray, tol: float) -> dict[str, float]:
    if len(un_x) == 0:
        return {"score": 0.0, "n_match": 0, "n_unexplained": 0, "intensity_frac": 0.0}
    hits = 0
    hit_i = 0.0
    matched_positions = []
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


def diagnose_with_llm(synthesis: str, candidates: pd.DataFrame) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")

    rows = []
    for _, row in candidates.iterrows():
        rows.append(
            f"- {row['phase']}: score={row['score']:.3f}, "
            f"一致ピーク数={int(row['n_match'])}/{int(row['n_unexplained'])}, "
            f"強度寄与={row['intensity_frac']:.3f}"
        )
    impurity_block = "\n".join(rows) if rows else "（有意な不純物一致なし）"

    prompt = (
        "あなたはMAX相（特にNb2AlC）の合成とXRDに詳しい材料科学者です。\n"
        f"採用した合成方法: {synthesis}\n"
        "XRDの未帰属ピークと理論パターンを比較した不純物スクリーニング結果:\n"
        f"{impurity_block}\n\n"
        "次の2点を、化学・熱力学の背景を踏まえて日本語で具体的に考察してください。\n"
        "1. なぜその不純物が生成しうるのか。\n"
        "2. 次回合成で改善すべきパラメータ（原料モル比、温度、保持時間、雰囲気、塩やエッチング条件など）。\n"
        "断定しすぎず、XRDスクリーニングは簡易一致であることも一言添えてください。"
    )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "材料合成プロセスの診断アシスタント。"},
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
        xaxis_title="2θ (°)",
        yaxis_title="Intensity (a.u.)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40),
        height=520,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Nb2AlC XRD 解析", layout="wide")
    st.title("Nb2AlC XRD 解析ベース")
    st.caption("読み込み・Plotly表示・格子定数の簡易フィット・理論パターン重ね合わせ・不純物スクリーニング")

    with st.sidebar:
        st.header("データ")
        exp_file = st.file_uploader("実験XRD（.txt / .csv）", type=["txt", "csv"])
        cif_file = st.file_uploader("主相CIF（任意）", type=["cif"])
        extra_cifs = st.file_uploader("候補物質CIF（任意・複数）", type=["cif"], accept_multiple_files=True)

        st.header("測定・構造")
        a0 = st.number_input("初期格子定数 a (Å)", value=DEFAULT_A, format="%.4f")
        c0 = st.number_input("初期格子定数 c (Å)", value=DEFAULT_C, format="%.4f")
        fwhm = st.slider("擬似ピーク半値幅 (°)", 0.05, 0.50, 0.18, 0.01)
        peak_tol = st.slider("ピーク一致許容 (°2θ)", 0.10, 0.80, 0.25, 0.05)

        st.header("合成プロセス")
        synthesis = st.selectbox("合成方法", list(SYNTHESIS_IMPURITIES.keys()))
        st.caption("選択に応じて混入しやすい不純物リストを自動ロードします。")

        run_fit = st.checkbox("a, c を簡易最適化する", value=True)
        show_sim = st.checkbox("理論パターンを曲線でも重ねる", value=True)

    if exp_file is None:
        st.info("左のサイドバーから 2θ と Intensity の実験データをアップロードしてください。")
        st.stop()

    try:
        data = load_xy(exp_file)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    two_theta = data["two_theta"].to_numpy(dtype=float)
    intensity = data["intensity"].to_numpy(dtype=float)
    order = np.argsort(two_theta)
    two_theta, intensity = two_theta[order], intensity[order]
    tmin, tmax = float(two_theta.min()), float(two_theta.max())

    if cif_file is not None:
        try:
            main_struct = structure_from_cif(cif_file)
            st.sidebar.success(f"CIFを読み込みました: {cif_file.name}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"CIF読込失敗のため内蔵Nb2AlCを使用: {exc}")
            main_struct = nb2alc_structure(a0, c0)
    else:
        main_struct = nb2alc_structure(a0, c0)

    fit_info: dict[str, Any] | None = None
    if run_fit:
        with st.spinner("格子定数 a, c を探索しています…"):
            fit_info = fit_lattice(two_theta, intensity, a0, c0, fwhm)
            main_struct = nb2alc_structure(fit_info["a"], fit_info["c"])

    main_x, main_y, main_hkls = xrd_pattern(main_struct, tmin, tmax)
    main_curve = sticks_to_curve(two_theta, main_x, main_y, fwhm)

    peak_x, peak_y = experimental_peaks(two_theta, intensity)
    un_x, un_y = unexplained_peaks(peak_x, peak_y, main_x, peak_tol)

    overlays = [
        {
            "name": "主相 理論ピーク",
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
                "name": "主相 シミュレーション",
                "x": main_x,
                "y": main_y,
                "grid": two_theta,
                "curve": main_curve,
                "color": "#ffbb78",
                "mode": "curve",
                "scale": 0.65,
            }
        )

    impurity_names = list(SYNTHESIS_IMPURITIES[synthesis])
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
            rows.append({"phase": name, "source": "合成ルート推定", **metrics})
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

    fig = make_figure(two_theta, intensity, overlays, unexplained_x=un_x)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("格子定数フィット")
        if fit_info:
            st.metric("a (Å)", f"{fit_info['a']:.4f}")
            st.metric("c (Å)", f"{fit_info['c']:.4f}")
            st.write(
                f"R = {fit_info['R']:.4f}　Rwp = {fit_info['Rwp']:.4f}　"
                f"S目安 = {fit_info['S']:.4f}"
            )
            st.caption("R は Σ|Iobs−sIcalc|/ΣIobs。S は残差二乗平均の目安です（簡易モデル）。")
        else:
            st.write(f"初期値のまま: a={a0:.4f} Å, c={c0:.4f} Å")

    with col2:
        st.subheader("主相ピーク")
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

    st.subheader("不純物候補の自動スクリーニング")
    st.write(
        f"合成方法 **{synthesis}** の一般的不純物: {', '.join(impurity_names)}。"
        f" 実験ピーク {len(peak_x)} 本のうち、主相で説明できないピークは {len(un_x)} 本です。"
    )
    if cand_df.empty:
        st.info("比較できる不純物パターンがありません。")
    else:
        show = cand_df[["phase", "source", "score", "n_match", "n_unexplained", "intensity_frac"]].copy()
        show["score"] = show["score"].map(lambda v: f"{v:.3f}")
        show["intensity_frac"] = show["intensity_frac"].map(lambda v: f"{v:.3f}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        best = cand_df.iloc[0]
        st.success(
            f"最も怪しい候補: **{best['phase']}**（score={best['score']:.3f}、"
            f"未帰属ピーク一致 {int(best['n_match'])}/{int(best['n_unexplained'])}）"
        )

    st.subheader("AI合成プロセス診断アシスタント")
    st.write("選択中の合成方法と、上のスクリーニング結果をLLMに渡して考察します。")
    if st.button("診断を実行", type="primary"):
        try:
            with st.spinner("LLMに問い合わせています…"):
                text = diagnose_with_llm(synthesis, cand_df if not cand_df.empty else pd.DataFrame())
            st.markdown(text)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.info("PowerShell 例: `$env:OPENAI_API_KEY=\"sk-...\"` のあと `streamlit run app.py`")


if __name__ == "__main__":
    main()
