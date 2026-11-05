#!/usr/bin/env python3
"""
generar_figuras.py — Regenera las Figuras 2.3, 2.4, 3.1, 3.2 y 3.3 de la tesis a
partir de resultados/resultados_experimentos.json (salida de ejecutar_experimentos.py).
Las Figuras 3.2 y 3.3 son representaciones gráficas de la salida del framework
(no capturas de pantalla de Streamlit).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "resultados" / "resultados_experimentos.json").read_text(encoding="utf-8"))
OUT = HERE / "resultados" / "figuras"; OUT.mkdir(exist_ok=True, parents=True)
G, A, R_, B = "#1D9E75", "#F0A02B", "#E34A4A", "#4C9AFF"
INK, MUTE = "#1A1A2E", "#6B7280"
plt.rcParams["font.family"] = "DejaVu Sans"


def card(ax, x, y, w, h, fc, ec, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))


# ----------------------------------------------------------------- Fig 2.3
def fig_2_3():
    f = RES["figura_2_3"]; c = RES["cobertura_test"]; pt = f["paciente"]
    fig = plt.figure(figsize=(13.39, 7.01), dpi=100, facecolor="#F9FAFB")
    ax = fig.add_axes([0.05, 0.14, 0.40, 0.66]); fig.text(0.25, 0.90, f"Distribución bootstrap (B={RES['entorno']['B']} réplicas)",
                                                          ha="center", fontsize=15, fontweight="bold", color=INK)
    preds = np.array(f["preds"]); lo, hi = f["lo"], f["hi"]
    bins = np.linspace(preds.min(), preds.max(), 26)
    n, edges, patches = ax.hist(preds, bins=bins, edgecolor="white")
    for p, e0, e1 in zip(patches, edges[:-1], edges[1:]):
        p.set_facecolor(G if (e1 >= lo and e0 <= hi) else "#CBD5E1")
    ax.axvline(f["pred"], color="#065F46", lw=2.5)
    ax.text(f["pred"], n.max() * 1.04, f" {f['pred']:.1f} h", color="#065F46", fontweight="bold", fontsize=12)
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]
    ax.set_xlabel(f"Media simulada de «Estadía post-UCI» (h) — paciente {pt['age']} años, APACHE {pt['apache']}", color=MUTE, fontsize=10)
    ax.text(0.02, -0.17, f"■ IC 95%: [{lo:.1f}, {hi:.1f}] h", color=G, transform=ax.transAxes, fontsize=10)
    ax2 = fig.add_axes([0.50, 0.04, 0.47, 0.88]); ax2.axis("off")
    ax2.text(0.5, 0.95, "Comparación de métodos de intervalo (95 %)", ha="center", fontsize=15, fontweight="bold", color=INK)
    rows = [("Conformal prediction (split)", "conformal_95", "#E6FAF3", G, "#064E3B", "Garantía formal de cobertura marginal"),
            ("Bootstrap por perturbación de entradas", "bootstrap_95", "#FDF8EC", A, "#78350F", "Sin garantía formal de cobertura"),
            ("Monte Carlo", "montecarlo_95", "#EEF4FF", B, "#1E3A8A", "Sin garantía formal de cobertura")]
    for i, (t, k, fc, ec, tc, sub) in enumerate(rows):
        y = 0.66 - i * 0.29; card(ax2, 0.03, y, 0.94, 0.24, fc, ec)
        ax2.text(0.5, y + 0.185, t, ha="center", fontsize=13, fontweight="bold", color=tc)
        ax2.text(0.5, y + 0.115, f"Cobertura (prueba, n={c[k]['n']}): {100*c[k]['cobertura']:.1f}%  |  Amplitud media: {c[k]['amplitud_media']:.1f} h",
                 ha="center", fontsize=10.5, color=tc)
        ax2.text(0.5, y + 0.045, sub, ha="center", fontsize=9.5, color=MUTE)
    ax2.text(0.03, 0.03, "Referencia: salida simulada (n_runs=50), no un desenlace clínico observado.\n"
             "Simulador sustituto del framework; repetir con el motor real de SimUci.", fontsize=8.5, color=MUTE, va="top")
    fig.savefig(OUT / "fig_2_3.png", dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


# ----------------------------------------------------------------- Fig 2.4
def fig_2_4():
    R = RES["indice_R"]; th = RES["stress_tests"]["umbral_shift_calibrado"]
    fig = plt.figure(figsize=(13.42, 7.36), dpi=100, facecolor="#F9FAFB")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.037, 0.13), 0.11, 0.80, boxstyle="round,pad=0,rounding_size=0.05", fc="#111", ec="none"))
    for cy, col, on in ((0.80, "#DC2626", False), (0.52, "#D97706", False), (0.24, "#16A34A", True)):
        ax.add_patch(plt.Circle((0.092, cy), 0.052, fc=col if on else {"#DC2626": "#3A0000", "#D97706": "#3F2A00"}[col], ec="none"))
    for i, (t, sub, fc, ec, tc) in enumerate([("VERDE — R ≥ 0.70", "Alta confianza", "#E6FAF3", G, "#064E3B"),
                                               ("ÁMBAR — 0.40 ≤ R < 0.70", "Confianza moderada", "#FDF8EC", A, "#78350F"),
                                               ("ROJO — R < 0.40", "Baja confianza", "#FEEEEE", R_, "#7F1D1D")]):
        y = 0.68 - i * 0.28; card(ax, 0.17, y, 0.24, 0.24, fc, ec)
        ax.text(0.29, y + 0.165, t, ha="center", fontsize=13, fontweight="bold", color=tc)
        ax.text(0.29, y + 0.08, sub, ha="center", fontsize=11, color=tc)
    card(ax, 0.45, 0.06, 0.52, 0.88, "white", "#DDE1E6")
    ax.text(0.71, 0.88, "Cálculo del Índice de Robustez R", ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.71, 0.80, "R = 0.4·r_ruido + 0.4·r_faltante + 0.2·r_shift", ha="center", fontsize=12.5, color="#374151")
    for i, tx in enumerate(["r_ruido    = 1 − min(1, Δ_ruido (σ=0.25) / umbral)", "r_faltante = 1 − min(1, Δ_falt (10 %) / umbral)",
                            "r_shift    = 1 − min(1, Δ_shift / umbral_shift)"]):
        ax.text(0.47, 0.70 - i * 0.055, tx, fontsize=11, color="#374151", family="DejaVu Sans Mono")
    for i, tx in enumerate(["umbral = 25 % de variación relativa (CLINICAL_THRESHOLD)",
                            f"umbral_shift = {100*th:.1f} % (percentil 50 sobre 30 pacientes de la referencia, acotado a [25 %, 150 %])",
                            "Pesos 0.4 / 0.4 / 0.2: PROPUESTOS por criterio de diseño (no estimados)"]):
        ax.text(0.71, 0.49 - i * 0.065, tx, ha="center", fontsize=9.8, color=MUTE)
    ax.text(0.71, 0.24, "R ∈ [0,1] — mayor valor = mayor robustez", ha="center", fontsize=13, fontweight="bold", color="#0B6B4F")
    ax.text(0.71, 0.17, f"(en esta ejecución, n={RES['stress_tests']['n_pacientes']}: {R['verde']} verde, {R['ambar']} ámbar, {R['rojo']} rojo)",
            ha="center", fontsize=9.8, color=MUTE)
    fig.savefig(OUT / "fig_2_4.png", dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


# ----------------------------------------------------------------- Fig 3.1
def fig_3_1():
    T = RES["stress_tests"]["tabla"]; R = RES["indice_R"]
    fig, axs = plt.subplots(2, 2, figsize=(13.39, 10.0), dpi=100, facecolor="#F9FAFB")
    col = lambda v: G if v < 0.10 else A if v < 0.25 else R_
    names = {"apache": "APACHE II", "uti_stay": "EstadiaUCI", "vam_time": "TiempoVAM", "age": "Edad", "preuti_stay": "Est. PreUCI"}
    ax = axs[0, 0]; rows = T[0:3]
    ax.barh([r["nivel"].replace("sigma=", "σ = ") for r in rows][::-1], [100 * r["variacion"] for r in rows][::-1],
            color=[col(r["variacion"]) for r in rows][::-1])
    for i, r in enumerate(rows[::-1]): ax.text(100 * r["variacion"] + 0.4, i, f"{100*r['variacion']:.1f}%  {r['estado']}", va="center", fontsize=11)
    ax.axvline(25, color=MUTE, ls=":"); ax.set_title("Stress-test: ruido gaussiano", fontweight="bold", color=INK); ax.set_xlim(0, 45)
    ax.set_xlabel("Variación relativa media (%) — umbral clínico 25 %", color=MUTE)
    ax = axs[0, 1]; rows = T[3:7]
    ax.barh([r["nivel"] for r in rows][::-1], [100 * r["variacion"] for r in rows][::-1], color=[col(r["variacion"]) for r in rows][::-1])
    for i, r in enumerate(rows[::-1]): ax.text(100 * r["variacion"] + 0.4, i, f"{100*r['variacion']:.1f}%  {r['estado']}", va="center", fontsize=11)
    ax.axvline(25, color=MUTE, ls=":"); ax.set_title("Stress-test: datos faltantes", fontweight="bold", color=INK); ax.set_xlim(0, 45)
    ax.set_xlabel("Variación relativa media (%) — umbral clínico 25 %", color=MUTE)
    ax = axs[1, 0]; o = RES["oat"]; ks = list(o)
    ax.barh([names[k] for k in ks][::-1], [o[k] for k in ks][::-1], color=B)
    for i, k in enumerate(ks[::-1]): ax.text(o[k] + 0.01, i, f"{o[k]:.2f}", va="center", fontsize=11)
    ax.set_xlim(0, 1.15); ax.set_title("Sensibilidad OAT (índice normalizado medio, n=41)", fontweight="bold", color=INK)
    ax = axs[1, 1]; v = np.array(R["R_valores"])
    ax.hist(v, bins=np.linspace(0.5, 0.8, 16), color=A, edgecolor="white")
    ax.axvline(0.70, color=G, ls="--", lw=2); ax.axvline(0.40, color=R_, ls="--", lw=2)
    ax.set_title(f"Distribución del índice R (n={len(v)})", fontweight="bold", color=INK)
    ax.set_xlabel(f"R  —  verde {R['verde']} ({100*R['verde']/len(v):.0f}%) · ámbar {R['ambar']} ({100*R['ambar']/len(v):.0f}%) · rojo {R['rojo']}", color=MUTE)
    for a in axs.ravel(): [a.spines[s].set_visible(False) for s in ("top", "right")]; a.set_facecolor("white")
    fig.text(0.5, 0.01, "Simulador sustituto; cohorte de prueba (n=41). La relación entre R y el error de predicción no se evaluó.", ha="center", fontsize=9, color=MUTE)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(OUT / "fig_3_1.png", dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


# ----------------------------------------------------------------- Fig 3.2 / 3.3
def dashboard(key, fname, dx):
    c = RES["casos"][key]; p = c["perfil"]
    colr = {"green": "#3DBE55", "amber": "#D39A22", "red": "#F25555"}[c["semaforo"]]
    word = {"green": "VERDE", "amber": "ÁMBAR", "red": "ROJO"}[c["semaforo"]]
    lab = {"green": "Alta confianza", "amber": "Confianza moderada", "red": "Baja confianza"}[c["semaforo"]]
    bg, panel, tx, mute = "#0B0F19", "#20242D", "#EEF1F6", "#98A0AE"
    fig = plt.figure(figsize=(10.4, 8.6), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.022, 0.968, "SimUci  — Marco de Incertidumbre y Robustez", fontsize=15, fontweight="bold", color="#59A5FF")
    ax.text(0.022, 0.935, f"Perfil: {p['age']} años · {dx} · APACHE II={p['apache']} · TiempoVAM={p['vam_time']} h · EstadiaUCI={p['uti_stay']} h · Est. PreUCI={p['preuti_stay']} h",
            fontsize=8.6, color=mute)
    kp = [("Predicción", f"{c['pred_sustituto']:.1f} horas"), ("IC 95% (conformal)", f"[{c['ic_lo']:.1f}, {c['ic_hi']:.1f}]"),
          ("Índice R", f"{c['R']:.2f}"), ("Semáforo", word)]
    for i, (a, b) in enumerate(kp):
        x = 0.022 + i * 0.2455; card(ax, x, 0.83, 0.232, 0.085, panel, panel)
        ax.text(x + 0.012, 0.893, a, fontsize=9.5, color=mute)
        ax.text(x + 0.012, 0.848, b, fontsize=15, fontweight="bold", color=colr if i >= 2 else tx)
    card(ax, 0.022, 0.775, 0.955, 0.04, panel, panel)
    lo, hi, pr = c["ic_lo"], c["ic_hi"], c["pred_sustituto"]; xmax = max(hi * 1.1, 1)
    x0, w = 0.03, 0.70
    ax.add_patch(plt.Rectangle((x0 + w * lo / xmax, 0.783), w * (hi - lo) / xmax, 0.024, fc=colr, alpha=0.45, ec="none"))
    ax.add_patch(plt.Rectangle((x0 + w * pr / xmax - 0.002, 0.779), 0.004, 0.032, fc=colr, ec="none"))
    ax.text(x0 + w * lo / xmax + 0.006, 0.7885, f"{lo:.1f} h", fontsize=8, color=tx); ax.text(x0 + w * hi / xmax - 0.055, 0.7885, f"{hi:.1f} h", fontsize=8, color=tx)
    ax.text(0.76, 0.7885, "IC 95% · horas post-egreso UCI", fontsize=8, color=mute)
    axs = fig.add_axes([0.07, 0.44, 0.52, 0.30], facecolor=panel)
    t = np.linspace(0, max(hi * 1.5, 40), 200)
    axs.fill_between(t, np.exp(-t / max(hi, .1)), np.exp(-t / max(lo, .1)), color=colr, alpha=0.2, label="Banda IC 95%")
    axs.plot(t, np.exp(-t / pr), color=colr, lw=2.5, label="Supervivencia estimada"); axs.axvline(pr, color=mute, ls="--", lw=1)
    axs.set_ylim(0, 1.05); axs.set_xlabel("Horas post-egreso UCI", color=mute, fontsize=9); axs.set_ylabel("P(supervivencia)", color=mute, fontsize=9)
    axs.tick_params(colors=mute, labelsize=8); [s.set_visible(False) for s in axs.spines.values()]; axs.grid(alpha=0.12)
    axs.legend(facecolor=panel, edgecolor="none", labelcolor=tx, fontsize=8, loc="upper right")
    axs.set_title("Trayectoria de supervivencia post-egreso UCI (modelo exponencial del dashboard)", color=tx, fontsize=9.5, loc="left", fontweight="bold")
    card(ax, 0.63, 0.44, 0.347, 0.30, panel, panel)
    ax.text(0.65, 0.71, "Semáforo de confianza", fontsize=12, fontweight="bold", color=tx)
    ax.text(0.65, 0.665, f"R = {c['R']:.2f}", fontsize=18, fontweight="bold", color=colr); ax.text(0.65, 0.635, lab, fontsize=10.5, color=colr)
    for i, (n_, v, w_) in enumerate([("r_ruido (0.4)", c["r_noise"], 0.4), ("r_faltante (0.4)", c["r_missing"], 0.4), ("r_shift (0.2)", c["r_shift"], 0.2)]):
        y = 0.585 - i * 0.04; ax.text(0.65, y, n_, fontsize=8.5, color=mute)
        ax.add_patch(plt.Rectangle((0.80, y - 0.004), 0.12, 0.016, fc="#30343D", ec="none")); ax.add_patch(plt.Rectangle((0.80, y - 0.004), 0.12 * v, 0.016, fc=colr, ec="none"))
        ax.text(0.93, y, f"{v:.2f}", fontsize=8.5, color=tx)
    ax.text(0.65, 0.462, f"Umbral base 25 %  |  umbral shift calibrado {100*RES['stress_tests']['umbral_shift_calibrado']:.1f} %", fontsize=7.5, color=mute)
    col = lambda v: "#3DBE55" if v < 0.10 else "#D39A22" if v < 0.25 else "#F25555"
    for j, (ttl, labs, vals) in enumerate([("Ruido gaussiano (var. %)", ["σ = 0.1", "σ = 0.25", "σ = 0.5"], c["ruido"]),
                                           ("Datos faltantes (var. %)", ["5% ausentes", "10% ausentes", "20% ausentes", "30% ausentes"], c["faltantes"])]):
        a = fig.add_axes([0.10 + j * 0.31, 0.17, 0.20, 0.20], facecolor=panel)
        a.barh(labs[::-1], [100 * v for v in vals][::-1], color=[col(v) for v in vals][::-1])
        for i, v in enumerate(vals[::-1]): a.text(100 * v + 0.4, i, f"{100*v:.1f}%", va="center", color=tx, fontsize=8)
        a.axvline(25, color=R_, ls=":", lw=1); a.set_xlim(0, 45); a.set_title(ttl, color=tx, fontsize=9.5, loc="left", fontweight="bold")
        a.tick_params(colors=mute, labelsize=8); [s.set_visible(False) for s in a.spines.values()]
    a = fig.add_axes([0.745, 0.17, 0.22, 0.20], facecolor=panel); nm = {"apache": "APACHE II", "uti_stay": "EstadiaUCI", "vam_time": "TiempoVAM", "age": "Edad", "preuti_stay": "Est. PreUCI"}
    ks = list(c["oat"]); a.barh([nm[k] for k in ks][::-1], [c["oat"][k] for k in ks][::-1], color=B)
    for i, k in enumerate(ks[::-1]): a.text(c["oat"][k] + 0.02, i, f"{c['oat'][k]:.2f}", va="center", color=tx, fontsize=8)
    a.set_xlim(0, 1.2); a.set_title("Sensibilidad OAT (norm.)", color=tx, fontsize=9.5, loc="left", fontweight="bold"); a.tick_params(colors=mute, labelsize=8)
    [s.set_visible(False) for s in a.spines.values()]
    card(ax, 0.022, 0.015, 0.955, 0.115, panel, panel); ax.text(0.04, 0.108, "Recomendaciones del framework", fontsize=10.5, fontweight="bold", color=tx)
    top = ks[0]
    rec = {"green": ("[OK] Predicción confiable. Puede usarse como referencia primaria para planificación del alta y seguimiento post-UCI.", "#3DBE55"),
           "amber": ("[!] Confianza moderada. Integre la predicción del modelo con la evaluación clínica directa del paciente.", "#D39A22"),
           "red": ("[!!] Alta fragilidad detectada. Priorice el criterio médico experto.", "#F25555")}[c["semaforo"]]
    ax.text(0.04, 0.083, rec[0], fontsize=8, color=rec[1]); ax.text(0.04, 0.058, f"[IC] Ventana de seguimiento: [{lo:.1f}, {hi:.1f}] horas post-egreso UCI (IC 95%).", fontsize=8, color="#59A5FF")
    ax.text(0.04, 0.033, f"[REG] Registrar con máxima precisión: {nm[top]} (índice de sensibilidad {c['oat'][top]:.2f}).", fontsize=8, color="#D39A22")
    fig.savefig(OUT / fname, dpi=100, facecolor=bg); plt.close(fig)


if __name__ == "__main__":
    fig_2_3(); fig_2_4(); fig_3_1()
    dashboard("caso1_mayor_R", "fig_3_2.png", "Diag. ingreso: EPOC descompensada (d1=14)")
    dashboard("caso2_menor_R", "fig_3_3.png", "Diag. ingreso: Intoxicación exógena (d1=1)")
    print("figuras ->", OUT)
