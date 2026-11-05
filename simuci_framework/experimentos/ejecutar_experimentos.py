#!/usr/bin/env python3
"""
ejecutar_experimentos.py
========================
Ejecución ÚNICA y reproducible de todas las cifras del Capítulo 3 de la tesis
(Tablas 3.1-3.4, Figuras 2.3, 3.1-3.3 y casos clínicos).

Uso (desde la raíz del proyecto SimUci-Streamlit):
    python simuci_framework/experimentos/ejecutar_experimentos.py \
        [--csv data/fichero_datos_MO_17_1_2023.csv] [--seed 42] [--B 1000]

Qué hace
--------
1. Describe la cohorte REAL (CSV de 208 pacientes).
2. Ajusta distribuciones (DistributionFitter) a las variables de estadía.
3. Divide la cohorte en entrenamiento (60 %), calibración (20 %) y prueba (20 %)
   -> 126 / 41 / 41 pacientes con semilla fija. La prueba no interviene ni en el
   entrenamiento del modelo sustituto ni en la calibración conformal.
4. Cobertura empírica y amplitud de conformal / bootstrap / Monte Carlo (90 % y
   95 %) sobre el conjunto de prueba, y cobertura conformal en 200 particiones
   aleatorias repetidas (global y por estrato de APACHE II).
5. Stress-tests e índice R de los pacientes de prueba, sensibilidad OAT, impacto
   por variable ausente y sensibilidad del semáforo a los pesos de R.
6. Casos ilustrativos (mayor y menor R del conjunto de prueba).

Motor de simulación
-------------------
Si el paquete ``simuci`` (motor SimPy real) está instalado y el proyecto es
importable, el adaptador usa ``run_experiment()`` real. En caso contrario usa el
simulador sustituto (_StubSimulator) y el JSON de salida lo indica en
``entorno.modo_motor``. Los valores de referencia de la cobertura son las
salidas simuladas (media de «Estadia Post UCI», n_runs=50), NO desenlaces
clínicos observados.

Salida: simuci_framework/experimentos/resultados/resultados_experimentos.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simuci_framework.framework.simuci_adapter import (  # noqa: E402
    ALL_FEATURES, NUMERIC_COLS, TARGET_COL, SimUciAdapter,
)
from simuci_framework.framework.uncertainty_module import (  # noqa: E402
    ConformalPredictor, DistributionFitter, UncertaintyQuantifier,
)
from simuci_framework.framework.robustness_module import (  # noqa: E402
    RobustnessEvaluator, W_NOISE, W_MISSING, W_SHIFT, _semaphore,
    CLINICAL_THRESHOLD,
)

warnings.filterwarnings("ignore")
OUT_DIR = Path(__file__).resolve().parent / "resultados"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson al 95 % para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def iqr(x):
    return [float(np.percentile(x, 25)), float(np.percentile(x, 75))]


def msd(x):
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)),
                min=float(np.min(x)), max=float(np.max(x)))


def med(x):
    return dict(median=float(np.median(x)), q1=iqr(x)[0], q3=iqr(x)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "fichero_datos_MO_17_1_2023.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=1000, help="réplicas bootstrap / Monte Carlo")
    ap.add_argument("--reps", type=int, default=200, help="particiones repetidas (conformal)")
    args = ap.parse_args()
    SEED = args.seed
    t0 = time.time()
    res: dict = {}

    # ------------------------------------------------------------------ 0
    adapter = SimUciAdapter(csv_path=args.csv, calibration_split=0.20,
                            test_split=0.20, seed=SEED)
    res["entorno"] = dict(
        modo_motor="stub (simulador sustituto)" if adapter.stub_mode else "SimUci real",
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, pandas=pd.__version__, semilla=SEED, B=args.B,
        csv=str(args.csv),
        n_cohorte=adapter.cohort_size, n_entrenamiento=adapter.train_size,
        n_calibracion=adapter.calibration_size, n_prueba=adapter.test_size,
    )
    print(res["entorno"])

    raw = pd.read_csv(args.csv, low_memory=False)

    # ------------------------------------------------------------------ 1
    try:
        from utils.constants.mappings import PREUCI_DIAG
    except Exception:  # pragma: no cover
        PREUCI_DIAG = {}
    dx = raw["Diag.Ing1"].value_counts()
    top_dx = [dict(codigo=int(c), nombre=PREUCI_DIAG.get(int(c), str(c)), n=int(n),
                   pct=float(100 * n / len(raw))) for c, n in dx.head(5).items()]
    uti_h = raw["Est. UCI"] * 24
    pre_h = raw["Est. PreUCI"] * 24
    post_h = raw["Est. PostUCI"] * 24
    res["cohorte"] = dict(
        n=int(len(raw)),
        edad=msd(raw["Edad"]), apache=msd(raw["APACHE"]),
        tiempo_vam_h=med(raw["TiempoVAM"]), estadia_uci_h=med(uti_h),
        est_preuci_h=med(pre_h), est_postuci_h=med(post_h),
        vam_gt_uci=int((raw["TiempoVAM"] > uti_h).sum()),
        fallece=int(raw["Evolución"].sum()),
        mortalidad_pct=float(100 * raw["Evolución"].mean()),
        top_diagnosticos=top_dx,
        fuera_de_limites_app=dict(
            uti_stay_gt_200h=int((uti_h > 200).sum()),
            vam_gt_700h=int((raw["TiempoVAM"] > 700).sum()),
            vam_lt_24h=int((raw["TiempoVAM"] < 24).sum()),
            preuti_gt_34h=int((pre_h > 34).sum()),
            apache_gt_36=int((raw["APACHE"] > 36).sum()),
        ),
        apache_p75=float(np.percentile(raw["APACHE"], 75)),
    )

    # ------------------------------------------------------------------ 2
    fits = {}
    for name, ser in [("EstadiaUCI", uti_h), ("TiempoVAM", raw["TiempoVAM"].astype(float)),
                      ("Est. PreUCI", pre_h), ("Est. PostUCI", post_h)]:
        f = DistributionFitter().fit(ser.to_numpy(dtype=float))
        pos = int((ser > 0).sum())
        fits[name] = dict(
            n_usado=pos, mejor=f.best_dist_, p_ks=float(f.best_pvalue_),
            params=[float(v) for v in f.all_results_[f.best_dist_]["params"]],
            todas={k: dict(p=float(v["p_value"]), ks=float(v["ks_stat"]))
                   for k, v in f.all_results_.items()},
        )
    res["ajuste_distribuciones"] = fits

    # ------------------------------------------------------------------ 3
    _ = adapter.get_conformal_inputs()          # puebla y = salida simulada (cal)
    train = adapter.get_train_data()
    cal, _cols = adapter.get_calibration_data()
    test = adapter.get_test_data()
    # población de y en entrenamiento (con semilla determinista)
    model = adapter._get_regression_model()
    X_cal = cal[ALL_FEATURES].to_numpy(float); y_cal = cal[TARGET_COL].to_numpy(float)
    X_te = test[ALL_FEATURES].to_numpy(float); y_te = test[TARGET_COL].to_numpy(float)
    res["partes"] = dict(n_train=len(train), n_cal=len(cal), n_test=len(test),
                         y_test_mean=float(y_te.mean()), y_test_sd=float(y_te.std(ddof=1)))
    yhat_te = model.predict(X_te)
    res["modelo_sustituto"] = dict(
        mae_test=float(np.mean(np.abs(y_te - yhat_te))),
        mae_cal=float(np.mean(np.abs(y_cal - model.predict(X_cal)))),
        y_cal_median=float(np.median(y_cal)),
    )

    def sim_with_seed():
        counter = {"i": 0}

        def f(**kw):
            if "_crn_seed" not in kw:
                counter["i"] += 1
                kw["_crn_seed"] = SEED * 1_000_003 + counter["i"]
            return adapter.simulate(**kw)
        return f

    cover = {}
    for conf in (0.90, 0.95):
        # conformal
        cp = ConformalPredictor(confidence=conf).fit(X_cal, y_cal, model)
        q = cp._threshold
        lo = np.maximum(0, yhat_te - q); hi = yhat_te + q
        k = int(((y_te >= lo) & (y_te <= hi)).sum())
        cover[f"conformal_{int(conf*100)}"] = dict(
            cobertura=k / len(y_te), k=k, n=len(y_te), wilson=wilson(k, len(y_te)),
            amplitud_media=float(np.mean(hi - lo)), q=float(q))
        # bootstrap y Monte Carlo (sobre simulate_fn)
        for method, tag in (("bootstrap", "bootstrap"), ("montecarlo", "montecarlo")):
            uq = UncertaintyQuantifier(method=method, confidence=conf, B=args.B, seed=SEED)
            if method == "bootstrap":
                uq.fit(pd.concat([train, cal])[ALL_FEATURES], NUMERIC_COLS)
            simf = sim_with_seed()
            los, his, preds = [], [], []
            for _, row in test.iterrows():
                p = {c: int(row[c]) for c in ALL_FEATURES}
                p.update(n_runs=200, percent=3)
                iv = uq.predict_interval(simf, p)
                los.append(iv.lower); his.append(iv.upper); preds.append(iv.prediction)
            los, his = np.array(los), np.array(his)
            k = int(((y_te >= los) & (y_te <= his)).sum())
            cover[f"{tag}_{int(conf*100)}"] = dict(
                cobertura=k / len(y_te), k=k, n=len(y_te), wilson=wilson(k, len(y_te)),
                amplitud_media=float(np.mean(his - los)),
                mae_pred_vs_y=float(np.mean(np.abs(np.array(preds) - y_te))))
        print("cobertura", int(conf * 100), {k: round(v["cobertura"], 3) for k, v in cover.items()
                                              if k.endswith(str(int(conf * 100)))})
    res["cobertura_test"] = cover

    # tiempos de respuesta (RNF-01)
    p0 = {c: int(test.iloc[0][c]) for c in ALL_FEATURES}; p0.update(n_runs=200, percent=3)
    t1 = time.time()
    uqb = UncertaintyQuantifier(method="bootstrap", confidence=0.95, B=args.B, seed=SEED)
    uqb.fit(pd.concat([train, cal])[ALL_FEATURES], NUMERIC_COLS)
    uqb.predict_interval(sim_with_seed(), p0)
    t_boot = time.time() - t1

    # --- particiones repetidas (conformal) -------------------------------
    full = pd.concat([train, cal, test], ignore_index=True)
    # y de todos los pacientes (misma regla: n_runs=50 y semilla por paciente)
    y_all = np.empty(len(full))
    for i, row in full.iterrows():
        p = {c: int(row[c]) for c in ALL_FEATURES}
        p.update(n_runs=50, percent=3, _crn_seed=SEED * 7 + i)
        y_all[i] = adapter.simulate(**p)
    X_all = full[ALL_FEATURES].to_numpy(float)
    apache_all = full["apache"].to_numpy(float)
    p75 = res["cohorte"]["apache_p75"]
    rep = {0.90: [], 0.95: []}
    strata = {0.90: dict(alto=[0, 0], bajo=[0, 0]), 0.95: dict(alto=[0, 0], bajo=[0, 0])}
    n = len(full)
    for r in range(args.reps):
        rng = np.random.default_rng(10_000 + r)
        idx = rng.permutation(n)
        n_cal_, n_te_ = int(n * 0.2), int(n * 0.2)
        ic, it, itr = idx[:n_cal_], idx[n_cal_:n_cal_ + n_te_], idx[n_cal_ + n_te_:]
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5,
                                   random_state=r, n_jobs=-1).fit(X_all[itr], y_all[itr])
        for conf in (0.90, 0.95):
            cpr = ConformalPredictor(confidence=conf).fit(X_all[ic], y_all[ic], rf)
            yh = rf.predict(X_all[it]); cov = (np.abs(y_all[it] - yh) <= cpr._threshold)
            rep[conf].append(float(cov.mean()))
            hi_mask = apache_all[it] >= p75
            strata[conf]["alto"][0] += int(cov[hi_mask].sum()); strata[conf]["alto"][1] += int(hi_mask.sum())
            strata[conf]["bajo"][0] += int(cov[~hi_mask].sum()); strata[conf]["bajo"][1] += int((~hi_mask).sum())
    res["conformal_particiones_repetidas"] = {
        str(int(c * 100)): dict(
            media=float(np.mean(v)), p2_5=float(np.percentile(v, 2.5)),
            p97_5=float(np.percentile(v, 97.5)), reps=args.reps,
            estrato_apache_ge_p75=dict(k=strata[c]["alto"][0], n=strata[c]["alto"][1],
                                       cobertura=strata[c]["alto"][0] / max(1, strata[c]["alto"][1])),
            estrato_apache_lt_p75=dict(k=strata[c]["bajo"][0], n=strata[c]["bajo"][1],
                                       cobertura=strata[c]["bajo"][0] / max(1, strata[c]["bajo"][1])))
        for c, v in rep.items()}
    print("repetidas", {k: round(v["media"], 3) for k, v in res["conformal_particiones_repetidas"].items()})

    # ------------------------------------------------------------------ 5
    ref = pd.concat([train, cal], ignore_index=True)   # referencia (sin prueba)
    ev = RobustnessEvaluator(simulate_fn=adapter.simulate, calibration_data=ref,
                             numeric_cols=NUMERIC_COLS, seed=SEED, n_jobs=3,
                             target_col=TARGET_COL)
    rows, noise_v, miss_v, shift_v, oat = [], [], [], [], []
    miss_per_var = {k: [] for k in ALL_FEATURES}
    t_eval = []
    reports = []
    for i, row in test.iterrows():
        p = {c: int(row[c]) for c in ALL_FEATURES}; p.update(n_runs=200, percent=3)
        t1 = time.time(); rep_ = ev.evaluate(p); t_eval.append(time.time() - t1)
        reports.append(rep_)
        noise_v.append([r.mean_variation for r in rep_.noise_results])
        miss_v.append([r.mean_variation for r in rep_.missing_results])
        sv = max(rep_.shift_result.relative_change_low, rep_.shift_result.relative_change_high)
        shift_v.append(sv)
        oat.append({s.variable: s.sensitivity_index for s in rep_.sensitivity_results})
        rows.append(dict(i=int(i), R=rep_.R, semaforo=rep_.semaphore, r_noise=rep_.r_noise,
                         r_missing=rep_.r_missing, r_shift=rep_.r_shift,
                         **{c: int(row[c]) for c in ALL_FEATURES},
                         base=rep_.base_prediction))
        # impacto por variable ausente (CRN con la misma semilla base)
        sp = dict(p); sp["n_runs"] = 1000; sp["_crn_seed"] = SEED + 99 + int(i)
        base_v = adapter.simulate(**sp)
        for k in ALL_FEATURES:
            q_ = dict(sp); q_[k] = ev._imputation.get(k, 0)
            v = adapter.simulate(**q_)
            miss_per_var[k].append(abs(v - base_v) / max(abs(base_v), 1e-9))
    df = pd.DataFrame(rows)
    noise_v, miss_v = np.array(noise_v), np.array(miss_v)
    from simuci_framework.framework.robustness_module import _component_score, _status_from_variation
    th_shift = float(ev.shift_threshold)
    nl = [0.10, 0.25, 0.50]; ml = [0.05, 0.10, 0.20, 0.30]
    tabla = []
    for j, s in enumerate(nl):
        m = float(noise_v[:, j].mean())
        tabla.append(dict(tipo="Ruido gaussiano", nivel=f"sigma={s:.2f}", variacion=m,
                          umbral=CLINICAL_THRESHOLD, estado=_status_from_variation(m),
                          componente=_component_score(m),
                          comp_media_pacientes=float(np.mean([_component_score(v) for v in noise_v[:, j]])),
                          sd=float(noise_v[:, j].std(ddof=1))))
    for j, s in enumerate(ml):
        m = float(miss_v[:, j].mean())
        tabla.append(dict(tipo="Datos faltantes", nivel=f"{int(s*100)}% ausentes", variacion=m,
                          umbral=CLINICAL_THRESHOLD, estado=_status_from_variation(m),
                          componente=_component_score(m),
                          comp_media_pacientes=float(np.mean([_component_score(v) for v in miss_v[:, j]])),
                          sd=float(miss_v[:, j].std(ddof=1))))
    ms = float(np.mean(shift_v))
    tabla.append(dict(tipo="Shift poblacional", nivel="p10/p90", variacion=ms,
                      umbral=th_shift, estado=("Crítico" if ms >= th_shift else "Alerta" if ms >= 0.4 * th_shift else "Estable"),
                      componente=_component_score(ms, threshold=th_shift),
                      comp_media_pacientes=float(np.mean([_component_score(v, th_shift) for v in shift_v])),
                      sd=float(np.std(shift_v, ddof=1))))
    res["stress_tests"] = dict(tabla=tabla, umbral_shift_calibrado=th_shift,
                               n_pacientes=len(df), n_referencia=len(ref))
    cnt = df["semaforo"].value_counts().to_dict()
    res["indice_R"] = dict(
        media=float(df.R.mean()), sd=float(df.R.std(ddof=1)), min=float(df.R.min()),
        max=float(df.R.max()), mediana=float(df.R.median()),
        verde=int(cnt.get("green", 0)), ambar=int(cnt.get("amber", 0)), rojo=int(cnt.get("red", 0)),
        r_noise=dict(media=float(df.r_noise.mean()), sd=float(df.r_noise.std(ddof=1))),
        r_missing=dict(media=float(df.r_missing.mean()), sd=float(df.r_missing.std(ddof=1))),
        r_shift=dict(media=float(df.r_shift.mean()), sd=float(df.r_shift.std(ddof=1))),
        R_valores=[float(v) for v in df.R])
    # Efecto del truncamiento de EstadiaUCI (> 200 h) por el límite de la aplicación
    g = df.assign(uti_trunc=df.uti_stay > 200)
    res["R_por_truncamiento_uti"] = {
        ("uti_stay_gt_200h" if k else "uti_stay_le_200h"): dict(
            n=int(len(v)), R_media=float(v.R.mean()), R_min=float(v.R.min()),
            R_max=float(v.R.max()), verde=int((v.semaforo == "green").sum()),
            ambar=int((v.semaforo == "amber").sum()), rojo=int((v.semaforo == "red").sum()))
        for k, v in g.groupby("uti_trunc")}
    res["R_spearman"] = {c: float(stats.spearmanr(df.R, df[c])[0])
                         for c in ("uti_stay", "vam_time", "apache", "age")}
    oat_df = pd.DataFrame(oat)
    res["oat"] = {k: float(v) for k, v in oat_df.mean().sort_values(ascending=False).items()}
    res["faltante_por_variable"] = {k: dict(media=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
                                    for k, v in sorted(miss_per_var.items(),
                                                       key=lambda kv: -np.mean(kv[1]))}
    # sensibilidad a pesos
    alts = {"propuestos (0.4/0.4/0.2)": (0.4, 0.4, 0.2), "iguales (1/3 c/u)": (1/3, 1/3, 1/3),
            "ruido+faltantes (0.5/0.5/0)": (0.5, 0.5, 0.0), "shift alto (0.25/0.25/0.5)": (0.25, 0.25, 0.5)}
    base_cls = df.semaforo.tolist()
    wres = {}
    for nm, (a, b, c) in alts.items():
        Rn = a * df.r_noise + b * df.r_missing + c * df.r_shift
        cls = ["green" if v >= 0.7 else "amber" if v >= 0.4 else "red" for v in Rn]
        wres[nm] = dict(R_media=float(Rn.mean()), verde=cls.count("green"), ambar=cls.count("amber"),
                        rojo=cls.count("red"),
                        concordancia=float(np.mean([x == y for x, y in zip(cls, base_cls)])))
    res["sensibilidad_pesos"] = wres
    res["tiempos"] = dict(evaluate_media_s=float(np.mean(t_eval)), evaluate_max_s=float(np.max(t_eval)),
                          bootstrap_B_s=float(t_boot), B=args.B)

    # ------------------------------------------------------------------ 6
    casos = {}
    cp95 = ConformalPredictor(confidence=0.95).fit(X_cal, y_cal, model)
    # Los casos se eligen entre pacientes cuyos valores caen DENTRO de los
    # límites de entrada de la aplicación (para poder introducirlos tal cual).
    dentro = df[(df.uti_stay <= 200) & (df.preuti_stay <= 34) &
                (df.vam_time.between(24, 700)) & (df.apache <= 36)]
    res["casos_candidatos_dentro_de_limites"] = int(len(dentro))
    for nombre, idx_ in (("caso1_mayor_R", int(dentro.R.idxmax())), ("caso2_menor_R", int(dentro.R.idxmin()))):
        r_ = reports[idx_]; row = test.iloc[idx_]
        x = row[ALL_FEATURES].to_numpy(float).reshape(1, -1)
        iv = cp95.predict_interval(x)
        casos[nombre] = dict(
            idx=idx_, perfil={c: int(row[c]) for c in ALL_FEATURES},
            media_simulada=float(r_.base_prediction), pred_sustituto=float(iv.prediction),
            ic_lo=float(iv.lower), ic_hi=float(iv.upper), R=float(r_.R), semaforo=r_.semaphore,
            r_noise=float(r_.r_noise), r_missing=float(r_.r_missing), r_shift=float(r_.r_shift),
            ruido=[float(v.mean_variation) for v in r_.noise_results],
            faltantes=[float(v.mean_variation) for v in r_.missing_results],
            shift=float(max(r_.shift_result.relative_change_low, r_.shift_result.relative_change_high)),
            oat={s.variable: float(s.sensitivity_index) for s in r_.sensitivity_results},
            y_ref=float(y_te[idx_]))
    res["casos"] = casos

    # histograma bootstrap de un paciente típico (Figura 2.3)
    ok = ((test.uti_stay <= 200) & (test.preuti_stay <= 34) & test.vam_time.between(24, 700)).to_numpy()
    dist_ = np.abs(test["apache"].to_numpy() - np.median(test["apache"])) + (~ok) * 1e6
    typ = int(np.argmin(dist_))
    pt = {c: int(test.iloc[typ][c]) for c in ALL_FEATURES}; pt.update(n_runs=200, percent=3)
    pb = UncertaintyQuantifier(method="bootstrap", confidence=0.95, B=args.B, seed=SEED)
    pb.fit(pd.concat([train, cal])[ALL_FEATURES], NUMERIC_COLS)
    ivb = pb.predict_interval(sim_with_seed(), pt)
    preds = pb.last_predictions_
    res["figura_2_3"] = dict(paciente=pt, preds=[float(v) for v in preds],
                             lo=float(ivb.lower), hi=float(ivb.upper), pred=float(ivb.prediction))

    res["duracion_total_s"] = time.time() - t0
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "resultados_experimentos.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "indice_R_por_paciente.csv", index=False)
    print("OK ->", OUT_DIR, f"({res['duracion_total_s']:.0f} s)")


if __name__ == "__main__":
    main()
