"""
UC4 SolarDrâa - Group 9 (Achraf El Makaoui & Anas Rwchi) - Streamlit demo.

Loads the REAL saved models from outputs/models/ and the REAL data from data_UC4/,
and recomputes everything live (no retraining, no hard-coded results).

Run from the project folder (the one that contains data_UC4/ and outputs/):
    pip install streamlit
    streamlit run demo_app.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# 0. Paths (same logic as the starter notebook: first folder that contains data_UC4/)
# =============================================================================
def trouver_racine():
    candidats = []
    if os.environ.get("UC4_RACINE"):
        candidats.append(Path(os.environ["UC4_RACINE"]))
    ici = Path(__file__).resolve().parent
    candidats += [Path.cwd(), *Path.cwd().parents, ici, *ici.parents,
                  Path("/content/drive/MyDrive/UC4_Solaire"),
                  Path("/mnt/c/Users/NOVA/Desktop/UC4_Solaire")]
    for c in candidats:
        if (c / "data_UC4").is_dir():
            return c
    return None

RACINE = trouver_racine()

# =============================================================================
# 1. Constants and data preparation - copied from uc4_demarrage_reconstruit.ipynb
#    (same functions, same hypotheses H1–H10, so the models get exactly their inputs)
# =============================================================================
H = 6
CAPACITE = {"A": 50, "B": 20}
SEUIL_JOUR_DEG = 5.0                                   # H1
DECOUPAGE = {
    "A": {"tr": (None, "2024-10-01"), "va": ("2024-10-01", "2025-01-01"), "te": ("2025-01-01", None)},
    "B": {"tr": (None, "2025-03-01"), "va": ("2025-03-01", "2025-04-01"), "te": ("2025-04-01", None)},
}
FICHIERS = {
    "mlp_ref": "mlp_reference_A.keras",
    "j1": "jalon1_mlp_retenu.keras",
    "j2_mlp": "jalon2_mlp_prevision.keras",
    "j2_cnn": "jalon2_cnn_ciel.keras",
    "j2_ae": "jalon2_ae_thermo.keras",
    "j3_mlp_B": "jalon3_mlp_site_B.keras",
    "j3_cnn_mt": "jalon3_cnn_multitache.keras",
    "j4_gru": "jalon4_rnn_site_A.keras",
}


def ciel_clair(hauteur_deg):
    s = np.sin(np.deg2rad(np.asarray(hauteur_deg, dtype="float64")))
    jour = s > 0
    s_sur = np.where(jour, s, 1.0)
    return np.where(jour, 1098.0 * s_sur * np.exp(-0.057 / s_sur), 0.0)


def decoupe(timestamps, site):
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    masques = []
    for ens in ("tr", "va", "te"):
        debut, fin = DECOUPAGE[site][ens]
        m = np.ones(len(t), dtype=bool)
        if debut is not None:
            m &= np.asarray(t >= pd.Timestamp(debut))
        if fin is not None:
            m &= np.asarray(t < pd.Timestamp(fin))
        masques.append(m)
    return tuple(masques)


def fabriquer_exemples(df, site):
    cap = CAPACITE[site]
    n = len(df)
    p = df["production"].to_numpy(dtype="float64") / cap
    hs = df["hauteur_soleil"].to_numpy(dtype="float64")
    cs = ciel_clair(hs)
    neb = df["nebulosite_prevue"].to_numpy(dtype="float64")
    tp = df["temperature_prevue"].to_numpy(dtype="float64")
    nett = df["jours_depuis_nettoyage"].to_numpy(dtype="float64")
    t = df["horodatage"]
    idx = np.arange(24, n - H)
    futur = lambda v: np.stack([v[idx + h] for h in range(1, H + 1)], axis=1)
    hist = np.stack([p[idx - 23 + k] for k in range(24)], axis=1)                       # H2
    heure = t.dt.hour.to_numpy()[idx]
    jour_an = t.dt.dayofyear.to_numpy()[idx]
    cal = np.column_stack([np.sin(2 * np.pi * heure / 24), np.cos(2 * np.pi * heure / 24),
                           np.sin(2 * np.pi * (jour_an - 1) / 365.25),
                           np.cos(2 * np.pi * (jour_an - 1) / 365.25)])                   # H4
    X = np.column_stack([hist, futur(cs) / 1000, futur(neb), futur(tp) / 40,
                         cal, nett[idx] / 30]).astype("float32")                          # H3
    Y = futur(p).astype("float32")
    J = futur(hs) > SEUIL_JOUR_DEG
    T = t.to_numpy()[idx]
    tr, va, te = decoupe(T, site)                                                        # H5
    k = np.where(hs > SEUIL_JOUR_DEG, p / np.where(cs > 0, cs, 1.0), np.nan)              # H7
    k = pd.Series(k).ffill().fillna(0.0).to_numpy()
    intelligente = np.clip(k[idx][:, None] * futur(cs), 0, 1).astype("float32")
    return {"X": X, "Y": Y, "J": J, "T": T, "tr": tr, "va": va, "te": te,
            "SP": intelligente, "origine": idx}


def fabriquer_sequences(df, site, L=6):
    """Official Jalon 4 starter code (inputs of the GRU)."""
    from numpy.lib.stride_tricks import sliding_window_view
    p = df.production.values / CAPACITE[site]
    cs = ciel_clair(df.hauteur_soleil.values) / 1000
    neb, tp = df.nebulosite_prevue.values, df.temperature_prevue.values / 40
    h = df.horodatage.dt.hour.values
    sh, ch = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    passe = np.column_stack([p, cs, neb, tp, sh, ch]).astype("float32")
    connu = np.column_stack([cs, neb, tp, sh, ch]).astype("float32")
    idx = np.arange(24, len(df) - H)
    passe = np.vstack([np.zeros((L, 6), "float32"), passe])
    hist = sliding_window_view(passe, L, axis=0)[idx + 1]
    futur = np.stack([connu[idx + k] for k in range(1, H + 1)], axis=1)
    return np.ascontiguousarray(hist.transpose(0, 2, 1)), futur


def rmse_par_horizon(Y, P, J, capacite=1.0):
    Y, P, J = np.asarray(Y, "float64"), np.asarray(P, "float64"), np.asarray(J, bool)
    err2 = np.where(J, (Y - P) ** 2, 0.0)
    return np.sqrt(err2.sum(axis=0) / np.maximum(J.sum(axis=0), 1)) * capacite


def auc(score_sain, score_defaut):
    """AUC = probability that a defective module scores higher than a healthy one (Mann–Whitney)."""
    s = np.concatenate([score_sain, score_defaut])
    rangs = pd.Series(s).rank().to_numpy()
    n0, n1 = len(score_sain), len(score_defaut)
    return float((rangs[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# =============================================================================
# 2. Loading (cached: computed once per session)
# =============================================================================
@st.cache_resource(show_spinner="Loading the data…")
def charger_donnees(racine):
    data = Path(racine) / "data_UC4"
    SERIES, EX = {}, {}
    for site in CAPACITE:
        df = pd.read_csv(data / f"uc4_site{site}.csv")
        df["horodatage"] = pd.to_datetime(df["horodatage"], format="%Y-%m-%d %H:%M")
        SERIES[site] = df
        EX[site] = fabriquer_exemples(df, site)

    def npz(chemin):
        with np.load(chemin, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    CIEL = {site: npz(data / f"uc4_ciel_{site}.npz") for site in CAPACITE}
    THERMO = npz(data / "uc4_thermographie.npz")

    # ---- Sun position in the fisheye image (H10), fitted on site A training images, as in the starter ----
    def geom(site, lignes):
        d = SERIES[site]
        z = 90.0 - d.hauteur_soleil.to_numpy()[lignes]
        az = np.deg2rad(d.azimut.to_numpy()[lignes])
        return z * np.sin(az), -z * np.cos(az)
    CA = CIEL["A"]
    tr_img_A = decoupe(SERIES["A"].horodatage.values[CA["heure"]], "A")[0]
    clair = tr_img_A & (CA["occultation"] < 0.02) & (CA["couverture"] < 0.05)
    lig_max, col_max = np.divmod(CA["X"][clair].reshape(clair.sum(), -1).argmax(1), 32)
    u, v = geom("A", CA["heure"][clair])
    kc, c0 = np.polyfit(u, col_max, 1)
    kr, r0 = np.polyfit(v, lig_max, 1)

    def position_soleil(site, lignes):
        u, v = geom(site, np.asarray(lignes))
        col = np.clip(np.rint(c0 + kc * u), 0, 31).astype(int)
        row = np.clip(np.rint(r0 + kr * v), 0, 31).astype(int)
        return row, col

    disque = CA["X"][tr_img_A].max(axis=0) > 0
    seuils = np.arange(100, 241, 5)
    mae = [np.abs((CA["X"][tr_img_A][:, disque] > s).mean(1) - CA["couverture"][tr_img_A]).mean() for s in seuils]
    seuil_couv = int(seuils[int(np.argmin(mae))])

    def lum_soleil(site, images, lignes):
        row, col = position_soleil(site, lignes)
        pad = np.pad(images, ((0, 0), (1, 1), (1, 1)))
        vois = np.stack([pad[np.arange(len(images)), row + 1 + dr, col + 1 + dc]
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)
        return vois.max(axis=1).astype("float64")

    b_tr = lum_soleil("A", CA["X"][tr_img_A], CA["heure"][tr_img_A])
    b_occ = float(np.median(b_tr[CA["occultation"][tr_img_A] > 0.98]))

    regles = {
        "couverture": lambda images: (images[:, disque] > seuil_couv).mean(axis=1),
        "occultation": lambda site, images, lignes: np.clip((255.0 - lum_soleil(site, images, lignes)) / (255.0 - b_occ), 0, 1),
        "thermo": lambda images: (lambda x: x.max(1) - np.median(x, 1))(images.reshape(len(images), -1).astype("float32")),
    }
    return SERIES, EX, CIEL, THERMO, position_soleil, regles


@st.cache_resource(show_spinner="Loading the saved models…")
def charger_modeles(racine):
    from tensorflow import keras
    dossier = Path(racine) / "outputs" / "models"
    modeles, manquants = {}, []
    for cle, nom in FICHIERS.items():
        chemin = dossier / nom
        if chemin.exists():
            modeles[cle] = keras.models.load_model(chemin, compile=False)
        else:
            manquants.append(nom)
    return modeles, manquants


def carte_soleil(position_soleil, site, lignes, sigma=1.5):
    row, col = position_soleil(site, np.asarray(lignes))
    yy, xx = np.mgrid[:32, :32]
    d2 = (yy[None] - row[:, None, None]) ** 2 + (xx[None] - col[:, None, None]) ** 2
    return np.exp(-d2 / (2 * sigma ** 2)).astype("float32")[..., None]


def entree_cnn(modele, CIEL, position_soleil, site):
    """Images /255 with a channel axis (+ sun map if the saved CNN has 2 channels)."""
    C = CIEL[site]
    X = C["X"][..., None].astype("float32") / 255.0
    if modele.input_shape[-1] == 2:
        X = np.concatenate([X, carte_soleil(position_soleil, site, C["heure"])], axis=-1)
    return X


@st.cache_resource(show_spinner="Running the models on the data…")
def calculer_predictions(racine):
    """All predictions used by the app, computed once with the saved models."""
    SERIES, EX, CIEL, THERMO, position_soleil, regles = charger_donnees(racine)
    M, _ = charger_modeles(racine)
    out = {"A": {}, "B": {}, "ciel": {}, "thermo": {}}

    # ---- Sky CNNs on every image of both sites ----
    for site in CAPACITE:
        C = CIEL[site]
        d = {"mesure": np.column_stack([C["couverture"], C["occultation"]]),
             "regle": np.column_stack([regles["couverture"](C["X"]), regles["occultation"](site, C["X"], C["heure"])])}
        if "j2_cnn" in M:
            d["cnn_j2"] = M["j2_cnn"].predict(entree_cnn(M["j2_cnn"], CIEL, position_soleil, site), verbose=0, batch_size=512)
        if "j3_cnn_mt" in M:
            p = M["j3_cnn_mt"].predict(C["X"][..., None].astype("float32") / 255.0, verbose=0, batch_size=512)
            d["cnn_j3"] = np.column_stack([np.asarray(p["couverture"])[:, 0], np.asarray(p["occultation"])[:, 0]])
        d["split"] = decoupe(SERIES[site].horodatage.values[C["heure"]], site)
        out["ciel"][site] = d

    # ---- Forecasts (all origins; the app only shows the test period) ----
    for site in CAPACITE:
        E = EX[site]
        P = {"Smart persistence": E["SP"]}
        if "mlp_ref" in M:
            P["MLP reference (starter, trained on A)"] = M["mlp_ref"].predict(E["X"], verbose=0, batch_size=2048)
        if site == "A":
            if "j1" in M:
                P["MLP Jalon 1 (retained)"] = M["j1"].predict(E["X"], verbose=0, batch_size=2048)
            if "j2_mlp" in M:
                X = E["X"]
                if M["j2_mlp"].input_shape[-1] == 53 and "cnn_j2" in out["ciel"]["A"]:
                    C = CIEL["A"]
                    l2i = np.full(len(SERIES["A"]), -1)
                    l2i[C["heure"]] = np.arange(len(C["heure"]))
                    orig = E["origine"]
                    F = np.zeros((len(orig), 6), dtype="float32")
                    for k, idx in enumerate([l2i[orig], l2i[orig - 1]]):
                        ok = idx >= 0
                        F[ok, 2 * k:2 * k + 2] = out["ciel"]["A"]["cnn_j2"][idx[ok]]
                        F[:, 4 + k] = ok
                    X = np.hstack([X, F])
                P["MLP + sky CNN (Jalon 2)"] = M["j2_mlp"].predict(X, verbose=0, batch_size=2048)
        else:
            if "j3_mlp_B" in M:
                P["MLP fine-tuned on B winter (Jalon 3)"] = M["j3_mlp_B"].predict(E["X"], verbose=0, batch_size=2048)
        if "j4_gru" in M:
            L = M["j4_gru"].input_shape[0][1]
            Hs, Fs = fabriquer_sequences(SERIES[site], site, L)
            nom = "GRU (Jalon 4)" if site == "A" else "GRU trained on A, as is (Jalon 4)"
            P[nom] = M["j4_gru"].predict([Hs, Fs], verbose=0, batch_size=2048)
        out[site] = {n: np.asarray(p, dtype="float32") for n, p in P.items()}   # raw outputs, as in the notebooks

    # ---- Thermography auto-encoder ----
    if "j2_ae" in M:
        TH = {s: THERMO["X_" + s][..., None].astype("float32") for s in ["train", "val", "test"]}
        mu, sd = TH["train"].mean(), TH["train"].std()
        res = {"mu": mu, "sd": sd}
        for s in ("val", "test"):
            Xn = (TH[s] - mu) / sd
            R = M["j2_ae"].predict(Xn, verbose=0, batch_size=256)
            e2 = (Xn - R) ** 2
            res[s] = {"recon": R[..., 0] * sd + mu, "e2": e2[..., 0],
                      "max": e2.max(axis=(1, 2, 3)), "mean": e2.mean(axis=(1, 2, 3)),
                      "regle": regles["thermo"](THERMO["X_" + s])}
        res["seuil"] = {k: float(np.percentile(res["val"][k], 95)) for k in ("max", "mean", "regle")}
        out["thermo"] = res
    return out


def gain_par_horizon(E, P, masque, cap):
    Y, J = E["Y"][masque], E["J"][masque]
    return 1 - rmse_par_horizon(Y, P[masque], J, cap) / rmse_par_horizon(Y, E["SP"][masque], J, cap)


# =============================================================================
# 3. Page layout
# =============================================================================
st.set_page_config(page_title="UC4 SolarDrâa - Group 9", page_icon="☀️", layout="wide")
st.title("☀️ UC4 SolarDrâa - forecasting demo")
st.caption("Group 9 · Achraf El Makaoui & Anas Rwchi · M2 SDIA, ENSET Mohammedia · all models are the saved "
           "models from our notebooks, run live on the test data (simulated data).")

if RACINE is None:
    st.error("Folder `data_UC4/` not found. Run the app from the project folder, or set the environment "
             "variable `UC4_RACINE` to the project folder.")
    st.stop()

SERIES, EX, CIEL, THERMO, position_soleil, regles = charger_donnees(str(RACINE))
MODELES, MANQUANTS = charger_modeles(str(RACINE))
if MANQUANTS:
    st.warning("Missing model files (the related parts are hidden): " + ", ".join(MANQUANTS))
PRED = calculer_predictions(str(RACINE))

page = st.sidebar.radio("Page", ["1 · Forecast explorer", "2 · Sky camera", "3 · Panel inspection",
                                 "4 · Test results (all models)"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Metric** - gain = 1 − RMSE(model) / RMSE(smart persistence), daylight hours only.\n\n"
                    "0 = as good as the reference · 0.30 = 30 % less error · < 0 = worse.")
ECRETER = st.sidebar.checkbox("Clip forecasts to [0, capacity] (deployment)", value=False,
                              help="Off = raw model outputs, exactly as evaluated in the notebooks. "
                                   "On = negative values and values above capacity are cut, as when the model is used.")
PRED = {**PRED, **{_s: {n: (np.clip(p, 0, 1) if ECRETER else p) for n, p in PRED[_s].items()}
                   for _s in ("A", "B")}}            # new dict: the cached predictions stay untouched
st.sidebar.caption(f"Project folder: `{RACINE}`")

COULEURS = {"Smart persistence": "#5b8db8"}


# -----------------------------------------------------------------------------
# Page 1 - Forecast explorer
# -----------------------------------------------------------------------------
if page.startswith("1"):
    st.header("Forecast explorer - test period")
    c1, c2, c3 = st.columns([1, 1, 1])
    site = c1.radio("Site", ["A", "B"], horizontal=True,
                    format_func=lambda s: "A · Ouarzazate 50 MW" if s == "A" else "B · Atlantic coast 20 MW")
    E, cap = EX[site], CAPACITE[site]
    T = pd.DatetimeIndex(E["T"])
    te = E["te"]
    jours = sorted(set(T[te].date))
    defaut = pd.Timestamp("2025-07-08").date() if site == "A" else pd.Timestamp("2025-05-15").date()
    jour = c2.date_input("Day", value=defaut if defaut in jours else jours[len(jours) // 2],
                         min_value=jours[0], max_value=jours[-1])
    h = c3.slider("Horizon (hours ahead)", 1, 6, 1)
    noms = list(PRED[site].keys())
    choix = st.multiselect("Models", noms, default=noms)

    # ---- Chart 1: one day, horizon h ----
    t_cible = T + pd.Timedelta(hours=h)
    m_jour = te & (np.asarray(t_cible.date) == jour)
    if m_jour.sum() == 0:
        st.info("No test forecast for this day.")
    else:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(t_cible[m_jour], E["Y"][m_jour, h - 1] * cap, color="black", lw=2.2, label="observed")
        for n in choix:
            style = ":" if n == "Smart persistence" else "-"
            ax.plot(t_cible[m_jour], PRED[site][n][m_jour, h - 1] * cap, style, lw=1.6, label=n)
        ax.axhline(cap, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("MW"); ax.set_title(f"Site {site} · {jour} · forecasts made {h} h ahead")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

        J = E["J"][m_jour, h - 1]
        if J.sum() > 0:
            y = E["Y"][m_jour, h - 1][J]
            r_sp = np.sqrt(np.mean((y - E["SP"][m_jour, h - 1][J]) ** 2)) * cap
            lignes = []
            for n in choix:
                r = np.sqrt(np.mean((y - PRED[site][n][m_jour, h - 1][J]) ** 2)) * cap
                lignes.append({"model": n, "RMSE this day (MW)": round(r, 2),
                               "gain this day": round(1 - r / r_sp, 3) if r_sp > 0 else np.nan})
            st.caption("Daylight hours of this day only (one day is noisy - the test-set numbers are on page 4).")
            st.dataframe(pd.DataFrame(lignes).set_index("model"))

    # ---- Chart 2: one forecast origin, the 6 horizons ----
    st.subheader("One forecast, made at a chosen hour")
    heures = sorted(set(T[te & (np.asarray(T.date) == jour)].hour))
    if heures:
        hh = st.select_slider("Forecast made at", options=heures, value=9 if 9 in heures else heures[0],
                              format_func=lambda x: f"{x:02d}:00")
        i = int(np.where(te & (np.asarray(T.date) == jour) & (T.hour == hh))[0][0])
        passe = E["X"][i, :24] * cap
        t_passe = [T[i] - pd.Timedelta(hours=23 - k) for k in range(24)]
        t_fut = [T[i] + pd.Timedelta(hours=k) for k in range(1, H + 1)]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.plot(t_passe, passe, color="black", lw=2, label="past 24 h (input)")
        ax.plot(t_fut, E["Y"][i] * cap, color="black", lw=2, ls="--", marker="o", label="observed next 6 h")
        for n in choix:
            ax.plot(t_fut, PRED[site][n][i] * cap, marker=".", lw=1.5, ls=":" if n == "Smart persistence" else "-", label=n)
        ax.axvline(T[i], color="#E39B2D", lw=2); ax.text(T[i], cap * 1.02, " t", color="#E39B2D")
        ax.set_ylabel("MW"); ax.set_ylim(-1, cap * 1.12); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)


# -----------------------------------------------------------------------------
# Page 2 - Sky camera
# -----------------------------------------------------------------------------
elif page.startswith("2"):
    st.header("Sky camera - cloud cover and sun occlusion")
    site = st.radio("Site", ["A", "B"], horizontal=True)
    C, Pc = CIEL[site], PRED["ciel"][site]
    te_img = Pc["split"][2]
    t_img = pd.DatetimeIndex(SERIES[site].horodatage.values[C["heure"]])
    idx_te = np.where(te_img)[0]

    if "img" not in st.session_state or st.session_state.get("img_site") != site:
        st.session_state.img = int(idx_te[len(idx_te) // 2]); st.session_state.img_site = site
    b1, b2, b3 = st.columns(3)
    if b1.button("Random image"):
        st.session_state.img = int(np.random.choice(idx_te))
    if b2.button("Random PARTLY hidden sun (0.2–0.8)"):
        cand = idx_te[(C["occultation"][idx_te] > 0.2) & (C["occultation"][idx_te] < 0.8)]
        if len(cand):
            st.session_state.img = int(np.random.choice(cand))
    if b3.button("Random clear sky"):
        cand = idx_te[C["occultation"][idx_te] < 0.02]
        if len(cand):
            st.session_state.img = int(np.random.choice(cand))
    j = st.session_state.img

    g, d = st.columns([1, 1.3])
    with g:
        row, col = position_soleil(site, [C["heure"][j]])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.imshow(C["X"][j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.plot(col, row, "o", mfc="none", mec="#E39B2D", ms=14, mew=2)
        ax.set_title(f"{t_img[j]:%Y-%m-%d %H:%M}  (orange = computed sun position)", fontsize=9)
        ax.axis("off")
        st.pyplot(fig); plt.close(fig)
    with d:
        sources = {"measured": "mesure", "simple rules": "regle",
                   "sky CNN (Jalon 2)": "cnn_j2", "multitask CNN (Jalon 3)": "cnn_j3"}
        sources = {nom: k for nom, k in sources.items() if k in Pc}
        df = pd.DataFrame({nom: Pc[k][j] for nom, k in sources.items()},
                          index=["cloud cover", "sun occlusion"]).T.round(3)
        st.dataframe(df)
        st.markdown("**Cloud cover** = share of the sky covered by clouds. **Occlusion** = how much the sun "
                    "itself is hidden (0 = visible, 1 = hidden).")
        if site == "B":
            st.info("Both CNNs here were trained on site A images. The version fine-tuned on site B images "
                    "(Jalon 3) was not saved, so morning fog can be over-estimated as cloud.")
        # error on the whole test set of this site
        te_m = te_img
        lignes = {nom: np.abs(Pc[k][te_m] - Pc["mesure"][te_m]).mean(0) for nom, k in sources.items() if k != "mesure"}
        st.caption(f"Mean absolute error on all {te_m.sum()} test images of site {site}:")
        st.dataframe(pd.DataFrame(lignes, index=["MAE cover", "MAE occlusion"]).T.round(4))

    # same day strip
    jour = t_img[j].date()
    du_jour = idx_te[np.asarray(t_img[idx_te].date) == jour]
    if len(du_jour) > 1:
        st.subheader(f"All images of {jour}")
        n = len(du_jour)
        fig, axs = plt.subplots(1, n, figsize=(1.35 * n, 1.9))
        for a, k in zip(np.atleast_1d(axs), du_jour):
            a.imshow(C["X"][k], cmap="gray", vmin=0, vmax=255)
            a.set_title(f"{t_img[k]:%H}h\nocc {C['occultation'][k]:.2f}", fontsize=7); a.axis("off")
        st.pyplot(fig); plt.close(fig)


# -----------------------------------------------------------------------------
# Page 3 - Panel inspection
# -----------------------------------------------------------------------------
elif page.startswith("3"):
    st.header("Panel inspection - thermal auto-encoder")
    if not PRED["thermo"]:
        st.error("Auto-encoder model missing.")
        st.stop()
    R = PRED["thermo"]
    lab = THERMO["defaut_test"]
    NOMS_T = {0: "healthy", 1: "type 1", 2: "type 2 (cracked cells - our hypothesis)", 3: "type 3", 4: "type 4"}
    c1, c2 = st.columns([1, 1])
    filtre = c1.selectbox("Show modules", ["any", 0, 1, 2, 3, 4], index=3,
                          format_func=lambda k: "any" if k == "any" else NOMS_T[k])
    score = c2.radio("Anomaly score", ["max", "mean"], horizontal=True,
                     help="Retained: max (keeps a small hot cell; the mean dilutes it over 960 pixels).")
    cand = np.arange(len(lab)) if filtre == "any" else np.where(lab == filtre)[0]
    if "mod" not in st.session_state or st.session_state.mod not in cand:
        st.session_state.mod = int(cand[0])
    if st.button("Random module"):
        st.session_state.mod = int(np.random.choice(cand))
    i = st.session_state.mod

    s, seuil = R["test"][score][i], R["seuil"][score]
    alarme = s > seuil
    st.markdown(f"Module **#{i}** · true label: **{NOMS_T[int(lab[i])]}** · score ({score}) = {s:.3f} · "
                f"threshold (95th percentile of healthy validation) = {seuil:.3f} → "
                + ("🔴 **ALARM**" if alarme else "🟢 **healthy**"))
    fig, axs = plt.subplots(1, 3, figsize=(12, 2.9))
    img = THERMO["X_test"][i].astype("float32")
    vmin, vmax = np.percentile(img, 1), img.max()
    for a, (x, t, cm, kw) in zip(axs, [(img, "thermal image (°C)", "inferno", dict(vmin=vmin, vmax=vmax)),
                                        (R["test"]["recon"][i], "reconstruction (°C)", "inferno", dict(vmin=vmin, vmax=vmax)),
                                        (R["test"]["e2"][i], "squared error", "magma", {})]):
        im = a.imshow(x, cmap=cm, **kw); a.set_title(t, fontsize=10); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.03)
    st.pyplot(fig); plt.close(fig)
    st.caption("The auto-encoder was trained on healthy modules only: the reconstruction looks healthy, "
               "so a defect stays in the error map.")

    with st.expander("Whole test set: detection per defect type at 5 % false alarms"):
        lignes = {}
        for nom, k in [("AE, max error", "max"), ("AE, mean error", "mean"), ("Rule: max − median", "regle")]:
            sc, th = R["test"][k], R["seuil"][k]
            lignes[nom] = {"false alarms (healthy)": (sc[lab == 0] > th).mean(),
                           **{f"type {t}": (sc[lab == t] > th).mean() for t in (1, 2, 3, 4)},
                           "AUC": auc(sc[lab == 0], sc[lab > 0])}
        st.dataframe(pd.DataFrame(lignes).T.round(3))
        st.caption("Targets: AUC ≥ 0.98, type 2 (cracked cells, our hypothesis) ≥ 0.85.")


# -----------------------------------------------------------------------------
# Page 4 - Test results
# -----------------------------------------------------------------------------
else:
    st.header("Test results - recomputed live from the saved models")
    for site, titre, cible in [("A", "Site A · test 2025", 0.30), ("B", "Site B · test April–June 2025", 0.27)]:
        E, cap = EX[site], CAPACITE[site]
        lignes = {n: gain_par_horizon(E, P, E["te"], cap) for n, P in PRED[site].items()}
        df = pd.DataFrame(lignes, index=[f"t+{k}" for k in range(1, H + 1)]).T
        df["mean"] = df.mean(axis=1)
        st.subheader(titre)
        st.dataframe(df.round(3).style.background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.5,
                                                            subset=list(df.columns)))
        st.caption(f"Targets: gain at 1 h ≥ 0.12 (site A), mean gain ≥ {cible:.2f}. "
                   + ("Forecasts clipped to [0, capacity] (deployment)." if ECRETER
                      else "Raw model outputs - same numbers as the notebooks."))
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for n in df.index:
            if n != "Smart persistence":
                ax.plot(range(1, H + 1), df.loc[n, [f"t+{k}" for k in range(1, H + 1)]], marker="o", label=n)
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("horizon (h)"); ax.set_ylabel("gain")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Sky state (test images)")
    lignes = {}
    for site in CAPACITE:
        Pc = PRED["ciel"][site]; te = Pc["split"][2]
        for k, nom in [("regle", "simple rules"), ("cnn_j2", "sky CNN (Jalon 2)"), ("cnn_j3", "multitask CNN (Jalon 3)")]:
            if k in Pc:
                e = np.abs(Pc[k][te] - Pc["mesure"][te]).mean(0)
                lignes[f"site {site} · {nom}"] = {"MAE cover": e[0], "MAE occlusion": e[1]}
    st.dataframe(pd.DataFrame(lignes).T.round(4))
    st.caption("Target: occlusion MAE ≤ 0.10. Site B here uses the CNNs trained on site A (the B-fine-tuned CNN was not saved).")

    st.subheader("Model sizes")
    dossier = RACINE / "outputs" / "models"
    st.dataframe(pd.DataFrame({"file": [f for f in FICHIERS.values() if (dossier / f).exists()],
                               "size (MB)": [round((dossier / f).stat().st_size / 1e6, 3)
                                             for f in FICHIERS.values() if (dossier / f).exists()]}), hide_index=True)
    st.caption("Target: ≤ 5 MB per model.")
