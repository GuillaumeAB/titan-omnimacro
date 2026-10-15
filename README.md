# 🌐 Titan OmniMacro v2.0

Institutional Multi-Asset Quantitative Trajectory Forecasting, Term Structure Modeling, Markov Regime Switching & Portfolio Rebalancing Engine.

---

## 🏛️ Architecture & Méthodologie

Titan OmniMacro est un moteur quantitatif unifié conçu pour projeter des distributions conjointes de rendements multi-actifs (actions US/Europe/Asie/EM, taux souverains, matières premières, devises, crypto) sans artifices ni biais d'anticipation (*look-ahead bias*).

```
titan/
├── data/          # Ingestion avec cache disque TTL : Yahoo Finance, FRED, Polymarket Gamma API
├── models/        # Modélisation mathématique pure (numpy/scipy) :
│   ├── covariance # Rétrécissement Ledoit-Wolf (2004) et projection PSD
│   ├── regime     # HMM gaussien 3-états en log-espace (Baum-Welch, Viterbi)
│   ├── rates      # Modèle de courbe de taux Dynamic Nelson-Siegel (Diebold-Li 2006)
│   └── factors    # Modèle factoriel hybride : OLS macro + PCA sur résidus
├── sim/           # Simulateur Monte Carlo joint conditionné aux régimes & chocs gaussiens
├── portfolio/     # Règle de rééquilibrage institutionnelle Swensen 5/25 & DCA passif
├── index/         # Prédiction de reconstitution annuelle Nasdaq-100 & admission S&P 500
└── backtest/      # Validation out-of-sample stricte : CRPS vs benchmark naïf, PIT histogram
```

---

## 🔬 Principes Clés

1. **Aucune donnée fabriquée** :
   - Les cotes des marchés de prédiction sont interrogées en direct via l'API publique Gamma de Polymarket avec horodatage UTC et volumes réels. En cas d'indisponibilité réseau, l'état hors-ligne est explicitement signalé.
2. **Cohérence mathématique & correction de convexité** :
   - Correction stricte de l'écart arithmétique/log : $\mu_{\log} = \mu_{\text{arith}} - \frac{1}{2} \sigma^2$.
   - Les scénarios de stress macroéconomique sont transmis via les bêtas estimés des facteurs (`MKT`, `RATES`, `OIL`, `USD`, `GOLD`, etc.) et conditionnés par projection gaussienne.
3. **Validation statistique out-of-sample** :
   - Backtest walk-forward sans biais de survie ni fuite d'information future.
   - Évaluation par Continuous Ranked Probability Score (CRPS) comparé au modèle de marche aléatoire gaussienne naïve, et calibration via l'histogramme PIT (*Probability Integral Transform*).

---

## 🚀 Installation & Exécution

### 1. Prérequis
Python 3.10+ (testé et validé sous Python 3.14).

```powershell
# Cloner le dépôt et se placer sur la branche rewrite
git clone https://github.com/GuillaumeAB/titan-omnimacro.git
cd titan-omnimacro
git checkout rewrite

# Créer l'environnement virtuel et installer les dépendances
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Exécuter la suite de tests unitaires
```powershell
pytest tests/ -v
```

Tous les tests s'exécutent en local sur des processus générateurs de données synthétiques (DGP) à vérité terrain connue, sans dépendance externe ni appel réseau.
