
"""
OODA-MAT
OODA-loop based interactive AI-assisted platform for alkali-free / cement-free
construction-material optimization.

Run:
    pip install -r requirements.txt
    streamlit run OODA-C3.py

This prototype:
- can connect its Conversation tab to the OpenAI Responses API;
- accepts uploaded experimental CSV data;
- trains Gaussian-process surrogate models;
- visualizes Observe–Orient–Decide–Act inputs and outputs;
- proposes constrained multi-objective candidate mixtures;
- records expert feedback and exports experiment packets.

Engineering notice:
Predictions are decision support, not proof of hydration mechanism or field suitability.
Validate all proposed mixtures experimentally and apply relevant standards.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_NAME = "OODA-C3"
VERSION = "3.6.0"

FEATURES = [
    "GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct",
    "CaCl2_pct", "Limestone_pct", "WB", "CuringTemp_C"
]

PERFORMANCE_TARGETS = [
    "Strength3d_MPa", "Strength28d_MPa", "Absorption_pct",
    "Cost_KRW_t", "CO2_kg_t", "Expansion_pct"
]

# Semi-quantitative XRD phase fractions. Use one consistent quantification method
# (preferably Rietveld + internal standard for amorphous content) across a project.
XRD_TARGETS = [
    "XRD_CASH_CSH_pct", "XRD_Ettringite_pct", "XRD_Hydrotalcite_pct",
    "XRD_Calcite_pct", "XRD_ResidualGypsum_pct", "XRD_Amorphous_pct"
]

# TG mass losses are separated into temperature windows to avoid mixing distinct
# dehydration/dehydroxylation/decarbonation mechanisms in a single total-loss value.
TG_TARGETS = [
    "TG_Loss_30_200_pct", "TG_Loss_200_400_pct",
    "TG_Loss_400_550_pct", "TG_Loss_550_800_pct", "TG_TotalLoss_pct"
]

# DTG peak descriptors. Peak assignment is system-dependent; temperatures and
# intensities are stored first, while phase attribution remains an expert decision.
DTG_TARGETS = [
    "DTG_Peak1_T_C", "DTG_Peak1_Intensity_pct_min",
    "DTG_Peak2_T_C", "DTG_Peak2_Intensity_pct_min",
    "DTG_Peak3_T_C", "DTG_Peak3_Intensity_pct_min"
]

CHEMICAL_TARGETS = XRD_TARGETS + TG_TARGETS + DTG_TARGETS
TARGETS = PERFORMANCE_TARGETS + CHEMICAL_TARGETS

INPUT_GROUP_COLUMNS = {
    "General": ["ExperimentID", *FEATURES, *PERFORMANCE_TARGETS, "AnalysisAge_d", "HeatingRate_C_min", "Source"],
    "XRD": ["ExperimentID", *XRD_TARGETS, "XRD_Method"],
    "TG": ["ExperimentID", *TG_TARGETS, "TG_Atmosphere"],
    "DTG": ["ExperimentID", *DTG_TARGETS],
}

BINDER_COMPONENTS = [
    "GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct",
    "CaCl2_pct", "Limestone_pct",
]

HYDRATION_MODELS = {
    "Performance surrogate": {
        "basis": (
            "Independent Gaussian-process regressors relate mixture composition, W/B, "
            "curing temperature and proxy chemistry descriptors to measured engineering "
            "performance. A Matern-5/2 covariance represents a smooth but non-linear "
            "response surface, while the white-noise term represents unexplained test and "
            "process variability."
        ),
        "required_input": (
            "GGBFS, fly ash, CaO, gypsum, CaCl2 and limestone mass fractions; W/B; "
            "curing temperature; and at least eight valid measurements for each requested "
            "performance target. Supplier lot, test age, specimen geometry and test standard "
            "should remain consistent or be added as explicit variables."
        ),
        "output": (
            "Predicted 3 d and 28 d strength, absorption, expansion, binder cost and embodied "
            "CO2, each with a model standard deviation. Cost and CO2 can fall back to the "
            "configured deterministic material-factor equations when no fitted model exists."
        ),
        "interpretation": (
            "Use for screening and experiment selection. It is an empirical surrogate and "
            "does not identify hydration products or prove a reaction mechanism."
        ),
        "targets": PERFORMANCE_TARGETS,
    },
    "XRD phase-fraction surrogate": {
        "basis": (
            "Separate Gaussian-process regressors map mixture and curing variables to each "
            "semi-quantitative XRD descriptor. The model learns correlations with quantified "
            "C-(A)-S-H/gel proxy, ettringite, hydrotalcite, calcite, residual gypsum and "
            "amorphous fraction."
        ),
        "required_input": (
            "The common mixture/process inputs plus phase fractions measured at a controlled "
            "analysis age using one documented quantification workflow. Rietveld analysis with "
            "an internal standard is recommended when amorphous content is required."
        ),
        "output": (
            "A predicted value and standard deviation for every XRD target having at least "
            "eight valid rows. Each phase is modeled independently; phase closure to 100% is "
            "not imposed by the current implementation."
        ),
        "interpretation": (
            "Treat the output as a phase-profile hypothesis. Broad humps or fitted gel "
            "contributions must not be interpreted as unique phase identification without "
            "supporting chemistry and thermal evidence."
        ),
        "targets": XRD_TARGETS,
    },
    "TG mass-loss surrogate": {
        "basis": (
            "Separate Gaussian-process regressors predict mass loss in fixed temperature "
            "windows (30-200, 200-400, 400-550 and 550-800 °C) and total loss. Windowing keeps "
            "overlapping dehydration, dehydroxylation and decarbonation signals as descriptors "
            "instead of assigning every loss to a single phase."
        ),
        "required_input": (
            "The common mixture/process inputs and baseline-corrected TG measurements acquired "
            "with consistent atmosphere, heating rate, sample mass, crucible, preconditioning "
            "and analysis age."
        ),
        "output": (
            "Predicted mass loss and standard deviation for each available temperature window "
            "and total mass loss. Window predictions and total loss are independent in the "
            "current model, so their arithmetic consistency must be checked."
        ),
        "interpretation": (
            "Temperature windows are comparative descriptors, not universal phase assignments. "
            "Interpret them jointly with XRD, DTG and composition."
        ),
        "targets": TG_TARGETS,
    },
    "DTG peak surrogate": {
        "basis": (
            "Gaussian-process regressors independently learn the temperature and intensity of "
            "up to three ordered DTG peaks from mixture and curing variables. Separating peak "
            "position from intensity avoids assuming that a change in one necessarily implies "
            "the same change in the other."
        ),
        "required_input": (
            "The common mixture/process inputs plus consistently detected DTG peak temperature "
            "and intensity. Peak ordering, smoothing, baseline correction, atmosphere and "
            "heating rate must be defined consistently across experiments."
        ),
        "output": (
            "Predicted peak temperatures and intensities with model standard deviations for "
            "each DTG descriptor having sufficient measured rows."
        ),
        "interpretation": (
            "Peak numbers are descriptors rather than fixed phase labels. Phase attribution "
            "remains an expert interpretation supported by XRD, TG windows and chemistry."
        ),
        "targets": DTG_TARGETS,
    },
}

# Performance columns are mandatory for the base optimizer. Chemical-analysis
# columns are optional; models are trained whenever at least eight valid rows exist.
REQUIRED_BASE = FEATURES + PERFORMANCE_TARGETS
OPTIONAL_ANALYSIS = CHEMICAL_TARGETS

TOOL_LINKS = {
    "Streamlit": {
        "stage": "All stages / UI",
        "url": "https://docs.streamlit.io/get-started/installation",
        "example": "pip install streamlit; streamlit run OODA-C3.py",
        "role": "Interactive dashboard, chat interface, file upload and visualization",
    },
    "scikit-learn": {
        "stage": "Orient",
        "url": "https://scikit-learn.org/stable/install.html",
        "example": "GaussianProcessRegressor(kernel=Matern(...))",
        "role": "Gaussian-process surrogate model and uncertainty estimation",
    },
    "Optuna": {
        "stage": "Decide",
        "url": "https://optuna.readthedocs.io/en/stable/installation.html",
        "example": "optuna.create_study(directions=['maximize','minimize'])",
        "role": "Optional constrained and multi-objective optimization engine",
    },
    "BoTorch": {
        "stage": "Decide",
        "url": "https://botorch.org/docs/getting_started",
        "example": "SingleTaskGP + acquisition function",
        "role": "Advanced Bayesian optimization for expensive experiments",
    },
    "MLflow": {
        "stage": "Act / Feedback",
        "url": "https://mlflow.org/docs/latest/ml/getting-started/",
        "example": "mlflow.log_params(...); mlflow.log_metrics(...)",
        "role": "Experiment, model, metric and artifact tracking",
    },
    "PHREEQC": {
        "stage": "Orient",
        "url": "https://www.usgs.gov/software/phreeqc-version-3",
        "example": "Calculate aqueous speciation and saturation indices",
        "role": "Pore-solution speciation and geochemical equilibrium support",
    },
    "PHREEQC downloads": {
        "stage": "Orient",
        "url": "https://water.usgs.gov/water-resources/software/PHREEQC/index.html",
        "example": "Install PHREEQC 3 and connect through file or Python wrapper",
        "role": "Official USGS executable, databases and examples",
    },
    "GEMS": {
        "stage": "Orient",
        "url": "https://gems.web.psi.ch/",
        "example": "Thermodynamic phase-equilibrium calculation",
        "role": "Thermodynamic modeling of cementitious and geochemical systems",
    },
    "Zotero": {
        "stage": "Observe",
        "url": "https://www.zotero.org/download/",
        "example": "Store DOI, test conditions and source PDFs",
        "role": "Literature and reference management",
    },
    "GROBID": {
        "stage": "Observe",
        "url": "https://github.com/kermitt2/grobid",
        "example": "Convert scientific PDFs to structured TEI XML",
        "role": "Structured extraction from papers",
    },
    "PostgreSQL": {
        "stage": "Observe / Feedback",
        "url": "https://www.postgresql.org/download/",
        "example": "Store material lots, mixtures, tests and decision logs",
        "role": "Structured research database",
    },
    "LangGraph": {
        "stage": "Workflow orchestration",
        "url": "https://docs.langchain.com/oss/python/langgraph/overview",
        "example": "Define Observe→Orient→Decide→Act state transitions",
        "role": "Optional multi-agent state-machine orchestration",
    },
}


# Embedded physics/chemistry-informed hydration models (standalone integration).
# Source module: OODA-MAT_v2.0/hydration_models.py

R_GAS = 8.314462618  # J mol-1 K-1
T_REF_K = 298.15

MIX_COLUMNS = [
    "GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct",
    "CaCl2_pct", "Limestone_pct", "WB", "CuringTemp_C"
]

DEFAULT_XRF = {
    "GGBFS": {"CaO": 40.0, "SiO2": 35.0, "Al2O3": 13.0, "MgO": 8.0, "SO3": 1.5},
    "FlyAsh": {"CaO": 5.0, "SiO2": 55.0, "Al2O3": 25.0, "MgO": 2.0, "SO3": 0.8},
}

DENSITY = {
    "GGBFS": 2.90, "FlyAsh": 2.30, "CaO": 3.34,
    "Gypsum": 2.32, "CaCl2": 2.15, "Limestone": 2.71,
    "hydrate": 2.05,
}

MODEL_INFO = {
    "Krstulovic-Dabic": {
        "classification": "Lumped thermokinetic reaction-degree model",
        "basis": (
            "Hydration is represented by three simultaneously admissible rate processes: "
            "nucleation and crystal growth (NG), reaction at the phase boundary (I), and "
            "diffusion through an increasingly continuous product layer (D). At each time, "
            "the smallest attainable conversion among the three formal branches is used as "
            "the controlling envelope. The present implementation applies a composition-based "
            "maximum reaction capacity and an induction period, and scales the kinetic "
            "constants with temperature, fineness, water availability and activator dosage. "
            "It is therefore a reduced reaction-degree model fitted to heat-release or "
            "reaction-degree data, not a species-resolved chemical model."
        ),
        "governing_equations": [
            "tau = max(t - t_ind, 0)",
            "alpha_NG = alpha_max * {1 - exp[-(K_NG * tau)^n]}",
            "alpha_I = alpha_max * {1 - [1 - K_I * tau]^3}",
            "alpha_D = alpha_max * {1 - [1 - sqrt(K_D * tau)]^3}",
            "alpha(t) = monotonic envelope of min(alpha_NG, alpha_I, alpha_D)",
            "K(T) = K_ref * exp[-Ea/R * (1/T - 1/T_ref)]",
            "Q(t) = Q_inf * alpha(t)"
        ],
        "state_variables": (
            "Reaction degree alpha, reaction rate d(alpha)/dt, cumulative heat Q, heat-flow "
            "proxy, controlling process, induction time and fitted kinetic constants."
        ),
        "inputs": (
            "Time, curing temperature, isothermal-calorimetry heat release or another measured "
            "reaction-degree proxy, GGBFS/fly-ash fractions, CaO, gypsum, CaCl2, limestone, "
            "W/B, fineness and independently estimated maximum reactive fractions."
        ),
        "outputs": (
            "Time-dependent reaction degree, rate, cumulative heat, heat-flow proxy, controlling "
            "NG/I/D process and kinetic transition behavior."
        ),
        "numerical_scheme": (
            "Closed-form branch equations are evaluated on the requested time grid. The "
            "minimum branch conversion is made non-decreasing to remove numerical reversals. "
            "No spatial discretization or chemical-equilibrium iteration is performed."
        ),
        "calibration": (
            "Fit t_ind, K_NG, n, K_I, K_D, Ea, alpha_max and Q_inf using multi-temperature "
            "calorimetry. Prefer simultaneous fitting of cumulative heat and heat-flow with "
            "bounded parameters. Determine alpha_max from selective dissolution, quantitative "
            "XRD/TG mass balance, or long-age calorimetry rather than assuming complete reaction."
        ),
        "validation": (
            "Validate against independent calorimetry curves and reaction degrees at 1, 3, 7 "
            "and 28 days. Check whether the inferred NG→I→D transition is stable when the "
            "fitting window, baseline correction and smoothing method are changed."
        ),
        "applicability": (
            "Useful for comparative kinetics and maturity effects in a fixed material family. "
            "For Ca-rich alkali-free slag binders, each raw-material lot requires recalibration "
            "because glass chemistry, fineness and sulfate/chloride activation alter the fitted "
            "constants."
        ),
        "limitation": (
            "The slowest-branch rule is phenomenological and does not independently predict "
            "hydrate species, ion concentrations, precipitation affinity or spatial pore structure."
        ),
        "references": [
            "Krstulovic and Dabic, Cement and Concrete Research 30 (2000) 693-698, doi:10.1016/S0008-8846(00)00231-3"
        ],
    },
    "GEMS reduced-order preview": {
        "classification": "Element-inventory and empirical phase-allocation preview",
        "basis": (
            "A true GEMS calculation minimizes the total Gibbs energy of a multiphase system "
            "subject to elemental mass balance, charge balance, phase-stability and activity-"
            "model constraints. OODA-MAT first converts the dry mixture and precursor XRF data "
            "to a bulk Ca-Si-Al-Mg-S-C-Cl-H-O inventory. The embedded preview then uses bounded "
            "allocation rules driven by reaction degree to estimate C-(A)-S-H, ettringite, "
            "hydrotalcite-like phase, carboaluminate, calcite, residual gypsum and unreacted "
            "precursor. The same inventory is exported as an auditable GEMS/CemGEMS recipe. "
            "The preview itself does not minimize Gibbs energy and must not be reported as an "
            "equilibrium calculation."
        ),
        "governing_equations": [
            "minimize G = sum_i(n_i * mu_i) subject to A*n = b and n_i >= 0",
            "mu_i = mu_i^0(T,P) + R*T*ln(a_i)",
            "Element inventory b is calculated from XRF plus stoichiometric CaO, SO3, CO2 and Cl contributions",
            "Embedded preview: phase_j = bounded allocation_j(element inventory, alpha, reaction extents)",
            "Phase fractions are normalized to 100% only for screening visualization"
        ],
        "state_variables": (
            "Bulk element inventory, phase amounts, aqueous composition, activities, saturation "
            "indices and chemical potentials in a true GEMS run; estimated phase fractions, pH "
            "proxy and ionic-strength proxy in the embedded preview."
        ),
        "inputs": (
            "Lot-specific XRF including CaO, SiO2, Al2O3, MgO, SO3, alkalis and Fe where relevant; "
            "water content; temperature and pressure; gypsum/anhydrite identity; limestone and "
            "CaCl2 purity; precursor reaction extents; permitted phases and Cemdata18 database."
        ),
        "outputs": (
            "Embedded preview: screening phase profile and GEMS input recipe. External GEMS/"
            "CemGEMS: equilibrium or constrained-equilibrium phase assemblage, pore-solution "
            "composition, activities, pH and saturation state."
        ),
        "numerical_scheme": (
            "The embedded preview is algebraic. A true GEMS calculation uses Gibbs-energy "
            "minimization with non-ideal aqueous and solid-solution models. Early-age simulation "
            "should be implemented as constrained equilibrium by prescribing measured precursor "
            "reaction extents at each age."
        ),
        "calibration": (
            "Use quantitative XRD with an internal standard, TG/DTG, pore-solution ICP/IC and "
            "measured reaction degrees to constrain the amount of reacted slag and fly ash. "
            "Review the enabled solid solutions for AFt, AFm, C-(A)-S-H and hydrotalcite and "
            "exclude phases that are kinetically inaccessible under the selected curing condition."
        ),
        "validation": (
            "Check elemental closure, charge balance, phase closure, predicted bound water and "
            "predicted pore-solution ions. Cross-check calcite/carboaluminate and sulfate-bearing "
            "phases against both XRD and TG/DTG rather than either technique alone."
        ),
        "applicability": (
            "Best suited to phase stability, sulfate-carbonate-aluminate competition, Mg-bearing "
            "hydrates and pore-solution chemistry. It should be coupled to measured kinetics for "
            "early-age alkali-free slag systems."
        ),
        "limitation": (
            "The embedded result is not Gibbs-energy minimization. Thermodynamic equilibrium "
            "does not determine the actual reaction rate, spatial microstructure or strength."
        ),
        "references": [
            "Lothenbach et al., Cemdata18, Cement and Concrete Research 115 (2019) 472-506, doi:10.1016/j.cemconres.2018.04.018",
            "Kulik et al., CemGEMS, RILEM Technical Letters 6 (2021) 36-52, doi:10.21809/rilemtechlett.2021.140"
        ],
    },
    "CEMHYD3D-inspired": {
        "classification": "Voxel/cellular-automaton-inspired microstructure surrogate",
        "basis": (
            "CEMHYD3D represents an initial three-dimensional digital microstructure as discrete "
            "voxels assigned to anhydrous phases, water and pores. Hydration proceeds through "
            "phase-specific dissolution, transport by random walks and local precipitation or "
            "growth rules; the evolving voxel topology is used to calculate phase connectivity "
            "and transport-related properties. OODA-MAT retains the physical ideas of reacted "
            "solid, hydrate-volume expansion, capillary-water consumption, gel porosity and "
            "percolation, but replaces the full phase-resolved cellular automaton with a reduced "
            "volume-balance model and an illustrative two-dimensional particle slice."
        ),
        "governing_equations": [
            "V_solid,0 = sum_i(m_i / rho_i)",
            "V_hydrate = V_solid,0 * alpha_norm * expansion_factor(composition)",
            "V_capillary = max[V_water,0 - V_bound_water - k_pore*V_hydrate, 0]",
            "phi_cap = V_capillary / V_total; phi_gel = V_gel_pore / V_total",
            "Connectivity = 1 / {1 + exp[-k*(V_hydrate/V_total - critical_fraction)]}",
            "Strength proxy = S_inf * (gel-space ratio)^m * connectivity factor"
        ],
        "state_variables": (
            "Unreacted precursor, hydrate volume, capillary-pore volume, gel-pore volume, "
            "hydrate fraction, connectivity, gel-space ratio and strength proxy."
        ),
        "inputs": (
            "Composition, phase densities, particle-size/fineness descriptors, W/B, reaction "
            "degree, hydrate expansion factors, bound-water coefficients and phase-specific "
            "reaction/morphology calibration."
        ),
        "outputs": (
            "Time-dependent capillary and gel porosity, hydrate volume fraction, connectivity, "
            "gel-space ratio, strength proxy and a representative microstructure image."
        ),
        "numerical_scheme": (
            "The present reduced model applies deterministic volume balance and logistic "
            "percolation equations. The optional image generator places particles stochastically "
            "and grows shells in two dimensions; it is a visualization, not a solved 3-D voxel field."
        ),
        "calibration": (
            "Calibrate initial packing from measured PSD and paste density; calibrate reaction "
            "degree with calorimetry; calibrate hydrate volume and bound water with TG and helium/"
            "water porosity; calibrate connectivity with MIP, electrical resistivity or transport "
            "tests; calibrate strength only after the pore model is fixed."
        ),
        "validation": (
            "Compare simulated total porosity, capillary-pore fraction, non-evaporable water and "
            "connectivity against independent measurements at several ages. Use image statistics "
            "or tomography when a genuine spatial validation is required."
        ),
        "applicability": (
            "Useful for linking reaction degree to pore filling and strength trends. Original "
            "CEMHYD3D phase rules were developed mainly for Portland-cement systems; alkali-free "
            "slag, chloride and sulfate reactions require new reaction and morphology parameters."
        ),
        "limitation": (
            "The embedded model is not NIST CEMHYD3D Version 3 and does not execute calibrated "
            "3-D phase-specific dissolution, diffusion and precipitation rules."
        ),
        "references": [
            "Bentz, CEMHYD3D Version 3.0, NISTIR 7232 (2005)"
        ],
    },
    "HYMOSTRUC3D-inspired": {
        "classification": "Particle-shell growth and contact/percolation surrogate",
        "basis": (
            "HYMOSTRUC treats the binder as a population of particles with a measured or assumed "
            "size distribution. Hydration causes unreacted cores to shrink and product shells to "
            "grow. Shell overlap creates interparticle contacts; the evolution of contacts and "
            "connected solid networks controls setting, stiffness and transport. OODA-MAT creates "
            "a reproducible statistical three-dimensional particle cloud, grows shells as a "
            "function of normalized reaction degree, and derives mean contact number, active "
            "particle fraction and a logistic percolation index."
        ),
        "governing_equations": [
            "Particle diameter follows a lognormal PSD parameterized by D50 and spread",
            "r_shell,i(t) = r_i * [1 + beta(composition)*alpha_norm^p]",
            "Contact_ij = 1 when distance_ij <= r_shell,i + r_shell,j",
            "Mean contact number = mean_i(sum_j Contact_ij)",
            "Percolation index = 1 / {1 + exp[-k*(mean contact number - z_c)]}",
            "Strength proxy = gel-space strength * [a + (1-a)*percolation]"
        ],
        "state_variables": (
            "Core and shell radii, contact matrix, mean coordination number, active-particle "
            "fraction, percolation index, capillary porosity and strength proxy."
        ),
        "inputs": (
            "Particle-size distribution or D50/spread proxy, packing density, W/B, reaction "
            "degree, hydrate-shell expansion, precursor fractions and temperature-dependent kinetics."
        ),
        "outputs": (
            "Contact density, active-particle fraction, percolation/connectivity, capillary "
            "porosity, strength proxy and particle-shell snapshot."
        ),
        "numerical_scheme": (
            "A seeded Monte Carlo particle cloud is generated once for each simulation. Pairwise "
            "distances are calculated, and shell overlap is evaluated at every age. The current "
            "implementation does not solve ion transport or local chemical equilibrium."
        ),
        "calibration": (
            "Use measured PSD rather than D50 alone; calibrate packing with paste density or image "
            "analysis; fit shell-growth parameters to calorimetry and chemically bound water; fit "
            "the percolation threshold to setting time, ultrasonic pulse velocity or resistivity."
        ),
        "validation": (
            "Validate contact/percolation timing independently from compressive strength. For "
            "slag-rich systems, verify long-age shell growth and pore-solution evolution with "
            "HYMOSTRUC3D-E-type chemistry or an external thermodynamic model."
        ),
        "applicability": (
            "Useful for particle-size, fineness, packing and setting/percolation studies. "
            "HYMOSTRUC3D-E is the more relevant conceptual reference for slag-containing binders "
            "because it extends the framework toward blended-cement reaction and pore solution."
        ),
        "limitation": (
            "The embedded particle cloud is statistical and is not a reconstruction of the "
            "actual paste. Full HYMOSTRUC3D-E chemistry and transport are not reproduced."
        ),
        "references": [
            "van Breugel, Simulation of hydration and formation of structure in hardening cement-based materials (1995)",
            "Gao et al., Extension of HYMOSTRUC3D for slag cement hydration and pore solution chemistry (2019)"
        ],
    },
    "OODA-MAT hybrid": {
        "classification": "Sequentially coupled kinetics-chemistry-microstructure screening model",
        "basis": (
            "The hybrid model uses one Krstulovic-Dabic reaction-degree trajectory as the common "
            "time coordinate. That trajectory constrains the reduced GEMS phase allocation, "
            "CEMHYD3D-inspired volume balance and HYMOSTRUC-inspired particle-shell network. "
            "The predicted strength combines gel-space/connectivity and contact-network terms, "
            "then applies bounded phase modifiers for residual gypsum, C-(A)-S-H and ettringite. "
            "Information therefore flows from kinetics to chemistry and microstructure, but no "
            "fully implicit feedback from pore solution or transport back to reaction rate is solved."
        ),
        "governing_equations": [
            "alpha(t) <- Krstulovic-Dabic kinetics",
            "phase assemblage(t) <- allocation(element inventory, alpha, precursor extents)",
            "porosity/connectivity(t) <- volume balance(alpha, W/B, composition)",
            "contact network(t) <- particle shell growth(alpha, PSD)",
            "f_hybrid = [0.60*f_gel-space + 0.40*f_contact] * sulfate_penalty * phase_bonus"
        ],
        "state_variables": (
            "Reaction degree, cumulative heat, phase assemblage, residual precursor, capillary "
            "and gel porosity, hydrate connectivity, particle percolation and hybrid strength."
        ),
        "inputs": (
            "Composition, XRF, reactive fractions, PSD/BET/Blaine, W/B, temperature, calorimetry, "
            "XRD-Rietveld, TG/DTG, pore-solution data, porosity/transport data and strength."
        ),
        "outputs": (
            "A time-resolved digital-thread summary of kinetics, phase development, pore filling, "
            "network formation and strength for OODA candidate comparison."
        ),
        "numerical_scheme": (
            "Sequential explicit coupling on a common time grid. Each submodel receives alpha(t) "
            "and mixture descriptors, then outputs state variables used by the final strength "
            "relation. This design is computationally stable and transparent but not strongly coupled."
        ),
        "calibration": (
            "Calibrate in stages: (1) kinetics from calorimetry; (2) precursor reaction extent and "
            "phases from XRD/TG and GEMS; (3) pore-volume coefficients from porosity/transport; "
            "(4) particle percolation from setting/resistivity; (5) strength coefficients last. "
            "Use separate calibration and validation batches and retain lot identifiers."
        ),
        "validation": (
            "Require simultaneous agreement with heat, reaction degree, at least two chemical-phase "
            "indicators, porosity/connectivity and strength. A good strength fit alone is insufficient "
            "because compensating errors may exist between kinetics, phase allocation and pore structure."
        ),
        "applicability": (
            "Recommended as the OODA-MAT screening and experiment-selection layer. Use the external "
            "GEMS/CemGEMS calculation and laboratory measurements as correction sources, not as "
            "optional decoration."
        ),
        "limitation": (
            "Not a fully coupled reactive-transport or thermodynamic-kinetic solver. Its numerical "
            "constants are priors and must not be transferred between raw-material lots without validation."
        ),
        "references": [
            "Couples the concepts documented in the Krstulovic-Dabic, Cemdata18/CemGEMS, CEMHYD3D and HYMOSTRUC3D references above"
        ],
    },
}

EXAMPLE_ALTERNATIVES = pd.DataFrame([
    {
        "Alternative": "A_EarlyStrength",
        "Description": "High-slag, CaO/CaCl2 accelerated reference",
        "GGBFS_pct": 79.0, "FlyAsh_pct": 8.0, "CaO_pct": 5.0,
        "Gypsum_pct": 5.0, "CaCl2_pct": 3.0, "Limestone_pct": 0.0,
        "WB": 0.38, "CuringTemp_C": 25.0,
        "Blaine_m2kg": 450.0, "D50_um": 12.0, "ReactiveSlag_frac": 0.78,
        "ReactiveFlyAsh_frac": 0.30,
    },
    {
        "Alternative": "B_LowChlorideSulfate",
        "Description": "Lower chloride with sulfate/aluminate control",
        "GGBFS_pct": 78.0, "FlyAsh_pct": 10.0, "CaO_pct": 4.0,
        "Gypsum_pct": 7.0, "CaCl2_pct": 1.0, "Limestone_pct": 0.0,
        "WB": 0.38, "CuringTemp_C": 25.0,
        "Blaine_m2kg": 440.0, "D50_um": 13.0, "ReactiveSlag_frac": 0.76,
        "ReactiveFlyAsh_frac": 0.32,
    },
    {
        "Alternative": "C_LimestoneModified",
        "Description": "Limestone nucleation/carboaluminate alternative",
        "GGBFS_pct": 74.0, "FlyAsh_pct": 10.0, "CaO_pct": 4.0,
        "Gypsum_pct": 6.0, "CaCl2_pct": 1.5, "Limestone_pct": 4.5,
        "WB": 0.37, "CuringTemp_C": 25.0,
        "Blaine_m2kg": 480.0, "D50_um": 10.0, "ReactiveSlag_frac": 0.76,
        "ReactiveFlyAsh_frac": 0.32,
    },
    {
        "Alternative": "D_LowCO2FlyAshRich",
        "Description": "Fly-ash-rich, reduced activator and carbon burden",
        "GGBFS_pct": 68.0, "FlyAsh_pct": 20.0, "CaO_pct": 3.0,
        "Gypsum_pct": 6.0, "CaCl2_pct": 1.0, "Limestone_pct": 2.0,
        "WB": 0.39, "CuringTemp_C": 30.0,
        "Blaine_m2kg": 420.0, "D50_um": 15.0, "ReactiveSlag_frac": 0.73,
        "ReactiveFlyAsh_frac": 0.38,
    },
])


def _get(mix: Mapping[str, float], key: str, default: float) -> float:
    try:
        value = float(mix.get(key, default))
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def validate_mix(mix: Mapping[str, float]) -> List[str]:
    issues: List[str] = []
    solids = sum(_get(mix, k, 0.0) for k in [
        "GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct", "CaCl2_pct", "Limestone_pct"
    ])
    if abs(solids - 100.0) > 0.5:
        issues.append(f"Dry-binder composition sums to {solids:.2f}%, not 100%.")
    wb = _get(mix, "WB", 0.38)
    if not 0.20 <= wb <= 0.70:
        issues.append("W/B is outside the model's screening domain (0.20-0.70).")
    if _get(mix, "CaO_pct", 0.0) > 8.0:
        issues.append("CaO exceeds 8%; expansion/free-lime calibration is required.")
    if _get(mix, "CaCl2_pct", 0.0) > 3.0:
        issues.append("CaCl2 exceeds the default screening domain and requires chloride/safety review.")
    return issues


def arrhenius_factor(temp_c: float, ea_kj_mol: float = 65.0, ref_c: float = 25.0) -> float:
    t = temp_c + 273.15
    tref = ref_c + 273.15
    exponent = -ea_kj_mol * 1000.0 / R_GAS * (1.0 / t - 1.0 / tref)
    return float(np.clip(np.exp(exponent), 0.05, 20.0))


def reaction_capacity(mix: Mapping[str, float]) -> float:
    slag = _get(mix, "GGBFS_pct", 0.0) / 100.0
    fa = _get(mix, "FlyAsh_pct", 0.0) / 100.0
    rs = _get(mix, "ReactiveSlag_frac", 0.76)
    rf = _get(mix, "ReactiveFlyAsh_frac", 0.32)
    cao = _get(mix, "CaO_pct", 0.0) / 100.0
    gypsum = _get(mix, "Gypsum_pct", 0.0) / 100.0
    # Maximum reacted fraction of the total dry binder over 28-90 d.
    cap = slag * rs + fa * rf + 0.95 * cao + 0.55 * gypsum
    return float(np.clip(cap, 0.25, 0.92))


def kinetic_parameters(mix: Mapping[str, float]) -> Dict[str, float]:
    slag = _get(mix, "GGBFS_pct", 75.0) / 100.0
    fa = _get(mix, "FlyAsh_pct", 10.0) / 100.0
    cao = _get(mix, "CaO_pct", 4.0)
    gypsum = _get(mix, "Gypsum_pct", 6.0)
    cacl2 = _get(mix, "CaCl2_pct", 1.5)
    limestone = _get(mix, "Limestone_pct", 0.0)
    wb = _get(mix, "WB", 0.38)
    temp = _get(mix, "CuringTemp_C", 25.0)
    blaine = _get(mix, "Blaine_m2kg", 450.0)

    temp_fac = arrhenius_factor(temp, ea_kj_mol=70.0)
    surface_fac = np.clip((blaine / 450.0) ** 0.45, 0.65, 1.55)
    water_fac = np.clip(1.0 - 3.0 * max(0.0, 0.34 - wb) - 1.1 * max(0.0, wb - 0.45), 0.35, 1.15)
    activation = 0.55 + 0.075 * cao + 0.10 * cacl2 + 0.020 * gypsum
    filler_nucleation = 1.0 + 0.025 * limestone * surface_fac
    dilution = np.clip(1.0 - 0.55 * fa - 0.005 * limestone, 0.68, 1.02)
    base = temp_fac * surface_fac * water_fac * activation * filler_nucleation * dilution

    k_ng = 0.025 * base
    n = np.clip(1.8 + 0.05 * cacl2 + 0.02 * limestone - 0.015 * fa * 100, 1.25, 2.6)
    k_i = 0.012 * base * (0.92 + 0.22 * slag)
    # A higher Blaine fineness generally shortens the characteristic diffusion
    # distance and increases reactive surface. The previous inverse-Blaine term
    # over-penalized fine powders and could reverse the expected early-age trend.
    # Water-demand/rheology penalties must be represented separately through W/B
    # or calibrated flow descriptors rather than embedded as a large negative
    # diffusion coefficient.
    diffusion_length_fac = np.clip((blaine / 450.0) ** 0.20, 0.80, 1.25)
    k_d = (
        0.0019 * temp_fac * water_fac
        * (0.85 + 0.25 * slag)
        * diffusion_length_fac
    )
    induction_h = np.clip(4.8 - 0.55 * cacl2 - 0.20 * cao + 0.08 * fa * 100 / 10.0, 0.4, 8.0)
    q_inf = 330.0 * slag + 210.0 * fa + 540.0 * cao / 100.0 + 210.0 * gypsum / 100.0
    q_inf = float(np.clip(q_inf, 180.0, 480.0))
    return {
        "K_NG_h-1": float(k_ng), "n": float(n), "K_I_h-1": float(k_i),
        "K_D_h-1": float(k_d), "induction_h": float(induction_h),
        "Q_inf_J_g": q_inf, "alpha_max": reaction_capacity(mix),
        "Ea_kJ_mol": 70.0,
    }


def krstulovic_dabic(mix: Mapping[str, float], times_h: Iterable[float]) -> pd.DataFrame:
    """Evaluate formal NG/I/D conversion equations and slowest-process envelope."""
    p = kinetic_parameters(mix)
    t = np.asarray(list(times_h), dtype=float)
    tau = np.maximum(t - p["induction_h"], 0.0)
    kng, n, ki, kd = p["K_NG_h-1"], p["n"], p["K_I_h-1"], p["K_D_h-1"]

    a_ng = 1.0 - np.exp(-np.power(kng * tau, n))
    a_i = 1.0 - np.power(np.clip(1.0 - ki * tau, 0.0, 1.0), 3.0)
    a_d = 1.0 - np.power(np.clip(1.0 - np.sqrt(np.maximum(kd * tau, 0.0)), 0.0, 1.0), 3.0)
    stack = np.vstack([a_ng, a_i, a_d])
    idx = np.argmin(stack + np.where(stack <= 0, 1e-12, 0), axis=0)
    raw = np.min(stack, axis=0)
    alpha = p["alpha_max"] * np.maximum.accumulate(np.clip(raw, 0.0, 1.0))
    if len(t) > 1:
        rate = np.gradient(alpha, t, edge_order=1)
    else:
        rate = np.zeros_like(alpha)
    heat = p["Q_inf_J_g"] * alpha
    heat_flow = p["Q_inf_J_g"] * rate
    process_names = np.array(["NG", "I", "D"])[idx]
    process_names[tau <= 0] = "Induction"
    return pd.DataFrame({
        "Time_h": t, "Alpha": alpha, "Rate_h-1": np.maximum(rate, 0.0),
        "CumulativeHeat_J_g": heat, "HeatFlow_J_g_h": np.maximum(heat_flow, 0.0),
        "Alpha_NG": p["alpha_max"] * a_ng,
        "Alpha_I": p["alpha_max"] * a_i,
        "Alpha_D": p["alpha_max"] * a_d,
        "ControllingProcess": process_names,
    })


def oxide_inventory(mix: Mapping[str, float], xrf: Mapping[str, Mapping[str, float]] | None = None) -> Dict[str, float]:
    xrf = xrf or DEFAULT_XRF
    slag = _get(mix, "GGBFS_pct", 0.0)
    fa = _get(mix, "FlyAsh_pct", 0.0)
    inv = {k: 0.0 for k in ["CaO", "SiO2", "Al2O3", "MgO", "SO3", "CO2", "Cl"]}
    for oxide in ["CaO", "SiO2", "Al2O3", "MgO", "SO3"]:
        inv[oxide] += slag * xrf["GGBFS"].get(oxide, 0.0) / 100.0
        inv[oxide] += fa * xrf["FlyAsh"].get(oxide, 0.0) / 100.0
    inv["CaO"] += _get(mix, "CaO_pct", 0.0)
    inv["CaO"] += _get(mix, "Gypsum_pct", 0.0) * 56.077 / 172.171
    inv["SO3"] += _get(mix, "Gypsum_pct", 0.0) * 80.063 / 172.171
    inv["CaO"] += _get(mix, "Limestone_pct", 0.0) * 56.077 / 100.087
    inv["CO2"] += _get(mix, "Limestone_pct", 0.0) * 44.010 / 100.087
    inv["CaO"] += _get(mix, "CaCl2_pct", 0.0) * 56.077 / 110.984
    inv["Cl"] += _get(mix, "CaCl2_pct", 0.0) * 70.906 / 110.984
    return inv


def gems_recipe(mix: Mapping[str, float], xrf: Mapping[str, Mapping[str, float]] | None = None) -> Dict:
    """Create an auditable export object for manual GEMS/CemGEMS recipe construction."""
    inv = oxide_inventory(mix, xrf)
    return {
        "model": "External GEMS/CemGEMS calculation required",
        "database": "Cemdata18 recommended for cementitious/alkali-activated phases",
        "temperature_C": _get(mix, "CuringTemp_C", 25.0),
        "pressure_bar": 1.0,
        "water_binder_mass_ratio": _get(mix, "WB", 0.38),
        "dry_binder_recipe_g_per_100g": {
            key.replace("_pct", ""): _get(mix, key, 0.0)
            for key in ["GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct", "CaCl2_pct", "Limestone_pct"]
        },
        "bulk_oxide_inventory_g_per_100g_binder": {k: round(v, 6) for k, v in inv.items()},
        "reaction_extent_inputs": {
            "ReactiveSlag_frac": _get(mix, "ReactiveSlag_frac", 0.76),
            "ReactiveFlyAsh_frac": _get(mix, "ReactiveFlyAsh_frac", 0.32),
            "note": "Use measured calorimetry/selective-dissolution/XRD constraints; do not assume full equilibrium at early age."
        },
        "candidate_phases_to_enable_or_review": [
            "C-(A)-S-H solid solution", "ettringite/AFt", "AFm/carboaluminate",
            "hydrotalcite-like phase", "calcite", "gypsum/anhydrite",
            "hydrogarnet where justified", "aqueous solution"
        ],
    }


def gems_phase_preview(mix: Mapping[str, float], times_h: Iterable[float]) -> pd.DataFrame:
    """Reduced-order phase-allocation preview; not Gibbs minimization."""
    kd = krstulovic_dabic(mix, times_h)
    inv = oxide_inventory(mix)
    slag = _get(mix, "GGBFS_pct", 0.0)
    fa = _get(mix, "FlyAsh_pct", 0.0)
    gypsum = _get(mix, "Gypsum_pct", 0.0)
    limestone = _get(mix, "Limestone_pct", 0.0)
    cacl2 = _get(mix, "CaCl2_pct", 0.0)
    cao = _get(mix, "CaO_pct", 0.0)

    rows = []
    for _, r in kd.iterrows():
        a = float(r["Alpha"])
        # Effective component reaction degrees; fly ash reacts more slowly.
        a_slag = np.clip(a / max(reaction_capacity(mix), 1e-6), 0.0, 1.0) * _get(mix, "ReactiveSlag_frac", 0.76)
        age_factor = 1.0 - math.exp(-float(r["Time_h"]) / 240.0)
        a_fa = np.clip((0.20 + 0.80 * age_factor) * a_slag, 0.0, _get(mix, "ReactiveFlyAsh_frac", 0.32))
        sulfate_reaction = np.clip(0.25 + 0.85 * a_slag + 0.05 * cacl2, 0.0, 1.0)
        limestone_reaction = np.clip((0.04 + 0.28 * a_slag) * (1.0 + 0.04 * inv["Al2O3"]), 0.0, 0.40)

        cash = 0.72 * slag * a_slag + 0.45 * fa * a_fa
        ettringite = min(gypsum * sulfate_reaction * 1.25, inv["Al2O3"] * 0.85) * (0.75 + 0.04 * cacl2)
        hydrotalcite = min(inv["MgO"] * 1.7 * a_slag, 0.13 * slag * a_slag)
        carbo = min(limestone * limestone_reaction * 1.35, inv["Al2O3"] * 0.35 * a_slag)
        calcite = limestone * (1.0 - limestone_reaction) + 0.06 * cao * a_slag
        residual_gypsum = gypsum * (1.0 - sulfate_reaction)
        unreacted = max(0.0, slag * (1.0 - a_slag) + fa * (1.0 - a_fa))
        other = max(0.0, 100.0 - (cash + ettringite + hydrotalcite + carbo + calcite + residual_gypsum + unreacted))
        phases = np.array([cash, ettringite, hydrotalcite, carbo, calcite, residual_gypsum, unreacted, other])
        phases = 100.0 * phases / max(phases.sum(), 1e-9)

        activator = cao + 0.75 * cacl2 + 0.18 * gypsum
        ph = np.clip(10.8 + 0.17 * activator + 0.45 * a_slag - 0.04 * limestone, 10.5, 13.4)
        ionic = np.clip(0.08 + 0.055 * cacl2 + 0.012 * gypsum + 0.006 * cao, 0.05, 0.80)
        rows.append({
            "Time_h": r["Time_h"], "Alpha": a,
            "CASH_CSH_pct": phases[0], "Ettringite_pct": phases[1],
            "Hydrotalcite_pct": phases[2], "Carboaluminate_pct": phases[3],
            "Calcite_pct": phases[4], "ResidualGypsum_pct": phases[5],
            "UnreactedPrecursor_pct": phases[6], "OtherSolids_pct": phases[7],
            "PoreSolution_pH_proxy": ph, "IonicStrength_molal_proxy": ionic,
        })
    data = pd.DataFrame(rows)
    float_columns = data.select_dtypes(include=["floating"]).columns
    data[float_columns] = data[float_columns].round(2)
    percentage_columns = [col for col in data.columns if "_pct" in col]
    data[percentage_columns] = data[percentage_columns].round(1)
    return data


def initial_volumes(mix: Mapping[str, float]) -> Dict[str, float]:
    masses = {
        "GGBFS": _get(mix, "GGBFS_pct", 0.0),
        "FlyAsh": _get(mix, "FlyAsh_pct", 0.0),
        "CaO": _get(mix, "CaO_pct", 0.0),
        "Gypsum": _get(mix, "Gypsum_pct", 0.0),
        "CaCl2": _get(mix, "CaCl2_pct", 0.0),
        "Limestone": _get(mix, "Limestone_pct", 0.0),
    }
    solid_v = sum(masses[k] / DENSITY[k] for k in masses)
    water_v = 100.0 * _get(mix, "WB", 0.38)  # rho water ~1 g/cm3
    return {"Solid_cm3": solid_v, "Water_cm3": water_v, "Paste_cm3": solid_v + water_v}


def microstructure_state(mix: Mapping[str, float], alpha: float, model: str = "CEMHYD3D") -> Dict[str, float]:
    v = initial_volumes(mix)
    wb = _get(mix, "WB", 0.38)
    slag = _get(mix, "GGBFS_pct", 75.0) / 100.0
    limestone = _get(mix, "Limestone_pct", 0.0) / 100.0
    alpha_norm = np.clip(alpha / max(reaction_capacity(mix), 1e-8), 0.0, 1.0)
    # Bound water and hydrate expansion priors for Ca-rich slag systems.
    bound_water = 100.0 * (0.19 * slag + 0.10 * (1 - slag)) * alpha_norm
    hydrate_v = v["Solid_cm3"] * alpha_norm * (1.18 + 0.28 * slag + 0.08 * limestone)
    capillary_v = max(0.0, v["Water_cm3"] - bound_water - 0.12 * hydrate_v)
    gel_pore_v = 0.25 * hydrate_v
    total = v["Solid_cm3"] * (1.0 - 0.45 * alpha_norm) + hydrate_v + capillary_v + gel_pore_v
    cap_por = capillary_v / max(total, 1e-9)
    gel_por = gel_pore_v / max(total, 1e-9)
    hydrate_frac = hydrate_v / max(total, 1e-9)
    critical = 0.23 if model.upper().startswith("CEM") else 0.20
    connectivity = 1.0 / (1.0 + math.exp(-18.0 * (hydrate_frac - critical)))
    gel_space = hydrate_v / max(hydrate_v + capillary_v + gel_pore_v, 1e-9)
    strength = (245.0 + 55.0 * slag) * gel_space ** 3.0 * (0.65 + 0.35 * connectivity)
    strength *= np.clip(1.0 - 0.75 * max(0.0, wb - 0.42), 0.55, 1.05)
    return {
        "CapillaryPorosity_frac": float(np.clip(cap_por, 0.0, 0.80)),
        "GelPorosity_frac": float(np.clip(gel_por, 0.0, 0.50)),
        "HydrateVolume_frac": float(np.clip(hydrate_frac, 0.0, 0.85)),
        "ConnectivityIndex": float(np.clip(connectivity, 0.0, 1.0)),
        "GelSpaceRatio": float(np.clip(gel_space, 0.0, 1.0)),
        "StrengthProxy_MPa": float(np.clip(strength, 0.0, 180.0)),
    }


def cemhyd3d_inspired(mix: Mapping[str, float], times_h: Iterable[float]) -> pd.DataFrame:
    kd = krstulovic_dabic(mix, times_h)
    rows = []
    for _, r in kd.iterrows():
        s = microstructure_state(mix, float(r["Alpha"]), model="CEMHYD3D")
        rows.append({"Time_h": r["Time_h"], "Alpha": r["Alpha"], **s})
    return pd.DataFrame(rows)


def voxel_slice(mix: Mapping[str, float], alpha: float, size: int = 120, seed: int = 22) -> np.ndarray:
    """Create a representative 2-D phase slice: 0 pore, 1 precursor, 2 hydrate, 3 filler."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    wb = _get(mix, "WB", 0.38)
    solid_target = np.clip(0.62 - 0.50 * (wb - 0.30), 0.36, 0.66)
    limestone_fraction = _get(mix, "Limestone_pct", 0.0) / 100.0
    alpha_norm = np.clip(alpha / max(reaction_capacity(mix), 1e-8), 0.0, 1.0)
    d50 = _get(mix, "D50_um", 12.0)
    mean_r = np.clip(2.5 + 0.18 * d50, 3.0, 7.0)
    occupied = np.zeros_like(grid, dtype=bool)
    particles = []
    attempts = 0
    while occupied.mean() < solid_target and attempts < 2500:
        attempts += 1
        r = float(np.clip(rng.lognormal(np.log(mean_r), 0.38), 1.4, 13.0))
        cx, cy = rng.uniform(r, size-r, 2)
        core = (xx-cx)**2 + (yy-cy)**2 <= r**2
        if (occupied & core).mean() > 0.001:
            continue
        occupied |= core
        particles.append((cx, cy, r, rng.random() < limestone_fraction * 2.5))
    shell_factor = 0.15 + 0.85 * alpha_norm
    for cx, cy, r, filler in particles:
        dist2 = (xx-cx)**2 + (yy-cy)**2
        shell_r = r * (1.0 + 0.55 * shell_factor)
        shell = dist2 <= shell_r**2
        core_r = r * (1.0 - 0.48 * alpha_norm)
        core = dist2 <= core_r**2
        grid[shell] = np.where(grid[shell] == 0, 2, grid[shell])
        grid[core] = 3 if filler else 1
    return grid


def _particle_cloud(mix: Mapping[str, float], n_particles: int = 160, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d50 = _get(mix, "D50_um", 12.0)
    diam = np.clip(rng.lognormal(np.log(d50), 0.50, n_particles), 1.0, 70.0)
    rad = diam / (diam.max() * 8.5)
    return pd.DataFrame({
        "x": rng.random(n_particles), "y": rng.random(n_particles), "z": rng.random(n_particles),
        "radius": rad, "diameter_um": diam,
    })


def hymostruc3d_inspired(mix: Mapping[str, float], times_h: Iterable[float], seed: int = 11) -> pd.DataFrame:
    kd = krstulovic_dabic(mix, times_h)
    particles = _particle_cloud(mix, seed=seed)
    xyz = particles[["x", "y", "z"]].to_numpy()
    r0 = particles["radius"].to_numpy()
    n = len(r0)
    dx = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(dx * dx, axis=2))
    np.fill_diagonal(dist, np.inf)
    rows = []
    slag = _get(mix, "GGBFS_pct", 75.0) / 100.0
    for _, rr in kd.iterrows():
        alpha = float(rr["Alpha"])
        an = np.clip(alpha / max(reaction_capacity(mix), 1e-8), 0.0, 1.0)
        shell = r0 * (1.0 + (0.38 + 0.18 * slag) * an ** 0.72)
        contacts = dist <= (shell[:, None] + shell[None, :])
        degrees = contacts.sum(axis=1)
        mean_degree = degrees.mean()
        active = (degrees >= 2).mean()
        percolation = 1.0 / (1.0 + np.exp(-1.35 * (mean_degree - 2.1)))
        micro = microstructure_state(mix, alpha, model="HYMOSTRUC")
        strength = micro["StrengthProxy_MPa"] * (0.55 + 0.45 * percolation)
        rows.append({
            "Time_h": rr["Time_h"], "Alpha": alpha,
            "MeanContactNumber": float(mean_degree),
            "ActiveParticleFraction": float(active),
            "PercolationIndex": float(percolation),
            "CapillaryPorosity_frac": micro["CapillaryPorosity_frac"],
            "StrengthProxy_MPa": float(np.clip(strength, 0.0, 180.0)),
        })
    return pd.DataFrame(rows)


def particle_snapshot(mix: Mapping[str, float], alpha: float, seed: int = 11) -> pd.DataFrame:
    p = _particle_cloud(mix, seed=seed)
    an = np.clip(alpha / max(reaction_capacity(mix), 1e-8), 0.0, 1.0)
    slag = _get(mix, "GGBFS_pct", 75.0) / 100.0
    p["shell_radius"] = p["radius"] * (1.0 + (0.38 + 0.18 * slag) * an ** 0.72)
    p["shell_diameter_plot"] = 800.0 * p["shell_radius"]
    return p


def hybrid_model(mix: Mapping[str, float], times_h: Iterable[float]) -> pd.DataFrame:
    kd = krstulovic_dabic(mix, times_h)
    phases = gems_phase_preview(mix, times_h)
    cem = cemhyd3d_inspired(mix, times_h)
    hym = hymostruc3d_inspired(mix, times_h)
    out = kd.merge(phases.drop(columns=["Alpha"]), on="Time_h")
    out = out.merge(cem.drop(columns=["Alpha"]), on="Time_h", suffixes=("", "_CEM"))
    out = out.merge(hym.drop(columns=["Alpha"]), on="Time_h", suffixes=("", "_HYM"))
    # Hybrid strength weighs gel-space and contact models; phase penalty for residual sulfate.
    residual_penalty = np.clip(1.0 - 0.025 * out["ResidualGypsum_pct"], 0.65, 1.0)
    phase_bonus = np.clip(0.85 + 0.004 * out["CASH_CSH_pct"] + 0.003 * out["Ettringite_pct"], 0.85, 1.18)
    out["HybridStrength_MPa"] = np.clip(
        (0.60 * out["StrengthProxy_MPa"] + 0.40 * out["StrengthProxy_MPa_HYM"])
        * residual_penalty * phase_bonus, 0.0, 180.0
    )
    return out


def run_model(model_name: str, mix: Mapping[str, float], times_h: Iterable[float]) -> pd.DataFrame:
    if model_name == "Krstulovic-Dabic":
        return krstulovic_dabic(mix, times_h)
    if model_name == "GEMS reduced-order preview":
        return gems_phase_preview(mix, times_h)
    if model_name == "CEMHYD3D-inspired":
        return cemhyd3d_inspired(mix, times_h)
    if model_name == "HYMOSTRUC3D-inspired":
        return hymostruc3d_inspired(mix, times_h)
    if model_name == "OODA-MAT hybrid":
        return hybrid_model(mix, times_h)
    raise ValueError(f"Unknown model: {model_name}")


def summary_at_ages(mix: Mapping[str, float], ages_h: Iterable[float] = (24, 72, 168, 672)) -> pd.DataFrame:
    times = np.unique(np.asarray(list(ages_h), dtype=float))
    kd = krstulovic_dabic(mix, times)
    gems = gems_phase_preview(mix, times)
    cem = cemhyd3d_inspired(mix, times)
    hym = hymostruc3d_inspired(mix, times)
    hyb = hybrid_model(mix, times)
    rows = []
    for i, age in enumerate(times):
        rows.append({
            "Age_h": age,
            "Alpha_KD": kd.iloc[i]["Alpha"],
            "Heat_J_g": kd.iloc[i]["CumulativeHeat_J_g"],
            "CASH_CSH_pct": gems.iloc[i]["CASH_CSH_pct"],
            "Ettringite_pct": gems.iloc[i]["Ettringite_pct"],
            "Hydrotalcite_pct": gems.iloc[i]["Hydrotalcite_pct"],
            "CapillaryPorosity_CEM": cem.iloc[i]["CapillaryPorosity_frac"],
            "Connectivity_CEM": cem.iloc[i]["ConnectivityIndex"],
            "Percolation_HYM": hym.iloc[i]["PercolationIndex"],
            "Strength_CEM_MPa": cem.iloc[i]["StrengthProxy_MPa"],
            "Strength_HYM_MPa": hym.iloc[i]["StrengthProxy_MPa"],
            "Strength_Hybrid_MPa": hyb.iloc[i]["HybridStrength_MPa"],
        })
    return pd.DataFrame(rows)


def compare_alternatives(alternatives: pd.DataFrame | None = None, ages_h: Iterable[float] = (24, 72, 168, 672)) -> pd.DataFrame:
    alternatives = EXAMPLE_ALTERNATIVES if alternatives is None else alternatives
    frames = []
    for _, row in alternatives.iterrows():
        s = summary_at_ages(row.to_dict(), ages_h)
        s.insert(0, "Alternative", row.get("Alternative", "Alternative"))
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def model_report_record(mix: Mapping[str, float], model_name: str, age_h: float = 672.0) -> Dict:
    df = run_model(model_name, mix, [age_h])
    return {
        "model": model_name,
        "age_h": age_h,
        "mix": {k: _get(mix, k, 0.0) for k in MIX_COLUMNS},
        "results": {k: (float(v) if isinstance(v, (np.floating, float, int)) else str(v)) for k, v in df.iloc[-1].to_dict().items()},
        "model_info": MODEL_INFO[model_name],
        "warnings": validate_mix(mix),
    }

@dataclass
class Constraints:
    min_strength3d: float = 30.0
    max_cost: float = 75000.0
    max_absorption: float = 3.5
    max_co2: float = 140.0
    max_expansion: float = 0.10
    max_cao: float = 6.0
    max_cacl2: float = 3.0
    wb_min: float = 0.30
    wb_max: float = 0.42


def synthetic_data(n: int = 42, seed: int = 18) -> pd.DataFrame:
    """Create a physically plausible demonstration dataset, not literature data."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fly = rng.uniform(5, 18)
        cao = rng.uniform(1, 6)
        gypsum = rng.uniform(3, 10)
        cacl2 = rng.uniform(0, 3)
        limestone = rng.uniform(0, 5)
        ggbfs = 100 - fly - cao - gypsum - cacl2 - limestone
        if ggbfs < 60:
            ggbfs = 60
            fly = 100 - ggbfs - cao - gypsum - cacl2 - limestone
        wb = rng.uniform(0.32, 0.42)
        temp = rng.choice([20, 25, 30, 40])

        # Synthetic response surface includes optimum regions and penalties.
        s3 = (
            10 + 0.23 * ggbfs + 1.8 * cao + 0.8 * gypsum + 1.3 * cacl2
            - 95 * (wb - 0.35)
            + 0.18 * (temp - 25)
            - 0.28 * (cao - 4.2) ** 2
            - 0.13 * (gypsum - 6.5) ** 2
            - 0.8 * max(0, cacl2 - 2.4) ** 2
            + rng.normal(0, 2.0)
        )
        s28 = s3 + 10 + 0.25 * fly - 0.5 * cacl2 + rng.normal(0, 2.2)
        absorption = (
            2.0 + 12 * (wb - 0.32) + 0.035 * fly - 0.04 * limestone
            + rng.normal(0, 0.22)
        )
        cost = (
            ggbfs * 58 + fly * 35 + cao * 160 + gypsum * 55 +
            cacl2 * 420 + limestone * 30
        ) * 10
        co2 = (
            ggbfs * 0.07 + fly * 0.02 + cao * 1.05 + gypsum * 0.08 +
            cacl2 * 0.85 + limestone * 0.06
        ) * 10
        expansion = max(
            0.0, 0.015 + 0.010 * max(0, cao - 3.5)
            + 0.005 * max(0, gypsum - 7)
            + rng.normal(0, 0.009)
        )

        # Synthetic chemical-analysis descriptors for software demonstration only.
        reaction = np.clip(0.35 + 0.045 * cao + 0.025 * gypsum + 0.018 * cacl2
                           - 1.7 * (wb - 0.35) + 0.006 * (temp - 25), 0.15, 0.90)
        xrd_cash = np.clip(8 + 0.20 * ggbfs * reaction + 0.10 * fly * reaction + rng.normal(0, 1.5), 3, 45)
        xrd_ett = np.clip(0.35 * gypsum * reaction + 0.20 * cao + rng.normal(0, 0.5), 0, 12)
        xrd_ht = np.clip(0.035 * ggbfs * reaction + rng.normal(0, 0.35), 0, 8)
        xrd_calcite = np.clip(0.75 * limestone + 0.30 * cao * (1-reaction) + rng.normal(0, 0.6), 0, 12)
        xrd_gypsum = np.clip(gypsum * (1-reaction) * 0.75 + rng.normal(0, 0.4), 0, 10)
        xrd_amorphous = np.clip(72 - 0.35*xrd_cash - 0.55*xrd_ett - 0.30*xrd_ht
                                - 0.45*xrd_calcite - 0.55*xrd_gypsum + rng.normal(0, 2.0), 25, 85)

        tg_30_200 = np.clip(2.0 + 0.10*xrd_cash + 0.22*xrd_ett + rng.normal(0, 0.35), 1, 12)
        tg_200_400 = np.clip(0.4 + 0.10*xrd_ht + 0.025*xrd_cash + rng.normal(0, 0.15), 0.1, 4)
        tg_400_550 = np.clip(0.15 + 0.03*cao*(1-reaction) + rng.normal(0, 0.08), 0, 2)
        tg_550_800 = np.clip(0.15 + 0.42*xrd_calcite + rng.normal(0, 0.25), 0, 8)
        tg_total = tg_30_200 + tg_200_400 + tg_400_550 + tg_550_800

        dtg_p1_t = np.clip(92 + 0.8*xrd_ett + rng.normal(0, 5), 65, 150)
        dtg_p1_i = np.clip(0.035*tg_30_200 + rng.normal(0, 0.025), 0.02, 0.8)
        dtg_p2_t = np.clip(245 + 5*xrd_ht + rng.normal(0, 12), 180, 360)
        dtg_p2_i = np.clip(0.055*tg_200_400 + rng.normal(0, 0.012), 0.005, 0.3)
        dtg_p3_t = np.clip(675 + 2.5*xrd_calcite + rng.normal(0, 18), 550, 800)
        dtg_p3_i = np.clip(0.040*tg_550_800 + rng.normal(0, 0.012), 0.005, 0.35)

        rows.append({
            "ExperimentID": f"DEMO-{i+1:03d}",
            "GGBFS_pct": ggbfs, "FlyAsh_pct": fly, "CaO_pct": cao,
            "Gypsum_pct": gypsum, "CaCl2_pct": cacl2,
            "Limestone_pct": limestone, "WB": wb,
            "CuringTemp_C": temp, "Strength3d_MPa": max(5, s3),
            "Strength28d_MPa": max(10, s28),
            "Absorption_pct": max(0.5, absorption),
            "Cost_KRW_t": cost, "CO2_kg_t": co2,
            "Expansion_pct": expansion,
            "XRD_CASH_CSH_pct": xrd_cash,
            "XRD_Ettringite_pct": xrd_ett,
            "XRD_Hydrotalcite_pct": xrd_ht,
            "XRD_Calcite_pct": xrd_calcite,
            "XRD_ResidualGypsum_pct": xrd_gypsum,
            "XRD_Amorphous_pct": xrd_amorphous,
            "TG_Loss_30_200_pct": tg_30_200,
            "TG_Loss_200_400_pct": tg_200_400,
            "TG_Loss_400_550_pct": tg_400_550,
            "TG_Loss_550_800_pct": tg_550_800,
            "TG_TotalLoss_pct": tg_total,
            "DTG_Peak1_T_C": dtg_p1_t,
            "DTG_Peak1_Intensity_pct_min": dtg_p1_i,
            "DTG_Peak2_T_C": dtg_p2_t,
            "DTG_Peak2_Intensity_pct_min": dtg_p2_i,
            "DTG_Peak3_T_C": dtg_p3_t,
            "DTG_Peak3_Intensity_pct_min": dtg_p3_i,
            "AnalysisAge_d": 3,
            "XRD_Method": "Synthetic demonstration; replace with verified Rietveld data",
            "TG_Atmosphere": "Synthetic demonstration; define N2 or air",
            "HeatingRate_C_min": 10.0,
            "Source": "Synthetic demonstration data"
        })
    return pd.DataFrame(rows)


def ensure_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    out = df.copy()
    if "ExperimentID" not in out.columns:
        out.insert(0, "ExperimentID", [f"EXP-{i+1:03d}" for i in range(len(out))])
    return out, missing


def input_group_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return only the columns belonging to one CSV input/output group."""
    return df.reindex(columns=INPUT_GROUP_COLUMNS[group]).copy()


def show_dataframe(data, *args, **kwargs):
    """Display floating-point outputs with two decimal places without changing calculations."""
    if isinstance(data, pd.DataFrame):
        float_columns = data.select_dtypes(include=["floating"]).columns
        if len(float_columns):
            formats = {column: "{:.2f}" for column in float_columns}
            data = data.style.format(formats, na_rep="")
    return st.dataframe(data, *args, **kwargs)


def merge_analysis_input(base: pd.DataFrame, analysis: pd.DataFrame, group: str) -> pd.DataFrame:
    """Merge one optional analysis table into the general table by ExperimentID."""
    if "ExperimentID" not in analysis.columns:
        raise ValueError(f"{group} CSV requires an ExperimentID column.")
    if analysis["ExperimentID"].isna().any() or analysis["ExperimentID"].duplicated().any():
        raise ValueError(f"{group} CSV ExperimentID values must be non-empty and unique.")
    unknown_ids = sorted(set(analysis["ExperimentID"]) - set(base["ExperimentID"]))
    if unknown_ids:
        preview = ", ".join(map(str, unknown_ids[:5]))
        raise ValueError(f"{group} CSV contains ExperimentID values absent from General CSV: {preview}")

    allowed = set(INPUT_GROUP_COLUMNS[group])
    selected = [col for col in analysis.columns if col in allowed]
    analysis = analysis.loc[:, selected].copy()
    overlapping = [col for col in analysis.columns if col != "ExperimentID" and col in base.columns]
    if overlapping:
        base = base.drop(columns=overlapping)
    return base.merge(analysis, on="ExperimentID", how="left", validate="one_to_one")


def load_split_input_tables(general_source, analysis_sources: Mapping[str, object]):
    """Read, validate and join the four separated CSV inputs by ExperimentID."""
    general = pd.read_csv(general_source, encoding="utf-8-sig")
    general, missing = ensure_columns(general)
    if missing:
        raise ValueError("Missing general-input columns: " + ", ".join(missing))
    if general["ExperimentID"].isna().any() or general["ExperimentID"].duplicated().any():
        raise ValueError("General CSV ExperimentID values must be non-empty and unique.")
    general["ExperimentID"] = general["ExperimentID"].astype(str).str.strip()
    if general["ExperimentID"].eq("").any() or general["ExperimentID"].duplicated().any():
        raise ValueError("General CSV ExperimentID values must be non-empty and unique.")

    merged = input_group_table(general, "General")
    report = [{"Input": "General", "Rows": len(general), "Matched_rows": len(general)}]
    for group in ("XRD", "TG", "DTG"):
        source = analysis_sources.get(group)
        if source is None:
            report.append({"Input": group, "Rows": 0, "Matched_rows": 0})
            continue
        analysis = pd.read_csv(source, encoding="utf-8-sig")
        if "ExperimentID" in analysis.columns:
            analysis["ExperimentID"] = analysis["ExperimentID"].astype(str).str.strip()
        merged = merge_analysis_input(merged, analysis, group)
        matched = analysis["ExperimentID"].isin(general["ExperimentID"]).sum()
        report.append({"Input": group, "Rows": len(analysis), "Matched_rows": int(matched)})
    return merged, pd.DataFrame(report)


def composition_error(df: pd.DataFrame) -> pd.Series:
    cols = ["GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct", "CaCl2_pct", "Limestone_pct"]
    return (df[cols].sum(axis=1) - 100.0).abs()


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    eps = 1e-6
    # Proxy engineering descriptors. Replace with oxide-based molar ratios when XRF data are available.
    x["Ca_source_index"] = x["GGBFS_pct"] * 0.40 + x["CaO_pct"] + x["Gypsum_pct"] * 0.33
    x["Al_source_index"] = x["GGBFS_pct"] * 0.13 + x["FlyAsh_pct"] * 0.25
    x["Sulfate_Al_proxy"] = (x["Gypsum_pct"] * 0.46) / (x["Al_source_index"] + eps)
    x["Activator_index"] = x["CaO_pct"] + 0.8 * x["CaCl2_pct"] + 0.25 * x["Gypsum_pct"]
    x["Water_solid_index"] = x["WB"] * (1 + 0.008 * x["FlyAsh_pct"])
    x["Maturity_proxy"] = (x["CuringTemp_C"] + 10.0) / 35.0
    return x


def make_gp() -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=True,
        random_state=42
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gp", gp)
    ])


@st.cache_resource(show_spinner="Training Gaussian-process models...")
def fit_models(df: pd.DataFrame) -> Dict[str, Pipeline]:
    x = engineered_features(df)
    models = {}
    for target in TARGETS:
        if target not in df.columns:
            continue
        valid = df[target].notna()
        if valid.sum() >= 8:
            model = make_gp()
            model.fit(x.loc[valid], df.loc[valid, target])
            models[target] = model
    return models


def predict_with_std(model: Pipeline, x: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    xp = model.named_steps["imputer"].transform(x)
    xp = model.named_steps["scaler"].transform(xp)
    return model.named_steps["gp"].predict(xp, return_std=True)


@st.cache_data(show_spinner="Calculating model diagnostics...")
def model_diagnostics(df: pd.DataFrame, _models: Dict[str, Pipeline]) -> pd.DataFrame:
    x = engineered_features(df)
    records = []
    for target, model in _models.items():
        valid = df[target].notna()
        y = df.loc[valid, target].values
        xv = x.loc[valid]
        if len(y) < 10:
            continue
        loo = LeaveOneOut()
        try:
            # Reuse the fitted hyperparameters during cross-validation. Re-running
            # kernel optimization for every LOO fold makes a Streamlit rerun take
            # several minutes when many chemical targets are present.
            fitted_gp = model.named_steps["gp"]
            diagnostic_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("gp", GaussianProcessRegressor(
                    kernel=fitted_gp.kernel_,
                    optimizer=None,
                    normalize_y=True,
                    random_state=42,
                )),
            ])
            pred = cross_val_predict(diagnostic_model, xv, y, cv=loo, n_jobs=None)
            records.append({
                "Target": target,
                "N": len(y),
                "MAE": mean_absolute_error(y, pred),
                "R2_LOO": r2_score(y, pred)
            })
        except Exception:
            pass
    return pd.DataFrame(records)


def hydration_model_catalog(models: Dict[str, Pipeline]) -> pd.DataFrame:
    """Summarize documentation and runtime availability for each model family."""
    records = []
    for name, info in HYDRATION_MODELS.items():
        trained = [target for target in info["targets"] if target in models]
        records.append({
            "Model": name,
            "Trained_targets": len(trained),
            "Expected_targets": len(info["targets"]),
            "Available_outputs": ", ".join(trained) if trained else "None",
        })
    return pd.DataFrame(records)



SENSITIVITY_BALANCE_OPTIONS = [
    "Proportional closure",
    "Paired precursor replacement",
]

HYDRATION_SENSITIVITY_INPUTS = [
    *FEATURES,
    "Blaine_m2kg",
    "D50_um",
    "ReactiveSlag_frac",
    "ReactiveFlyAsh_frac",
]


def _closed_reference_profile(df: pd.DataFrame, reference_id: str) -> pd.Series:
    """Return a complete reference profile with binder fractions closed to 100 wt.%."""
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    matches = df["ExperimentID"].astype(str) == str(reference_id)
    if matches.any():
        reference = numeric.loc[matches].iloc[0].fillna(medians).copy()
    else:
        reference = medians.copy()

    binder = reference[BINDER_COMPONENTS].clip(lower=0.0)
    total = float(binder.sum())
    if not np.isfinite(total) or total <= 1e-12:
        binder[:] = [75.0, 10.0, 4.0, 6.0, 1.5, 3.5]
        total = float(binder.sum())
    reference.loc[BINDER_COMPONENTS] = 100.0 * binder / total
    return reference.astype(float)


def _apply_composition_closure(
    reference: pd.Series,
    feature: str,
    value: float,
    balance_mode: str,
) -> pd.Series | None:
    """Perturb one variable while preserving a non-negative 100 wt.% binder total."""
    trial = reference.astype(float).copy()
    value = float(value)

    if feature not in BINDER_COMPONENTS:
        trial[feature] = value
        return trial

    if value < 0.0 or value > 100.0:
        return None

    trial[feature] = value
    other_components = [item for item in BINDER_COMPONENTS if item != feature]

    if balance_mode == "Paired precursor replacement":
        balance_feature = "FlyAsh_pct" if feature == "GGBFS_pct" else "GGBFS_pct"
        delta = value - float(reference[feature])
        trial[balance_feature] = float(reference[balance_feature]) - delta
        if trial[balance_feature] < -1e-9:
            return None
        trial[balance_feature] = max(0.0, float(trial[balance_feature]))
        # Numerical closure is applied to the selected balance phase.
        closure_error = 100.0 - float(trial[BINDER_COMPONENTS].sum())
        trial[balance_feature] += closure_error
    else:
        remaining_mass = 100.0 - value
        reference_others = reference[other_components].clip(lower=0.0)
        other_total = float(reference_others.sum())
        if remaining_mass < -1e-9 or other_total <= 1e-12:
            return None
        trial.loc[other_components] = reference_others * remaining_mass / other_total

    if (trial[BINDER_COMPONENTS] < -1e-8).any():
        return None
    trial.loc[BINDER_COMPONENTS] = trial[BINDER_COMPONENTS].clip(lower=0.0)
    if abs(float(trial[BINDER_COMPONENTS].sum()) - 100.0) > 1e-6:
        return None
    return trial


def _applicability_distance(
    training: pd.DataFrame,
    query: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """Return standardized nearest-neighbour distance and a 95% training threshold."""
    train = training[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = train.median()
    train = train.fillna(medians)
    query = query[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians)
    scale = train.std(ddof=0).replace(0.0, 1.0)
    train_z = ((train - train.mean()) / scale).to_numpy(dtype=float)
    query_z = ((query - train.mean()) / scale).to_numpy(dtype=float)

    if len(train_z) < 2:
        return np.zeros(len(query_z)), float("inf")

    train_dist = np.sqrt(
        np.sum((train_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(train_dist, np.inf)
    nn_train = np.min(train_dist, axis=1)
    threshold = float(np.quantile(nn_train[np.isfinite(nn_train)], 0.95))

    query_dist = np.sqrt(
        np.sum((query_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2)
    )
    return np.min(query_dist, axis=1), threshold


@st.cache_data(show_spinner="Calculating constrained local sensitivity...")
def sensitivity_analysis(
    df: pd.DataFrame,
    _model: Pipeline,
    target: str,
    reference_id: str,
    grid_points: int = 21,
    balance_mode: str = "Proportional closure",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constraint-aware local OAT analysis with uncertainty and domain diagnostics.

    Binder perturbations preserve a 100 wt.% dry-binder total. The result reports
    range effect, local standardized slope, nonlinearity, monotonicity, predictive
    uncertainty and the fraction of the sweep outside the empirical applicability
    domain. It is a model-response diagnostic, not a causal attribution.
    """
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    reference = _closed_reference_profile(df, reference_id)
    reference_frame = pd.DataFrame([reference], columns=FEATURES)
    baseline, baseline_std = predict_with_std(
        _model, engineered_features(reference_frame)
    )
    baseline_prediction = float(baseline[0])
    baseline_uncertainty = float(baseline_std[0])

    target_values = pd.to_numeric(df[target], errors="coerce")
    target_scale = float(target_values.std())
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        target_scale = max(abs(baseline_prediction), 1.0)

    detail_records = []
    summary_records = []

    for feature in FEATURES:
        valid = numeric[feature].dropna()
        if valid.empty:
            continue

        low, high = valid.quantile([0.05, 0.95]).astype(float)
        if feature in BINDER_COMPONENTS:
            low = max(0.0, low)
            high = min(100.0, high)
        if not np.isfinite(low) or not np.isfinite(high) or math.isclose(low, high):
            continue

        rows = []
        accepted_values = []
        for value in np.linspace(low, high, grid_points):
            trial = _apply_composition_closure(
                reference, feature, float(value), balance_mode
            )
            if trial is None:
                continue
            rows.append(trial)
            accepted_values.append(float(value))

        if len(rows) < 3:
            continue

        sweep = pd.DataFrame(rows, columns=FEATURES)
        prediction, prediction_std = predict_with_std(
            _model, engineered_features(sweep)
        )
        domain_distance, domain_threshold = _applicability_distance(df, sweep)
        outside = domain_distance > domain_threshold
        values = np.asarray(accepted_values, dtype=float)
        slopes = np.gradient(prediction, values)
        reference_index = int(np.argmin(np.abs(values - float(reference[feature]))))
        local_slope = float(slopes[reference_index])
        input_scale = float(valid.std())
        if not np.isfinite(input_scale) or input_scale <= 1e-12:
            input_scale = max(abs(float(reference[feature])), 1.0)

        linear_response = np.interp(
            values,
            [values[0], values[-1]],
            [prediction[0], prediction[-1]],
        )
        nonlinearity = float(
            np.sqrt(np.mean((prediction - linear_response) ** 2)) / target_scale
        )
        differences = np.diff(prediction)
        monotonic_fraction = float(max(
            np.mean(differences >= -1e-10),
            np.mean(differences <= 1e-10),
        ))
        response_range = float(np.max(prediction) - np.min(prediction))
        signed_change = float(prediction[-1] - prediction[0])
        normalized_slope = local_slope * input_scale / target_scale
        uncertainty_index = float(np.mean(prediction_std) / target_scale)
        outside_fraction = float(np.mean(outside))

        if monotonic_fraction < 0.80:
            direction = "Non-monotonic"
        elif signed_change > 1e-9:
            direction = "Increasing"
        elif signed_change < -1e-9:
            direction = "Decreasing"
        else:
            direction = "Flat"

        if outside_fraction <= 0.10 and uncertainty_index <= 0.20:
            confidence = "Higher"
        elif outside_fraction <= 0.35 and uncertainty_index <= 0.40:
            confidence = "Moderate"
        else:
            confidence = "Low"

        for position, value, pred, std, distance, is_outside in zip(
            np.linspace(0.0, 100.0, len(sweep)),
            values,
            prediction,
            prediction_std,
            domain_distance,
            outside,
        ):
            detail_records.append({
                "Target": target,
                "Variable": feature,
                "Range_position_pct": float(position),
                "Variable_value": float(value),
                "Prediction": float(pred),
                "Prediction_std": float(std),
                "Change_from_reference": float(pred - baseline_prediction),
                "Applicability_distance": float(distance),
                "Applicability_threshold": float(domain_threshold),
                "Outside_domain": bool(is_outside),
                "Balance_rule": balance_mode,
            })

        summary_records.append({
            "Variable": feature,
            "Low_5pct": float(values[0]),
            "Reference": float(reference[feature]),
            "High_95pct": float(values[-1]),
            "Reference_prediction": baseline_prediction,
            "Reference_prediction_std": baseline_uncertainty,
            "Prediction_at_low": float(prediction[0]),
            "Prediction_at_high": float(prediction[-1]),
            "Signed_change": signed_change,
            "OAT_range_effect": response_range / target_scale,
            "Local_standardized_slope": normalized_slope,
            "Nonlinearity_index": nonlinearity,
            "Monotonic_fraction": monotonic_fraction,
            "Mean_uncertainty_index": uncertainty_index,
            "Outside_domain_fraction": outside_fraction,
            "Direction": direction,
            "Confidence": confidence,
            "Balance_rule": balance_mode,
        })

    summary = pd.DataFrame(summary_records)
    if not summary.empty:
        summary = summary.sort_values(
            ["OAT_range_effect", "Mean_uncertainty_index"],
            ascending=[False, True],
        )
    return summary.reset_index(drop=True), pd.DataFrame(detail_records)


@st.cache_data(show_spinner="Calculating constraint-aware global sensitivity...")
def global_sensitivity_analysis(
    df: pd.DataFrame,
    _model: Pipeline,
    target: str,
    n_samples: int = 800,
    seed: int = 42,
    balance_mode: str = "Proportional closure",
) -> pd.DataFrame:
    """Global permutation response analysis on feasible, approximately in-domain mixtures.

    This is not a Sobol decomposition. A variable is permuted across feasible candidate
    mixtures, binder closure is restored, and the mean change in model prediction is
    normalized by the measured target standard deviation.
    """
    rng = np.random.default_rng(seed)
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bounds = numeric.quantile([0.05, 0.95])

    pool = candidate_space(max(4000, n_samples * 8), seed=seed)
    for feature in FEATURES:
        low, high = bounds.loc[0.05, feature], bounds.loc[0.95, feature]
        if np.isfinite(low) and np.isfinite(high) and low < high:
            pool = pool[pool[feature].between(low, high)]
    if len(pool) < max(100, n_samples // 3):
        # Fall back to bootstrapped observed compositions if the candidate filter is tight.
        base = numeric.fillna(numeric.median()).sample(
            n=n_samples, replace=True, random_state=seed
        ).reset_index(drop=True)
        binder_sum = base[BINDER_COMPONENTS].sum(axis=1).replace(0.0, np.nan)
        base.loc[:, BINDER_COMPONENTS] = (
            base[BINDER_COMPONENTS].div(binder_sum, axis=0) * 100.0
        )
    else:
        base = pool.sample(
            n=min(n_samples, len(pool)), random_state=seed
        ).reset_index(drop=True)

    base_prediction, base_std = predict_with_std(
        _model, engineered_features(base)
    )
    target_scale = float(pd.to_numeric(df[target], errors="coerce").std())
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        target_scale = max(float(np.std(base_prediction)), 1.0)

    records = []
    for feature in FEATURES:
        permuted_values = rng.permutation(base[feature].to_numpy())
        perturbed_rows = []
        valid_indices = []

        for idx, value in enumerate(permuted_values):
            reference = base.iloc[idx].copy()
            trial = _apply_composition_closure(
                reference, feature, float(value), balance_mode
            )
            if trial is not None:
                perturbed_rows.append(trial)
                valid_indices.append(idx)

        if len(perturbed_rows) < 20:
            continue

        perturbed = pd.DataFrame(perturbed_rows, columns=FEATURES)
        perturbed_prediction, perturbed_std = predict_with_std(
            _model, engineered_features(perturbed)
        )
        original_prediction = base_prediction[np.asarray(valid_indices)]
        delta = perturbed_prediction - original_prediction
        rho = pd.Series(
            base.iloc[valid_indices][feature].to_numpy(dtype=float)
        ).corr(
            pd.Series(np.asarray(original_prediction, dtype=float)),
            method="spearman",
        )
        domain_distance, domain_threshold = _applicability_distance(df, perturbed)

        records.append({
            "Variable": feature,
            "Global_permutation_effect": float(np.mean(np.abs(delta)) / target_scale),
            "P95_prediction_change": float(np.quantile(np.abs(delta), 0.95) / target_scale),
            "Spearman_model_association": float(rho) if pd.notna(rho) else 0.0,
            "Mean_prediction_uncertainty": float(np.mean(perturbed_std) / target_scale),
            "Outside_domain_fraction": float(np.mean(domain_distance > domain_threshold)),
            "Valid_perturbation_fraction": float(len(valid_indices) / len(base)),
            "Balance_rule": balance_mode,
            "Method": "Constraint-aware global permutation; not Sobol",
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("Global_permutation_effect", ascending=False)
    return result.reset_index(drop=True)


def chemical_performance_linkage(df: pd.DataFrame, min_pairs: int = 8) -> pd.DataFrame:
    """Estimate screening-level chemical/performance associations in measured data."""
    records = []
    for chemical in CHEMICAL_TARGETS:
        if chemical not in df.columns:
            continue
        for performance in PERFORMANCE_TARGETS:
            if performance not in df.columns:
                continue
            paired = df[[chemical, performance]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) < min_pairs or paired[chemical].nunique() < 2 or paired[performance].nunique() < 2:
                continue
            rho = paired[chemical].corr(paired[performance], method="spearman")
            if pd.notna(rho):
                records.append({
                    "Chemical_state": chemical,
                    "Engineering_performance": performance,
                    "Spearman_rho": float(rho),
                    "Abs_rho": abs(float(rho)),
                    "Direction": "Positive" if rho >= 0 else "Negative",
                    "Paired_experiments": len(paired),
                })
    return pd.DataFrame(records).sort_values("Abs_rho", ascending=False).reset_index(drop=True) if records else pd.DataFrame()


def candidate_linkage_evidence(
    candidate: pd.Series, df: pd.DataFrame, linkage: pd.DataFrame
) -> pd.DataFrame:
    """Combine a candidate chemical deviation with measured-data associations."""
    if linkage.empty:
        return pd.DataFrame()
    rows = []
    for item in linkage.itertuples(index=False):
        pred_col = f"Pred_{item.Chemical_state}"
        if pred_col not in candidate.index or pd.isna(candidate.get(pred_col)):
            continue
        observed = pd.to_numeric(df[item.Chemical_state], errors="coerce").dropna()
        scale = observed.std()
        if observed.empty or pd.isna(scale) or scale <= 1e-12:
            continue
        chemical_z = (float(candidate[pred_col]) - float(observed.median())) / float(scale)
        directional_index = chemical_z * float(item.Spearman_rho)
        rows.append({
            "Chemical_state": item.Chemical_state,
            "Engineering_performance": item.Engineering_performance,
            "Candidate_chemical_prediction": float(candidate[pred_col]),
            "Chemical_deviation_z": chemical_z,
            "Measured_Spearman_rho": float(item.Spearman_rho),
            "Directional_link_index": directional_index,
            "Linked_direction_for_candidate": (
                "Higher than reference" if directional_index > 0
                else "Lower than reference" if directional_index < 0
                else "Neutral"
            ),
            "Paired_experiments": int(item.Paired_experiments),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Evidence_strength"] = result["Measured_Spearman_rho"].abs()
        result = result.sort_values(
            ["Evidence_strength", "Directional_link_index"], ascending=[False, False]
        ).reset_index(drop=True)
    return result


def _hydration_variable_bounds(mix: Mapping[str, float], variable: str) -> Tuple[float, float]:
    value = _get(mix, variable, {
        "WB": 0.38,
        "CuringTemp_C": 25.0,
        "Blaine_m2kg": 450.0,
        "D50_um": 12.0,
        "ReactiveSlag_frac": 0.76,
        "ReactiveFlyAsh_frac": 0.32,
    }.get(variable, 0.0))

    fixed_bounds = {
        "GGBFS_pct": (max(40.0, value - 15.0), min(92.0, value + 15.0)),
        "FlyAsh_pct": (max(0.0, value - 10.0), min(45.0, value + 10.0)),
        "CaO_pct": (max(0.0, value - 2.0), min(10.0, value + 2.0)),
        "Gypsum_pct": (max(0.0, value - 3.0), min(15.0, value + 3.0)),
        "CaCl2_pct": (max(0.0, value - 1.5), min(5.0, value + 1.5)),
        "Limestone_pct": (max(0.0, value - 5.0), min(15.0, value + 5.0)),
        "WB": (max(0.25, value - 0.05), min(0.55, value + 0.05)),
        "CuringTemp_C": (max(5.0, value - 10.0), min(60.0, value + 10.0)),
        "Blaine_m2kg": (max(150.0, value - 150.0), min(1200.0, value + 150.0)),
        "D50_um": (max(1.0, value - 8.0), min(80.0, value + 8.0)),
        "ReactiveSlag_frac": (max(0.10, value - 0.15), min(1.0, value + 0.15)),
        "ReactiveFlyAsh_frac": (max(0.02, value - 0.15), min(1.0, value + 0.15)),
    }
    return fixed_bounds[variable]


def _hydration_mechanistic_note(variable: str) -> str:
    notes = {
        "GGBFS_pct": "Changes reactive Ca-Si-Al-Mg precursor supply and usually C-(A)-S-H/hydrotalcite capacity.",
        "FlyAsh_pct": "May dilute early slag reaction but can supply later Si and Al when its reactive fraction is verified.",
        "CaO_pct": "Raises Ca availability and alkalinity proxy; may accelerate dissolution but increases free-lime/expansion risk.",
        "Gypsum_pct": "Controls sulfate availability, ettringite formation and residual-sulfate penalty.",
        "CaCl2_pct": "Accelerates early kinetics and ionic strength; chloride compatibility must be reviewed separately.",
        "Limestone_pct": "Adds filler/nucleation surface and carbonate for carboaluminate, while excessive replacement dilutes precursor.",
        "WB": "Improves transport at low values but increases capillary pore volume and lowers gel-space strength at high values.",
        "CuringTemp_C": "Acts through Arrhenius acceleration; a fitted activation energy is required for reliable extrapolation.",
        "Blaine_m2kg": "In the kinetic prior, greater fineness modestly accelerates dissolution and shortens diffusion distance. Any adverse water-demand, agglomeration or rheology effect must be introduced through measured flow/W/B calibration rather than assumed automatically.",
        "D50_um": "Controls representative particle radius, shell-growth distance, packing and contact development.",
        "ReactiveSlag_frac": "Directly controls maximum slag reaction capacity and therefore heat, hydrate volume and long-age strength.",
        "ReactiveFlyAsh_frac": "Controls the delayed fly-ash contribution to Si/Al release and long-age phase development.",
    }
    return notes.get(variable, "Review the variable against model assumptions and independent measurements.")


@st.cache_data(show_spinner="Calculating hydration-model sensitivity...")
def hydration_sensitivity_analysis(
    mix_json: str,
    model_name: str,
    output_name: str,
    age_h: float,
    balance_mode: str = "Proportional closure",
    grid_points: int = 9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constrained OAT sensitivity for the embedded hydration models."""
    mix = json.loads(mix_json)
    reference = pd.Series({
        key: _get(mix, key, 0.0) for key in HYDRATION_SENSITIVITY_INPUTS
    }, dtype=float)
    binder_total = float(reference[BINDER_COMPONENTS].sum())
    if binder_total > 1e-12:
        reference.loc[BINDER_COMPONENTS] = (
            100.0 * reference[BINDER_COMPONENTS] / binder_total
        )

    baseline_df = run_model(model_name, reference.to_dict(), [float(age_h)])
    if output_name not in baseline_df.columns:
        raise ValueError(f"{output_name} is not available for {model_name}.")
    baseline = float(baseline_df.iloc[-1][output_name])
    output_scale = max(abs(baseline), 1.0)

    summary_records = []
    detail_records = []
    for variable in HYDRATION_SENSITIVITY_INPUTS:
        low, high = _hydration_variable_bounds(reference.to_dict(), variable)
        if not np.isfinite(low) or not np.isfinite(high) or math.isclose(low, high):
            continue

        values = np.linspace(low, high, grid_points)
        accepted_values = []
        predictions = []

        for value in values:
            if variable in BINDER_COMPONENTS:
                trial = _apply_composition_closure(
                    reference, variable, float(value), balance_mode
                )
                if trial is None:
                    continue
            else:
                trial = reference.copy()
                trial[variable] = float(value)

            result = run_model(model_name, trial.to_dict(), [float(age_h)])
            value_out = result.iloc[-1][output_name]
            if pd.notna(value_out) and np.isfinite(float(value_out)):
                accepted_values.append(float(value))
                predictions.append(float(value_out))

        if len(predictions) < 3:
            continue

        x = np.asarray(accepted_values, dtype=float)
        y = np.asarray(predictions, dtype=float)
        slopes = np.gradient(y, x)
        ref_value = float(reference[variable])
        ref_index = int(np.argmin(np.abs(x - ref_value)))
        local_slope = float(slopes[ref_index])
        x_scale = max(float(x[-1] - x[0]), 1e-12)
        linear_response = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        nonlinearity = float(
            np.sqrt(np.mean((y - linear_response) ** 2)) / output_scale
        )
        differences = np.diff(y)
        monotonic_fraction = float(max(
            np.mean(differences >= -1e-10),
            np.mean(differences <= 1e-10),
        ))
        signed_change = float(y[-1] - y[0])
        if monotonic_fraction < 0.80:
            direction = "Non-monotonic"
        elif signed_change > 1e-9:
            direction = "Increasing"
        elif signed_change < -1e-9:
            direction = "Decreasing"
        else:
            direction = "Flat"

        for position, value, prediction in zip(
            np.linspace(0.0, 100.0, len(x)), x, y
        ):
            detail_records.append({
                "Model": model_name,
                "Output": output_name,
                "Age_h": float(age_h),
                "Variable": variable,
                "Range_position_pct": float(position),
                "Variable_value": float(value),
                "Prediction": float(prediction),
                "Change_from_reference": float(prediction - baseline),
                "Balance_rule": balance_mode,
            })

        summary_records.append({
            "Variable": variable,
            "Low": float(x[0]),
            "Reference": ref_value,
            "High": float(x[-1]),
            "Output_at_low": float(y[0]),
            "Baseline_output": baseline,
            "Output_at_high": float(y[-1]),
            "Relative_range_effect": float((np.max(y) - np.min(y)) / output_scale),
            "Local_normalized_slope": float(local_slope * x_scale / output_scale),
            "Nonlinearity_index": nonlinearity,
            "Monotonic_fraction": monotonic_fraction,
            "Direction": direction,
            "Mechanistic_interpretation": _hydration_mechanistic_note(variable),
            "Balance_rule": balance_mode,
        })

    summary = pd.DataFrame(summary_records)
    if not summary.empty:
        summary = summary.sort_values("Relative_range_effect", ascending=False)
    return summary.reset_index(drop=True), pd.DataFrame(detail_records)



def candidate_space(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    tries = 0
    while len(records) < n and tries < n * 20:
        tries += 1
        fly = rng.uniform(5, 20)
        cao = rng.uniform(1, 6)
        gypsum = rng.uniform(3, 10)
        cacl2 = rng.uniform(0, 3)
        limestone = rng.uniform(0, 5)
        ggbfs = 100 - fly - cao - gypsum - cacl2 - limestone
        if 60 <= ggbfs <= 88:
            records.append({
                "GGBFS_pct": ggbfs, "FlyAsh_pct": fly, "CaO_pct": cao,
                "Gypsum_pct": gypsum, "CaCl2_pct": cacl2,
                "Limestone_pct": limestone,
                "WB": rng.uniform(0.30, 0.42),
                "CuringTemp_C": rng.choice([20, 25, 30, 40])
            })
    return pd.DataFrame(records)


def estimate_cost_carbon(cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    # Editable default unit values represented indirectly in formulas.
    out["Cost_KRW_t_rule"] = (
        out["GGBFS_pct"] * 58 + out["FlyAsh_pct"] * 35 +
        out["CaO_pct"] * 160 + out["Gypsum_pct"] * 55 +
        out["CaCl2_pct"] * 420 + out["Limestone_pct"] * 30
    ) * 10
    out["CO2_kg_t_rule"] = (
        out["GGBFS_pct"] * 0.07 + out["FlyAsh_pct"] * 0.02 +
        out["CaO_pct"] * 1.05 + out["Gypsum_pct"] * 0.08 +
        out["CaCl2_pct"] * 0.85 + out["Limestone_pct"] * 0.06
    ) * 10
    return out


def propose_candidates(
    models: Dict[str, Pipeline],
    constraints: Constraints,
    batch_size: int = 6,
    exploration_weight: float = 0.7
) -> pd.DataFrame:
    cands = estimate_cost_carbon(candidate_space())
    x = engineered_features(cands)

    for target, model in models.items():
        mean, std = predict_with_std(model, x)
        cands[f"Pred_{target}"] = mean
        cands[f"Std_{target}"] = std

    # Rule fallback when a target model is unavailable.
    if "Pred_Cost_KRW_t" not in cands:
        cands["Pred_Cost_KRW_t"] = cands["Cost_KRW_t_rule"]
        cands["Std_Cost_KRW_t"] = 0.0
    if "Pred_CO2_kg_t" not in cands:
        cands["Pred_CO2_kg_t"] = cands["CO2_kg_t_rule"]
        cands["Std_CO2_kg_t"] = 0.0

    feasible = pd.Series(True, index=cands.index)
    feasible &= cands["CaO_pct"] <= constraints.max_cao
    feasible &= cands["CaCl2_pct"] <= constraints.max_cacl2
    feasible &= cands["WB"].between(constraints.wb_min, constraints.wb_max)

    checks = {
        "Pred_Strength3d_MPa": (">=", constraints.min_strength3d),
        "Pred_Cost_KRW_t": ("<=", constraints.max_cost),
        "Pred_Absorption_pct": ("<=", constraints.max_absorption),
        "Pred_CO2_kg_t": ("<=", constraints.max_co2),
        "Pred_Expansion_pct": ("<=", constraints.max_expansion),
    }
    for col, (op, value) in checks.items():
        if col in cands:
            feasible &= cands[col] >= value if op == ">=" else cands[col] <= value

    cands["Feasible"] = feasible

    # Utility = exploitation + exploration - normalized burden.
    def z(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    strength = cands.get("Pred_Strength3d_MPa", pd.Series(0, index=cands.index))
    strength_std = cands.get("Std_Strength3d_MPa", pd.Series(0, index=cands.index))
    cost = cands["Pred_Cost_KRW_t"]
    co2 = cands["Pred_CO2_kg_t"]
    absorption = cands.get("Pred_Absorption_pct", pd.Series(0, index=cands.index))
    expansion = cands.get("Pred_Expansion_pct", pd.Series(0, index=cands.index))

    cands["Utility"] = (
        1.4 * z(strength)
        + exploration_weight * z(strength_std)
        - 0.50 * z(cost)
        - 0.45 * z(co2)
        - 0.35 * z(absorption)
        - 0.30 * z(expansion)
    )

    ranked = cands.sort_values(["Feasible", "Utility"], ascending=[False, False]).copy()
    selected = []
    purposes = ["Exploitation", "Exploration", "Low-cost", "Low-CO2", "Mechanism", "Robustness"]
    pool = ranked.head(500)

    # Diversity-aware greedy selection.
    numeric = FEATURES
    scaled = (pool[numeric] - pool[numeric].mean()) / (pool[numeric].std() + 1e-9)
    for k in range(min(batch_size, len(pool))):
        if k == 0:
            idx = pool.index[0]
        else:
            already = scaled.loc[selected]
            dist = np.sqrt(((scaled.values[:, None, :] - already.values[None, :, :]) ** 2).sum(axis=2))
            min_dist = dist.min(axis=1)
            score = pool["Utility"].values + 0.25 * min_dist
            score[[pool.index.get_loc(i) for i in selected]] = -np.inf
            idx = pool.index[int(np.argmax(score))]
        selected.append(idx)

    result = pool.loc[selected].copy()
    result.insert(0, "CandidateID", [f"OODA-CAND-{i+1:02d}" for i in range(len(result))])
    result.insert(1, "Purpose", purposes[:len(result)])
    return result.reset_index(drop=True)


def constraint_assessment(row: pd.Series, c: Constraints) -> List[str]:
    issues = []
    if row.get("CaO_pct", 0) > c.max_cao:
        issues.append("CaO exceeds the configured limit.")
    if row.get("CaCl2_pct", 0) > c.max_cacl2:
        issues.append("CaCl2 exceeds the configured limit.")
    if not c.wb_min <= row.get("WB", 0) <= c.wb_max:
        issues.append("W/B is outside the configured range.")
    if row.get("Pred_Strength3d_MPa", 999) < c.min_strength3d:
        issues.append("Predicted 3-day strength is below target.")
    if row.get("Pred_Cost_KRW_t", 0) > c.max_cost:
        issues.append("Predicted cost exceeds target.")
    if row.get("Pred_Absorption_pct", 0) > c.max_absorption:
        issues.append("Predicted absorption exceeds target.")
    if row.get("Pred_CO2_kg_t", 0) > c.max_co2:
        issues.append("Predicted embodied carbon exceeds target.")
    if row.get("Pred_Expansion_pct", 0) > c.max_expansion:
        issues.append("Predicted expansion exceeds target.")
    return issues


def stage_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stage": "Observe",
            "Input": "Raw-material lots, XRF, XRD phase fractions, TG mass-loss windows, DTG peaks, PSD/BET, mixture, process, curing, performance, cost and CO2",
            "Analysis": "Schema validation, unit harmonization, missingness, outlier and lot-drift checks",
            "Output": "Clean experimental table, data-quality report, raw-material fingerprint"
        },
        {
            "Stage": "Orient",
            "Input": "Validated data + expert constraints + chemistry/process context",
            "Analysis": "Engineered descriptors, Gaussian-process models, uncertainty, mechanistic rules, domain check",
            "Output": "Performance + XRD/TG/DTG predictions, uncertainty intervals, governing variables, risks and mechanistic hypotheses"
        },
        {
            "Stage": "Decide",
            "Input": "Surrogate models, constraints, objectives, candidate design space",
            "Analysis": "Constrained multi-objective search, exploration/exploitation balance, diversity selection",
            "Output": "Ranked candidate mixtures, Pareto trade-offs, reasons for selection"
        },
        {
            "Stage": "Act",
            "Input": "Approved candidate + laboratory capabilities + test standards",
            "Analysis": "Batch calculation, test matrix, stop criteria, quality-control checklist",
            "Output": "Experiment packet, specimen IDs, test schedule, downloadable CSV/JSON"
        },
        {
            "Stage": "Feedback",
            "Input": "Measured results, deviations, failure observations, expert approval/rejection",
            "Analysis": "Prediction-error decomposition, data/process/chemistry/model-failure classification",
            "Output": "Updated dataset, decision log, revised constraints and next OODA cycle"
        },
    ])


def build_packet(row: pd.Series) -> Dict:
    mix = {k: round(float(row[k]), 4) for k in FEATURES if k in row}
    predicted = {
        k.replace("Pred_", ""): round(float(row[k]), 4)
        for k in row.index if str(k).startswith("Pred_")
    }
    return {
        "application": APP_NAME,
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": row.get("CandidateID", "MANUAL"),
        "purpose": row.get("Purpose", "User-selected"),
        "mixture_dry_mass_percent": mix,
        "predicted_performance": {
            k: v for k, v in predicted.items() if k in PERFORMANCE_TARGETS
        },
        "predicted_chemical_analysis": {
            "XRD": {k: v for k, v in predicted.items() if k in XRD_TARGETS},
            "TG": {k: v for k, v in predicted.items() if k in TG_TARGETS},
            "DTG": {k: v for k, v in predicted.items() if k in DTG_TARGETS},
            "interpretation_notice": (
                "Predicted phase fractions and thermal descriptors are hypotheses. "
                "Confirm with consistent XRD quantification and TG-DTG protocols."
            )
        },
        "mandatory_tests": [
            "Flow or rheology", "Initial/final setting",
            "Isothermal calorimetry 72 h", "1 d and 3 d compressive strength",
            "28 d compressive strength", "Absorption",
            "XRD and TG-DTG on representative mixtures",
            "Length change / expansion"
        ],
        "quality_control": [
            "Record supplier and lot for every raw material",
            "Correct batch mass for moisture",
            "Record room, material and discharge temperatures",
            "Use replicate specimens and preserve raw machine files",
            "Do not treat predicted mechanism as confirmed without phase/thermal evidence"
        ],
        "stop_conditions": [
            "Unexpected flash set",
            "Mix temperature exceeds laboratory safety threshold",
            "Visible swelling or cracking",
            "Mass balance or dosing deviation exceeds configured tolerance"
        ]
    }


def _legacy_assistant_reply(prompt: str, df: pd.DataFrame, candidates: pd.DataFrame | None,
                    constraints: Constraints, diagnostics: pd.DataFrame) -> str:
    q = prompt.lower().strip()
    if any(k in q for k in ["observe", "관측", "데이터", "입력"]):
        err = composition_error(df)
        return (
            f"OBSERVE 분석: 현재 {len(df)}개 실험이 로드되었습니다. "
            f"성분합계 오차가 1 wt.%를 초과하는 행은 {(err > 1).sum()}개입니다. "
            "우선 원료 lot, 함수율, 시험체 형상, 양생이력과 시험표준을 함께 보존해야 합니다."
        )
    if any(k in q for k in ["orient", "해석", "모델", "불확실"]):
        if diagnostics.empty:
            return (
                "ORIENT 분석: 데이터가 충분하지 않거나 교차검증 진단을 만들지 못했습니다. "
                "최소 8–10개 이상의 완결된 실험과 반복시험을 확보하십시오."
            )
        best = diagnostics.sort_values("MAE").iloc[0]
        return (
            f"ORIENT 분석: {best['Target']} 모델의 leave-one-out MAE는 "
            f"{best['MAE']:.2f}입니다. 예측 평균만 사용하지 말고 표준편차와 "
            "학습범위 이탈 여부를 함께 판단해야 합니다. CaO·석고·CaCl2 효과는 "
            "열량, XRD, TG-DTG 또는 용출자료 없이 인과로 확정할 수 없습니다."
        )
    if any(k in q for k in ["decide", "결정", "추천", "후보", "최적"]):
        if candidates is None or candidates.empty:
            return "DECIDE 분석을 위해 먼저 '후보 생성' 버튼을 실행하십시오."
        feasible = int(candidates["Feasible"].sum()) if "Feasible" in candidates else 0
        top = candidates.iloc[0]
        return (
            f"DECIDE 분석: {len(candidates)}개 후보 중 {feasible}개가 설정 제약을 충족합니다. "
            f"우선 검토 후보는 {top['CandidateID']}이며, 목적은 {top['Purpose']}입니다. "
            "최고 예측강도 후보만 선택하지 말고 exploitation, exploration, mechanism "
            "및 robustness 후보를 한 batch에서 병행하는 것이 권장됩니다."
        )
    if any(k in q for k in ["act", "실험", "작업지시", "packet"]):
        return (
            "ACT 분석: 승인 후보를 선택하면 건조질량비, W/B, 양생온도, 필수시험, "
            "품질관리 및 중단조건을 포함한 JSON experiment packet을 내려받을 수 있습니다."
        )
    if any(k in q for k in ["feedback", "피드백", "오차", "실패"]):
        return (
            "FEEDBACK 분석: 예측과 실측 차이를 즉시 모델오차로 처리하지 마십시오. "
            "① 계량·시험 데이터 오류, ② 혼합·양생 process drift, ③ 원료 lot/반응성 drift, "
            "④ surrogate model failure 순으로 분해하고, 원인을 decision log에 기록해야 합니다."
        )
    if any(k in q for k in ["도구", "tool", "다운로드", "링크"]):
        return (
            "도구 페이지에서 단계별 공식 링크와 설치 예시를 확인할 수 있습니다. "
            "최소 구성은 Streamlit + scikit-learn + pandas이며, 고급 최적화에는 "
            "Optuna/BoTorch, 실험추적에는 MLflow, pore-solution 해석에는 PHREEQC를 추가합니다."
        )
    return (
        "OODA-MAT 응답: 질문을 OBSERVE, ORIENT, DECIDE, ACT 또는 FEEDBACK 관점으로 "
        "구체화하면 현재 데이터와 후보를 기반으로 답합니다. 예: "
        "'현재 데이터의 품질 문제는?', '다음 실험 후보를 설명해줘', "
        "'예측과 실측 차이를 어떻게 분류하지?'"
    )


def assistant_reply(prompt: str, df: pd.DataFrame, candidates: pd.DataFrame | None,
                    constraints: Constraints, diagnostics: pd.DataFrame) -> str:
    """Return a Korean answer followed by its English explanation."""
    q = prompt.lower().strip()

    def bilingual(korean: str, english: str) -> str:
        return f"### 한국어\n{korean}\n\n### English\n{english}"

    if any(k in q for k in ["observe", "관측", "데이터", "입력"]):
        errors = composition_error(df)
        count = int((errors > 1).sum())
        return bilingual(
            f"OBSERVE 분석: 현재 실험 {len(df)}개가 로드되었습니다. 배합 합계 오차가 "
            f"1 wt.%를 초과하는 실험은 {count}개입니다. 원료 로트, 함수율, 시험체 형상, "
            "양생 이력과 시험 시점을 함께 확인해야 합니다.",
            f"OBSERVE analysis: {len(df)} experiments are loaded, and {count} exceed a 1 wt.% "
            "binder mass-balance error. Review raw-material lots, moisture, specimen geometry, "
            "curing history, and test age together.",
        )
    if any(k in q for k in ["orient", "해석", "모델", "불확실"]):
        if diagnostics.empty:
            return bilingual(
                "ORIENT 분석: 유효 데이터가 부족하여 교차검증 진단을 만들 수 없습니다. "
                "목표변수별로 최소 8개의 완전한 실험 행을 확보해 주세요.",
                "ORIENT analysis: There are not enough valid observations for cross-validation. "
                "Provide at least eight complete experimental rows for each target.",
            )
        best = diagnostics.sort_values("MAE").iloc[0]
        return bilingual(
            f"ORIENT 분석: 가장 낮은 교차검증 오차를 보인 목표는 {best['Target']}이며 "
            f"leave-one-out MAE는 {best['MAE']:.2f}입니다. 평균 예측과 함께 표준편차와 "
            "학습 범위 이탈 여부를 확인해야 합니다.",
            f"ORIENT analysis: The lowest-error target is {best['Target']}, with a leave-one-out "
            f"MAE of {best['MAE']:.2f}. Review predictive uncertainty and applicability-domain "
            "status together with the mean prediction.",
        )
    if any(k in q for k in ["decide", "결정", "추천", "후보", "최적"]):
        if candidates is None or candidates.empty:
            return bilingual(
                "DECIDE 분석을 위해 먼저 후보 생성 버튼을 실행해 주세요.",
                "Run candidate generation before requesting a DECIDE analysis.",
            )
        feasible = int(candidates["Feasible"].sum()) if "Feasible" in candidates else 0
        top = candidates.iloc[0]
        return bilingual(
            f"DECIDE 분석: 후보 {len(candidates)}개 중 {feasible}개가 제약을 충족합니다. "
            f"우선 검토 후보는 {top['CandidateID']}이며 목적은 {top['Purpose']}입니다. "
            "활용·탐색·메커니즘 후보를 함께 시험하는 것이 좋습니다.",
            f"DECIDE analysis: {feasible} of {len(candidates)} candidates satisfy the constraints. "
            f"Review {top['CandidateID']} first; its purpose is {top['Purpose']}. Test exploitation, "
            "exploration, and mechanism-focused candidates together.",
        )
    if any(k in q for k in ["act", "실험", "작업지시", "packet"]):
        return bilingual(
            "ACT 분석: 승인된 후보를 선택하면 배합비, W/B, 양생 조건, 필수 시험, "
            "품질관리 및 중단 조건이 포함된 실험 패킷을 만들 수 있습니다.",
            "ACT analysis: Select an approved candidate to create an experiment packet containing "
            "the mixture, W/B, curing conditions, required tests, quality controls, and stop criteria.",
        )
    if any(k in q for k in ["feedback", "피드백", "오차", "실패"]):
        return bilingual(
            "FEEDBACK 분석: 예측과 실측의 차이를 즉시 모델 오차로 판단하지 마세요. "
            "계량·시험 오류, 공정 편차, 원료 변화, 모델 실패 순서로 원인을 분해해야 합니다.",
            "FEEDBACK analysis: Do not immediately classify a prediction-to-measurement gap as "
            "model error. Check measurement error, process drift, raw-material change, and model failure in order.",
        )
    if any(k in q for k in ["도구", "tool", "다운로드", "링크"]):
        return bilingual(
            "Tools 탭에서 단계별 공식 링크와 설치 예시를 확인할 수 있습니다.",
            "The Tools tab provides official links and installation examples for each stage.",
        )
    return bilingual(
        "질문을 OBSERVE, ORIENT, DECIDE, ACT 또는 FEEDBACK 관점으로 구체화하면 "
        "현재 데이터와 후보를 기반으로 답변할 수 있습니다.",
        "Frame the question around OBSERVE, ORIENT, DECIDE, ACT, or FEEDBACK so OODA-MAT "
        "can answer using the loaded data and candidates.",
    )


def build_openai_agent_prompt(prompt: str, df: pd.DataFrame,
                              candidates: pd.DataFrame | None,
                              constraints: Constraints,
                              diagnostics: pd.DataFrame,
                              messages: List[Dict]) -> Tuple[str, str]:
    """Build bounded OODA-MAT instructions and a compact, data-grounded user prompt."""
    composition_errors = composition_error(df)
    data_context = {
        "experiment_count": int(len(df)),
        "experiment_id_sample": df["ExperimentID"].astype(str).head(10).tolist(),
        "missing_required_cells": int(df[REQUIRED_BASE].isna().sum().sum()),
        "composition_error_over_1pct": int((composition_errors > 1).sum()),
        "available_chemical_targets": [col for col in CHEMICAL_TARGETS if col in df.columns],
        "constraints": asdict(constraints),
    }
    if not diagnostics.empty:
        data_context["model_diagnostics"] = diagnostics.sort_values("MAE").head(10).round(2).to_dict("records")
    if candidates is not None and not candidates.empty:
        candidate_columns = [
            col for col in ["CandidateID", "Purpose", "Feasible", "Score"] if col in candidates.columns
        ]
        data_context["candidate_summary"] = candidates[candidate_columns].head(10).round(2).to_dict("records")

    recent_dialogue = []
    for message in messages[-8:]:
        content = str(message.get("content", ""))
        recent_dialogue.append(f"{message.get('role', 'user')}: {content[:1200]}")

    instructions = (
        "You are the OODA-MAT materials-engineering analysis agent. Answer only from the supplied "
        "project context and clearly label assumptions. Do not invent measurements, standards, "
        "citations, mechanisms, or model results. Predictions are decision support, not experimental "
        "proof. Structure every answer with '### 한국어' first and a faithful '### English' explanation "
        "under it. Use concise engineering language, report numeric values to at most two decimal places, "
        "and organize recommendations using OBSERVE, ORIENT, DECIDE, ACT, and FEEDBACK when applicable."
    )
    agent_input = (
        "PROJECT CONTEXT\n"
        + json.dumps(data_context, ensure_ascii=False, indent=2, default=str)
        + "\n\nRECENT CONVERSATION\n"
        + "\n".join(recent_dialogue)
        + "\n\nUSER QUESTION\n"
        + prompt
    )
    return instructions, agent_input


def openai_agent_reply(api_key: str, model: str, prompt: str, df: pd.DataFrame,
                       candidates: pd.DataFrame | None, constraints: Constraints,
                       diagnostics: pd.DataFrame, messages: List[Dict]) -> Tuple[str, str, str]:
    """Send the prepared OODA-MAT prompt through the OpenAI Responses API."""
    from openai import OpenAI

    instructions, agent_input = build_openai_agent_prompt(
        prompt, df, candidates, constraints, diagnostics, messages
    )
    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
    models_to_try = list(dict.fromkeys([model, "gpt-4o-mini"]))
    last_error = None
    for candidate_model in models_to_try:
        try:
            response = client.responses.create(
                model=candidate_model,
                instructions=instructions,
                input=agent_input,
            )
            return response.output_text, agent_input, candidate_model
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) not in (403, 404):
                raise
    raise last_error


def test_openai_connection(api_key: str, model: str) -> str:
    """Make a minimal user-triggered request and return the model that succeeded."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
    models_to_try = list(dict.fromkeys([model, "gpt-4o-mini"]))
    last_error = None
    for candidate_model in models_to_try:
        try:
            client.responses.create(
                model=candidate_model,
                instructions="Return only the word OK.",
                input="Connection test",
                max_output_tokens=8,
            )
            return candidate_model
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) not in (403, 404):
                raise
    raise last_error


def configured_openai_api_key() -> str:
    """Read the API key without embedding it in source code."""
    secret_key = ""
    try:
        for name in ("OPENAI_API_KEY", "OPENAI_API_Key", "openai_api_key"):
            if st.secrets.get(name, ""):
                secret_key = st.secrets[name]
                break
    except Exception:
        secret_key = ""
    return normalize_openai_api_key(secret_key or os.getenv("OPENAI_API_KEY", ""))


def normalize_openai_api_key(value: object) -> str:
    """Accept a raw key or a pasted OPENAI_API_KEY=... assignment."""
    key = str(value or "").strip()
    if "=" in key and key.split("=", 1)[0].strip().lower() == "openai_api_key":
        key = key.split("=", 1)[1].strip()
    return key.strip().strip('"').strip("'").strip()


def safe_openai_error(exc: Exception) -> str:
    """Return actionable API error details without exposing an API key."""
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", str(exc))
    message = " ".join(message.split())[:500]
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    details = [type(exc).__name__]
    if status:
        details.append(f"HTTP {status}")
    if code:
        details.append(f"code={code}")
    if message:
        details.append(message)
    return " | ".join(details)


def codex_cli_status() -> Tuple[bool, str]:
    """Return whether the local Codex CLI is installed and authenticated."""
    executable = shutil.which("codex")
    if not executable:
        return False, "Codex CLI를 찾을 수 없습니다."
    try:
        result = subprocess.run(
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return False, f"Codex 상태 확인 실패: {type(exc).__name__}"
    status_text = (result.stdout + "\n" + result.stderr).strip()
    logged_in = result.returncode == 0 and "not logged in" not in status_text.lower()
    return logged_in, status_text or f"codex login status exited with {result.returncode}"


def codex_cli_reply(prompt: str, df: pd.DataFrame, candidates: pd.DataFrame | None,
                    constraints: Constraints, diagnostics: pd.DataFrame,
                    messages: List[Dict]) -> Tuple[str, str]:
    """Run an ephemeral, read-only Codex CLI turn and return only its final answer."""
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI를 찾을 수 없습니다.")
    logged_in, status = codex_cli_status()
    if not logged_in:
        raise RuntimeError(f"Codex CLI 로그인이 필요합니다. 현재 상태: {status}")

    instructions, agent_input = build_openai_agent_prompt(
        prompt, df, candidates, constraints, diagnostics, messages
    )
    codex_prompt = (
        instructions
        + "\n\nYou are answering inside the OODA-MAT Conversation tab. "
        + "Do not edit files. Do not execute laboratory or external actions. "
        + "Use the supplied context to answer the question only.\n\n"
        + agent_input
    )
    with tempfile.TemporaryDirectory(prefix="ooda_codex_") as temp_dir:
        output_path = os.path.join(temp_dir, "last_message.md")
        command = [
            executable, "exec", "-",
            "--sandbox", "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color", "never",
            "--output-last-message", output_path,
            "--cd", os.getcwd(),
        ]
        result = subprocess.run(
            command,
            input=codex_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise RuntimeError(f"Codex CLI 실행 실패: {detail}")
        if not os.path.exists(output_path):
            raise RuntimeError("Codex CLI가 최종 응답 파일을 생성하지 않았습니다.")
        with open(output_path, "r", encoding="utf-8") as handle:
            answer = handle.read().strip()
    if not answer:
        raise RuntimeError("Codex CLI가 빈 응답을 반환했습니다.")
    return answer, codex_prompt


def init_state():
    if "data" not in st.session_state:
        st.session_state.data = synthetic_data()
    if "candidates" not in st.session_state:
        st.session_state.candidates = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "OODA-MAT이 시작되었습니다. 왼쪽에서 데이터와 제약조건을 설정한 뒤 "
                "단계별 분석 또는 대화창을 사용하십시오."
            )
        }]
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if "split_input_report" not in st.session_state:
        st.session_state.split_input_report = None


st.set_page_config(
    page_title="OODA-C3",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_state()
if (len(st.session_state.messages) == 1
        and st.session_state.messages[0].get("role") == "assistant"
        and "### 한국어" not in st.session_state.messages[0].get("content", "")):
    st.session_state.messages[0]["content"] = (
        "### 한국어\nOODA-MAT 대화가 시작되었습니다. 데이터를 업로드하고 제약조건을 설정한 후 "
        "분석할 내용을 질문해 주세요.\n\n"
        "### English\nThe OODA-MAT conversation is ready. Upload data, configure the constraints, "
        "and ask your analysis question."
    )

st.title("OODA-C3")
st.caption("OODA-loop based AI-assisted optimization for cement-free construction materials")

with st.sidebar:
    st.header("Project controls")
    st.caption("Upload the general table first. XRD, TG and DTG tables are joined by ExperimentID.")
    uploaded_general = st.file_uploader("General input CSV", type=["csv"], key="general_csv")
    uploaded_analysis = {
        "XRD": st.file_uploader("XRD input CSV", type=["csv"], key="xrd_csv"),
        "TG": st.file_uploader("TG input CSV", type=["csv"], key="tg_csv"),
        "DTG": st.file_uploader("DTG input CSV", type=["csv"], key="dtg_csv"),
    }
    if uploaded_general is not None:
        try:
            incoming, input_report = load_split_input_tables(uploaded_general, uploaded_analysis)
            st.session_state.data = incoming
            st.session_state.split_input_report = input_report
            loaded_groups = input_report.loc[input_report["Rows"] > 0, "Input"].tolist()
            st.success(f"Loaded and joined {len(incoming)} rows: {', '.join(loaded_groups)}")
        except Exception as exc:
            st.error(f"CSV read error: {exc}")

    if st.session_state.split_input_report is not None:
        st.caption("Split-input join status")
        show_dataframe(st.session_state.split_input_report, hide_index=True, width="stretch")

    if st.button("Reset to demonstration data"):
        st.session_state.data = synthetic_data()
        st.session_state.candidates = None
        st.session_state.split_input_report = None
        st.rerun()

    st.subheader("Engineering constraints")
    c = Constraints(
        min_strength3d=st.number_input("Minimum 3-day strength (MPa)", 5.0, 100.0, 30.0),
        max_cost=st.number_input("Maximum binder cost (KRW/t)", 10000.0, 500000.0, 75000.0, 1000.0),
        max_absorption=st.number_input("Maximum absorption (%)", 0.1, 20.0, 3.5, 0.1),
        max_co2=st.number_input("Maximum embodied CO2 (kg/t)", 1.0, 1000.0, 140.0, 5.0),
        max_expansion=st.number_input("Maximum expansion (%)", 0.0, 2.0, 0.10, 0.01),
        max_cao=st.number_input("Maximum CaO (%)", 0.0, 20.0, 6.0, 0.5),
        max_cacl2=st.number_input("Maximum CaCl2 (%)", 0.0, 10.0, 3.0, 0.25),
        wb_min=st.number_input("Minimum W/B", 0.10, 0.70, 0.30, 0.01),
        wb_max=st.number_input("Maximum W/B", 0.10, 0.70, 0.42, 0.01),
    )
    batch_size = st.slider("Candidate batch size", 3, 12, 6)
    exploration_weight = st.slider("Exploration weight", 0.0, 2.0, 0.7, 0.1)

df, missing = ensure_columns(st.session_state.data)
models = fit_models(df) if not missing else {}
diagnostics = model_diagnostics(df, models) if models else pd.DataFrame()

tabs = st.tabs([
    "OODA overview", "1 Observe", "2 Orient", "3 Decide",
    "4 Act", "5 Feedback", "Hydration Simulation", "Chemistry → Performance",
    "Tools", "Conversation"
])

st.markdown(
    """
    <style>
    /* Make the main workflow window names visually distinct. */
    .stTabs [data-baseweb="tab-list"] > [data-baseweb="tab"] {
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] > [data-baseweb="tab"] p {
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with tabs[0]:
    st.subheader("Complete OODA-MAT methodology")
    show_dataframe(stage_table(), width="stretch", hide_index=True)

    st.markdown(
        """
        <style>
        /* Keep every Sankey-stage label in a clean Gothic sans-serif face. */
        .stPlotlyChart .sankey .node-label,
        .stPlotlyChart .sankey text {
            font-family: "Malgun Gothic", "Arial", sans-serif !important;
            font-style: normal !important;
            text-shadow: none !important;
            filter: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    flow = go.Figure(go.Sankey(
        node=dict(
            label=["Raw data", "OBSERVE", "Validated data", "ORIENT",
                   "Models & hypotheses", "DECIDE", "Candidate mixtures",
                   "ACT", "Measured results", "FEEDBACK"]
        ),
        link=dict(
            source=[0,1,2,3,4,5,6,7,8,9],
            target=[1,2,3,4,5,6,7,8,9,1],
            value=[1,1,1,1,1,1,1,1,1,1]
        )
    ))
    flow.update_layout(
        title="OODA-MAT closed-loop information flow",
        height=480,
        font={
            "family": "Malgun Gothic, Arial, sans-serif",
            "size": 14,
            "color": "#111111",
        },
    )
    st.plotly_chart(flow, width="stretch")

    st.markdown("""
    **Core governance rules**

    1. LLM text is never the numerical source of truth.
    2. Every prediction must include uncertainty and applicability-domain checks.
    3. Hard engineering constraints are applied before ranking.
    4. Mechanisms require independent evidence such as calorimetry, XRD, TG-DTG or pore solution.
    5. Expert approval is retained before laboratory execution.
    """)

with tabs[1]:
    st.subheader("OBSERVE — data acquisition, validation and visualization")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiments", len(df))
    available_analysis = [c for c in OPTIONAL_ANALYSIS if c in df.columns]
    col2.metric("Missing required cells", int(df[REQUIRED_BASE].isna().sum().sum()))
    comp_err = composition_error(df)
    col3.metric("Composition error >1%", int((comp_err > 1).sum()))
    col4.metric("Material sources", int(df.get("Source", pd.Series(["Unknown"])).nunique()))

    st.markdown("#### Input tables")
    st.caption("The four tables use ExperimentID as their common key. XRD, TG and DTG values may be blank.")
    input_tabs = st.tabs(["General", "XRD", "TG", "DTG"])
    for input_tab, group in zip(input_tabs, ["General", "XRD", "TG", "DTG"]):
        with input_tab:
            group_df = input_group_table(df, group)
            show_dataframe(group_df, width="stretch", hide_index=True)
            st.download_button(
                f"Download {group} CSV",
                group_df.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
                f"OODA_MAT_{group.lower()}_input.csv",
                "text/csv",
                key=f"download_{group.lower()}_input",
            )

    st.markdown("#### Output: data-quality visualization")
    quality = pd.DataFrame({
        "Variable": REQUIRED_BASE + available_analysis,
        "Missing_pct": [100 * df[c].isna().mean() for c in REQUIRED_BASE + available_analysis],
        "Unique_values": [df[c].nunique(dropna=True) for c in REQUIRED_BASE + available_analysis]
    })
    fig = px.bar(quality, x="Variable", y="Missing_pct",
                 title="Missing-data fraction by variable")
    st.plotly_chart(fig, width="stretch")

    fig2 = px.scatter(
        df.assign(CompositionError_pct=comp_err),
        x="ExperimentID", y="CompositionError_pct",
        hover_data=["GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct"],
        title="Binder composition mass-balance error"
    )
    st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Chemical-analysis coverage")
    coverage_groups = {
        "XRD": XRD_TARGETS,
        "TG": TG_TARGETS,
        "DTG": DTG_TARGETS,
    }
    coverage = []
    for group, cols in coverage_groups.items():
        present = [col for col in cols if col in df.columns]
        valid_fraction = float(df[present].notna().mean().mean()) if present else 0.0
        coverage.append({"Analysis": group, "Available_columns": len(present),
                         "Expected_columns": len(cols), "Valid_fraction_pct": 100*valid_fraction})
    show_dataframe(pd.DataFrame(coverage), width="stretch", hide_index=True)

    st.markdown("#### Output: validated tables")
    output_cols = st.columns(4)
    for output_col, group in zip(output_cols, ["General", "XRD", "TG", "DTG"]):
        group_df = input_group_table(df, group)
        output_col.download_button(
            f"Validated {group}",
            group_df.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
            f"OODA_MAT_validated_{group.lower()}.csv",
            "text/csv",
            key=f"download_validated_{group.lower()}",
        )
    st.download_button(
        "Download merged validated dataset",
        df.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
        "OODA_MAT_validated_data.csv",
        "text/csv",
    )

with tabs[2]:
    st.subheader("ORIENT — mechanistic descriptors, surrogate models and uncertainty")
    st.markdown("#### Input")
    st.write("Validated experimental table, engineered descriptors and configured engineering constraints.")
    show_dataframe(engineered_features(df).head(20), width="stretch")

    st.markdown("### Hydration Models")
    st.caption(
        "Each listed family contains independent Gaussian-process regressors for its "
        "available target variables. Availability depends on the uploaded columns and "
        "requires at least eight valid measurements per target."
    )
    show_dataframe(hydration_model_catalog(models), width="stretch", hide_index=True)
    selected_hydration_model = st.selectbox(
        "Hydration model documentation",
        list(HYDRATION_MODELS),
        key="hydration_model_documentation",
    )
    model_info = HYDRATION_MODELS[selected_hydration_model]
    trained_hydration_targets = [
        item for item in model_info["targets"] if item in models
    ]
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        st.markdown("**Basis**")
        st.write(model_info["basis"])
        st.markdown("**Required Input**")
        st.write(model_info["required_input"])
    with doc_col2:
        st.markdown("**Output**")
        st.write(model_info["output"])
        st.markdown("**Interpretation and limitations**")
        st.write(model_info["interpretation"])
    st.markdown("**Models available in the current dataset**")
    st.write(", ".join(trained_hydration_targets) if trained_hydration_targets else "None")

    st.markdown("#### Output")
    if diagnostics.empty:
        st.warning("Insufficient complete data for cross-validation diagnostics.")
    else:
        show_dataframe(diagnostics, width="stretch", hide_index=True)

    model_targets = [t for t in TARGETS if t in models]
    target = st.selectbox("Target for model visualization", model_targets) if model_targets else None
    if target:
        x = engineered_features(df)
        valid = df[target].notna()
        mean, std = predict_with_std(models[target], x.loc[valid])
        plot_df = pd.DataFrame({
            "Measured": df.loc[valid, target].values,
            "Predicted": mean,
            "Lower95": mean - 1.96 * std,
            "Upper95": mean + 1.96 * std,
            "ExperimentID": df.loc[valid, "ExperimentID"].values
        })
        fig = px.scatter(plot_df, x="Measured", y="Predicted", hover_name="ExperimentID",
                         error_y=1.96 * std, title=f"Measured versus model prediction: {target}")
        lo = min(plot_df["Measured"].min(), plot_df["Predicted"].min())
        hi = max(plot_df["Measured"].max(), plot_df["Predicted"].max())
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line_dash="dash")
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Variable Sensitivity Analysis")
        st.caption(
            "The analysis now separates local constrained OAT response from global "
            "constraint-aware permutation response. Binder fractions are always closed "
            "to 100 wt.%, predictive uncertainty is retained, and local sweeps are checked "
            "against an empirical applicability-domain distance."
        )
        sensitivity_balance_mode = st.selectbox(
            "Binder mass-balance rule",
            SENSITIVITY_BALANCE_OPTIONS,
            key="sensitivity_balance_mode",
            help=(
                "Proportional closure rescales all other binder constituents. Paired "
                "replacement balances GGBFS against fly ash and all other binder changes "
                "against GGBFS."
            ),
        )
        sensitivity_reference = st.selectbox(
            "Sensitivity reference composition",
            ["Median input profile", *df["ExperimentID"].astype(str).tolist()],
            key="sensitivity_reference",
        )
        local_tab, global_tab, interpretation_tab = st.tabs([
            "Local constrained OAT",
            "Global permutation response",
            "How to interpret",
        ])

        with local_tab:
            sensitivity_summary, sensitivity_detail = sensitivity_analysis(
                df,
                models[target],
                target,
                sensitivity_reference,
                balance_mode=sensitivity_balance_mode,
            )
            if sensitivity_summary.empty:
                st.warning(
                    "Sensitivity could not be calculated because input ranges are insufficient."
                )
            else:
                sensitivity_fig = px.bar(
                    sensitivity_summary,
                    x="OAT_range_effect",
                    y="Variable",
                    color="Confidence",
                    orientation="h",
                    hover_data=[
                        "Low_5pct", "Reference", "High_95pct",
                        "Prediction_at_low", "Prediction_at_high",
                        "Local_standardized_slope", "Nonlinearity_index",
                        "Mean_uncertainty_index", "Outside_domain_fraction",
                        "Direction",
                    ],
                    title=f"Constraint-aware local sensitivity: {target}",
                )
                sensitivity_fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(sensitivity_fig, width="stretch")

                response_fig = px.line(
                    sensitivity_detail,
                    x="Range_position_pct",
                    y="Change_from_reference",
                    color="Variable",
                    line_dash="Outside_domain",
                    hover_data=[
                        "Prediction", "Prediction_std",
                        "Applicability_distance", "Applicability_threshold",
                        "Balance_rule",
                    ],
                    title=f"Local response around the selected reference: {target}",
                    labels={
                        "Range_position_pct": "Position within the permitted sweep range (%)",
                        "Change_from_reference": "Prediction change from reference",
                    },
                )
                response_fig.add_hline(
                    y=0.0, line_dash="dash", line_color="gray"
                )
                st.plotly_chart(response_fig, width="stretch")
                show_dataframe(
                    sensitivity_summary.round(5),
                    width="stretch",
                    hide_index=True,
                )

        with global_tab:
            global_samples = st.slider(
                "Global sensitivity sample count",
                200, 2000, 800, 100,
                key="global_sensitivity_samples",
            )
            if st.button(
                "Run global permutation sensitivity",
                key="run_global_sensitivity",
            ):
                st.session_state.global_sensitivity_result = {
                    "target": target,
                    "balance_mode": sensitivity_balance_mode,
                    "data": global_sensitivity_analysis(
                        df,
                        models[target],
                        target,
                        n_samples=global_samples,
                        balance_mode=sensitivity_balance_mode,
                    ),
                }

            global_state = st.session_state.get("global_sensitivity_result")
            if (
                global_state
                and global_state.get("target") == target
                and global_state.get("balance_mode") == sensitivity_balance_mode
            ):
                global_result = global_state["data"]
                if global_result.empty:
                    st.warning("No valid global perturbations were generated.")
                else:
                    global_fig = px.bar(
                        global_result,
                        x="Global_permutation_effect",
                        y="Variable",
                        color="Outside_domain_fraction",
                        orientation="h",
                        hover_data=[
                            "P95_prediction_change",
                            "Spearman_model_association",
                            "Mean_prediction_uncertainty",
                            "Valid_perturbation_fraction",
                            "Method",
                        ],
                        title=f"Global model-response sensitivity: {target}",
                    )
                    global_fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )
                    st.plotly_chart(global_fig, width="stretch")
                    show_dataframe(
                        global_result.round(5),
                        width="stretch",
                        hide_index=True,
                    )
            else:
                st.info(
                    "Run the global analysis after selecting the target and mass-balance rule."
                )

        with interpretation_tab:
            show_dataframe(
                pd.DataFrame([
                    {
                        "Metric": "OAT_range_effect",
                        "Meaning": "Prediction range over the observed 5-95% input sweep divided by measured target standard deviation.",
                        "Use": "Ranks local model response around the selected reference."
                    },
                    {
                        "Metric": "Local_standardized_slope",
                        "Meaning": "Finite-difference slope at the reference, normalized by input and target variation.",
                        "Use": "Indicates local sign and steepness."
                    },
                    {
                        "Metric": "Nonlinearity_index",
                        "Meaning": "Deviation from the straight line between sweep endpoints.",
                        "Use": "Large values warn that endpoint direction alone is misleading."
                    },
                    {
                        "Metric": "Outside_domain_fraction",
                        "Meaning": "Fraction of sweep points beyond the 95% nearest-neighbour distance of training data.",
                        "Use": "High values reduce confidence and indicate extrapolation."
                    },
                    {
                        "Metric": "Global_permutation_effect",
                        "Meaning": "Mean absolute model change after constraint-aware permutation, normalized by target variation.",
                        "Use": "Screens global dependency; it is not a Sobol index."
                    },
                ]),
                width="stretch",
                hide_index=True,
            )
            st.warning(
                "Sensitivity describes the fitted model, not a proven causal hydration "
                "mechanism. Correlated composition variables, lot effects, measurement "
                "uncertainty and closure choice must be reviewed before experimental decisions."
            )

    st.markdown("#### Measured chemical-analysis profiles")
    selected_exp = st.selectbox("Experiment for XRD/TG/DTG profile", df["ExperimentID"].astype(str).tolist())
    erow = df.loc[df["ExperimentID"].astype(str) == selected_exp].iloc[0]
    chem_tabs = st.tabs(["XRD", "TG", "DTG"])
    with chem_tabs[0]:
        xrd_rows = [{"Phase": c.replace("XRD_", "").replace("_pct", ""), "Fraction_pct": erow.get(c, np.nan)}
                    for c in XRD_TARGETS]
        xrd_df = pd.DataFrame(xrd_rows).dropna()
        if xrd_df.empty:
            st.info("No XRD phase-fraction data for this experiment.")
        else:
            st.plotly_chart(px.bar(xrd_df, x="Phase", y="Fraction_pct",
                                   title=f"XRD semi-quantitative phase profile: {selected_exp}"),
                            width="stretch")
    with chem_tabs[1]:
        tg_rows = [{"Temperature_window": c.replace("TG_Loss_", "").replace("_pct", ""),
                    "Mass_loss_pct": erow.get(c, np.nan)} for c in TG_TARGETS[:-1]]
        tg_df = pd.DataFrame(tg_rows).dropna()
        if tg_df.empty:
            st.info("No TG temperature-window data for this experiment.")
        else:
            st.plotly_chart(px.bar(tg_df, x="Temperature_window", y="Mass_loss_pct",
                                   title=f"TG mass-loss windows: {selected_exp}"),
                            width="stretch")
    with chem_tabs[2]:
        dtg_rows = []
        for i in (1, 2, 3):
            t = erow.get(f"DTG_Peak{i}_T_C", np.nan)
            intensity = erow.get(f"DTG_Peak{i}_Intensity_pct_min", np.nan)
            if pd.notna(t) and pd.notna(intensity):
                dtg_rows.append({"Peak": f"Peak {i}", "Temperature_C": t, "Intensity_pct_min": intensity})
        dtg_df = pd.DataFrame(dtg_rows)
        if dtg_df.empty:
            st.info("No DTG peak data for this experiment.")
        else:
            st.plotly_chart(px.scatter(dtg_df, x="Temperature_C", y="Intensity_pct_min",
                                       text="Peak", size="Intensity_pct_min",
                                       title=f"DTG peak descriptors: {selected_exp}"),
                            width="stretch")

    st.info(
        "Proxy descriptors in this prototype must be replaced by XRF-derived molar ratios, "
        "reactive glass fraction, fineness and lot-specific reactivity for research use."
    )

with tabs[3]:
    st.subheader("DECIDE — constrained multi-objective candidate selection")
    st.markdown("#### Input")
    st.json(asdict(c))
    if st.button("Generate OODA candidate batch", type="primary"):
        if not models:
            st.error("Models cannot be fitted. Check the uploaded data.")
        else:
            st.session_state.candidates = propose_candidates(
                models, c, batch_size=batch_size,
                exploration_weight=exploration_weight
            )

    cand = st.session_state.candidates
    st.markdown("#### Output")
    if cand is None:
        st.info("Press the candidate-generation button.")
    else:
        show_cols = [
            "CandidateID", "Purpose", "Feasible", "Utility",
            *FEATURES,
            "Pred_Strength3d_MPa", "Std_Strength3d_MPa",
            "Pred_Absorption_pct", "Pred_Cost_KRW_t",
            "Pred_CO2_kg_t", "Pred_Expansion_pct",
            "Pred_XRD_CASH_CSH_pct", "Pred_XRD_Ettringite_pct",
            "Pred_XRD_ResidualGypsum_pct", "Pred_XRD_Amorphous_pct",
            "Pred_TG_TotalLoss_pct", "Pred_DTG_Peak1_T_C",
            "Pred_DTG_Peak2_T_C", "Pred_DTG_Peak3_T_C"
        ]
        show_cols = [x for x in show_cols if x in cand.columns]
        show_dataframe(cand[show_cols], width="stretch", hide_index=True)

        if {"Pred_Cost_KRW_t", "Pred_Strength3d_MPa"}.issubset(cand.columns):
            fig = px.scatter(
                cand, x="Pred_Cost_KRW_t", y="Pred_Strength3d_MPa",
                size="Std_Strength3d_MPa", symbol="Purpose",
                hover_name="CandidateID", color="Feasible",
                title="Strength–cost trade-off; marker size indicates uncertainty"
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Predicted candidate chemistry")
        chemistry_candidate = st.selectbox("Candidate chemistry profile", cand["CandidateID"].tolist(), key="chem_candidate")
        crow = cand.loc[cand["CandidateID"] == chemistry_candidate].iloc[0]
        profile_tabs = st.tabs(["Predicted XRD", "Predicted TG", "Predicted DTG"])
        with profile_tabs[0]:
            rows = [{"Phase": col.replace("XRD_", "").replace("_pct", ""),
                     "Predicted_fraction_pct": crow.get(f"Pred_{col}", np.nan),
                     "Std": crow.get(f"Std_{col}", np.nan)} for col in XRD_TARGETS]
            pp = pd.DataFrame(rows).dropna(subset=["Predicted_fraction_pct"])
            if pp.empty:
                st.info("XRD models require at least eight valid measured rows per phase variable.")
            else:
                st.plotly_chart(px.bar(pp, x="Phase", y="Predicted_fraction_pct",
                                       error_y="Std", title=f"Predicted XRD profile: {chemistry_candidate}"),
                                width="stretch")
        with profile_tabs[1]:
            rows = [{"Window": col.replace("TG_Loss_", "").replace("_pct", ""),
                     "Predicted_mass_loss_pct": crow.get(f"Pred_{col}", np.nan),
                     "Std": crow.get(f"Std_{col}", np.nan)} for col in TG_TARGETS[:-1]]
            pp = pd.DataFrame(rows).dropna(subset=["Predicted_mass_loss_pct"])
            if pp.empty:
                st.info("TG models require at least eight valid measured rows per temperature window.")
            else:
                st.plotly_chart(px.bar(pp, x="Window", y="Predicted_mass_loss_pct",
                                       error_y="Std", title=f"Predicted TG windows: {chemistry_candidate}"),
                                width="stretch")
        with profile_tabs[2]:
            rows = []
            for i in (1,2,3):
                t = crow.get(f"Pred_DTG_Peak{i}_T_C", np.nan)
                intensity = crow.get(f"Pred_DTG_Peak{i}_Intensity_pct_min", np.nan)
                if pd.notna(t) and pd.notna(intensity):
                    rows.append({"Peak": f"Peak {i}", "Temperature_C": t, "Intensity_pct_min": intensity})
            pp = pd.DataFrame(rows)
            if pp.empty:
                st.info("DTG models require at least eight valid measured rows for peak descriptors.")
            else:
                st.plotly_chart(px.scatter(pp, x="Temperature_C", y="Intensity_pct_min",
                                           size="Intensity_pct_min", text="Peak",
                                           title=f"Predicted DTG peaks: {chemistry_candidate}"),
                                width="stretch")

        st.warning("Chemical predictions are screening hypotheses, not phase identification. Keep analysis age, atmosphere, heating rate, sample preparation and quantification method constant.")

        st.download_button(
            "Download candidate batch CSV",
            cand.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
            "OODA_MAT_candidates.csv", "text/csv"
        )

with tabs[4]:
    st.subheader("ACT — experiment packet and execution controls")
    cand = st.session_state.candidates
    if cand is None or cand.empty:
        st.warning("Generate candidates in the DECIDE tab first.")
    else:
        selected_id = st.selectbox("Candidate to approve", cand["CandidateID"].tolist())
        row = cand.loc[cand["CandidateID"] == selected_id].iloc[0]
        issues = constraint_assessment(row, c)
        if issues:
            st.error("Critic-agent result: HOLD")
            for issue in issues:
                st.write("- " + issue)
        else:
            st.success("Critic-agent result: conditionally acceptable for laboratory verification")

        packet = build_packet(row)
        st.json(packet)
        st.download_button(
            "Download experiment packet JSON",
            json.dumps(packet, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{selected_id}_experiment_packet.json",
            "application/json"
        )

        batch_mass = st.number_input("Dry binder batch mass (kg)", 0.1, 1000.0, 10.0, 0.5)
        mass_rows = []
        for comp in ["GGBFS_pct", "FlyAsh_pct", "CaO_pct", "Gypsum_pct", "CaCl2_pct", "Limestone_pct"]:
            mass_rows.append({"Component": comp.replace("_pct", ""), "Dry_mass_kg": batch_mass * row[comp] / 100})
        mass_rows.append({"Component": "Water", "Dry_mass_kg": batch_mass * row["WB"]})
        show_dataframe(pd.DataFrame(mass_rows), width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("FEEDBACK — measured-result return and expert learning")
    st.info(
        "**Purpose (목적)**  · ACT 단계에서 실험한 후보 배합의 실측 결과를 "
        "예측값과 연결하고, 차이의 원인을 분리하여 다음 OBSERVE–ORIENT–DECIDE "
        "순환의 데이터 품질, 모델, 제약조건과 실험 계획을 개선합니다."
    )
    st.markdown(
        """
        **Process (진행 절차)**

        1. **Reference selection** — 실험한 후보 배합 또는 기존 실험을 선택합니다.
        2. **Measured-result entry** — 강도와 XRD, TG, DTG 실측값을 같은 시료·재령 기준으로 입력합니다.
        3. **Discrepancy review** — 예측–실측 차이와 실험/공정 이상을 기록하고 원인을 일차 분류합니다.
        4. **Expert decision** — Accept, Repeat, Modify, Reject 중 후속 조치를 결정합니다.
        5. **Loop closure** — 기록을 저장·내보낸 후 검증된 결과를 다음 모델 학습과 후보 선정에 반영합니다.
        """
    )
    st.caption(
        "Feedback 기록은 현재 세션에 보관됩니다. CSV를 내보낸 뒤 검토·승인된 "
        "결과만 학습 데이터에 편입하세요."
    )
    cand = st.session_state.candidates
    ids = ["Manual / existing experiment"] + ([] if cand is None else cand["CandidateID"].tolist())
    f_id = st.selectbox("Experiment or candidate", ids)
    measured_s3 = st.number_input("Measured 3-day strength (MPa)", 0.0, 200.0, 30.0)
    st.markdown("##### Optional measured chemical-analysis feedback")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        measured_xrd_cash = st.number_input("XRD C-(A)-S-H proxy / quantified fraction (%)", 0.0, 100.0, 0.0)
        measured_xrd_ett = st.number_input("XRD ettringite (%)", 0.0, 100.0, 0.0)
        measured_xrd_amorph = st.number_input("XRD amorphous fraction (%)", 0.0, 100.0, 0.0)
    with fcol2:
        measured_tg_total = st.number_input("TG total mass loss (%)", 0.0, 100.0, 0.0)
        measured_tg_low = st.number_input("TG loss 30–200 C (%)", 0.0, 100.0, 0.0)
    with fcol3:
        measured_dtg_p1 = st.number_input("DTG peak 1 temperature (C)", 0.0, 1000.0, 0.0)
        measured_dtg_p3 = st.number_input("DTG peak 3 temperature (C)", 0.0, 1000.0, 0.0)
    observation = st.text_area("Observed anomalies / process deviations")
    cause = st.selectbox(
        "Preliminary discrepancy classification",
        ["No discrepancy", "Data/test error", "Process drift",
         "Raw-material chemistry drift", "Model failure", "Unknown"]
    )
    decision = st.selectbox("Expert decision", ["Accept", "Repeat", "Modify", "Reject"])
    if st.button("Record feedback"):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reference": f_id,
            "measured_strength3d_MPa": measured_s3,
            "XRD_CASH_CSH_pct": measured_xrd_cash if measured_xrd_cash > 0 else None,
            "XRD_Ettringite_pct": measured_xrd_ett if measured_xrd_ett > 0 else None,
            "XRD_Amorphous_pct": measured_xrd_amorph if measured_xrd_amorph > 0 else None,
            "TG_TotalLoss_pct": measured_tg_total if measured_tg_total > 0 else None,
            "TG_Loss_30_200_pct": measured_tg_low if measured_tg_low > 0 else None,
            "DTG_Peak1_T_C": measured_dtg_p1 if measured_dtg_p1 > 0 else None,
            "DTG_Peak3_T_C": measured_dtg_p3 if measured_dtg_p3 > 0 else None,
            "observation": observation,
            "preliminary_cause": cause,
            "expert_decision": decision
        }
        st.session_state.feedback_log.append(record)
        st.success("Feedback recorded in the current session.")

    if st.session_state.feedback_log:
        fb = pd.DataFrame(st.session_state.feedback_log)
        show_dataframe(fb, width="stretch", hide_index=True)
        st.download_button(
            "Download feedback log",
            fb.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
            "OODA_MAT_feedback_log.csv", "text/csv"
        )

    st.markdown("""
    **Mandatory error-decomposition order**

    1. Check weighing, moisture correction, specimen geometry and test-machine raw files.
    2. Check mixing energy, discharge temperature, curing temperature and humidity.
    3. Check material lot, storage carbonation/hydration and chemical/reactivity drift.
    4. Only then attribute the residual discrepancy to model inadequacy.
    """)

with tabs[8]:
    st.subheader("Stage-specific tools, official links and examples")
    tools_df = pd.DataFrame([
        {"Tool": name, **info} for name, info in TOOL_LINKS.items()
    ])
    show_dataframe(
        tools_df[["Tool", "stage", "role", "example"]],
        width="stretch", hide_index=True
    )
    for name, info in TOOL_LINKS.items():
        st.markdown(f"**{name}** — [{info['url']}]({info['url']})")
        st.code(info["example"], language="bash" if "pip " in info["example"] else "python")

with tabs[9]:
    st.subheader("Conversation with OODA-MAT")
    conversation_backend = st.radio(
        "AI response backend",
        ["Codex CLI", "OpenAI API", "Local"],
        horizontal=True,
        key="conversation_backend",
        help="Codex CLI는 로컬 Codex 로그인, OpenAI API는 별도의 API 키가 필요합니다.",
    )
    st.caption(
        "OpenAI Responses API에 현재 데이터 요약과 질문을 함께 전달합니다. "
        "API 키가 없으면 내장된 로컬 분석 응답을 사용합니다."
    )
    agent_col1, agent_col2 = st.columns([2, 1])
    with agent_col1:
        entered_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="코드나 세션 기록에 저장하지 않습니다. Streamlit secrets의 OPENAI_API_KEY도 사용할 수 있습니다.",
            key="conversation_openai_api_key",
        )
    with agent_col2:
        if st.session_state.get("conversation_openai_model") == "gpt-4.1-mini":
            st.session_state.conversation_openai_model = "gpt-4o-mini"
        openai_model = st.text_input(
            "OpenAI model",
            value="gpt-4o-mini",
            key="conversation_openai_model",
        )
    active_api_key = normalize_openai_api_key(entered_api_key) or configured_openai_api_key()
    codex_logged_in, codex_status_text = codex_cli_status()
    if conversation_backend == "Codex CLI":
        if codex_logged_in:
            st.success("Codex CLI가 로그인되어 있습니다. 읽기 전용 Agent 응답을 사용합니다.")
        else:
            st.warning(
                "Codex CLI가 로그인되지 않았습니다. 터미널에서 `codex login --device-auth`를 "
                "실행한 후 이 페이지를 다시 실행하세요. 현재 상태: " + codex_status_text
            )
    if active_api_key:
        st.success("OpenAI Agent mode is ready.")
        if st.button("OpenAI API 연결 시험", key="test_openai_connection"):
            try:
                with st.spinner("OpenAI API 연결을 확인하고 있습니다..."):
                    tested_model = test_openai_connection(
                        active_api_key, openai_model.strip() or "gpt-4o-mini"
                    )
                st.session_state.openai_connection_status = (
                    f"연결 성공: `{tested_model}` 모델을 사용할 수 있습니다."
                )
            except Exception as exc:
                st.session_state.openai_connection_status = (
                    "연결 실패: " + safe_openai_error(exc)
                )
    else:
        st.info("OPENAI_API_KEY가 없어 Local fallback mode로 동작합니다.")

    if st.session_state.get("openai_connection_status"):
        status_text = st.session_state.openai_connection_status
        if status_text.startswith("연결 성공"):
            st.success(status_text)
        else:
            st.error(status_text)

    if st.session_state.get("last_agent_prompt"):
        with st.expander("OpenAI에 전달된 정리된 Prompt 보기"):
            st.code(st.session_state.last_agent_prompt, language="text")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("OODA-MAT에 분석할 내용을 한국어 또는 영어로 질문하세요")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if conversation_backend == "Codex CLI":
            if codex_logged_in:
                try:
                    with st.spinner("Codex가 질문과 현재 데이터를 분석하고 있습니다..."):
                        reply, prepared_prompt = codex_cli_reply(
                            prompt,
                            df,
                            st.session_state.candidates,
                            c,
                            diagnostics,
                            st.session_state.messages,
                        )
                    st.session_state.last_agent_prompt = prepared_prompt
                except Exception as exc:
                    reply = (
                        "### Codex CLI 연결 오류\n"
                        + safe_openai_error(exc)
                        + "\n\n터미널에서 `codex login --device-auth` 실행 후 다시 시도하세요."
                    )
            else:
                reply = (
                    "### 한국어\nCodex CLI 로그인이 필요합니다. 터미널에서 "
                    "`codex login --device-auth`를 실행한 후 다시 질문해 주세요.\n\n"
                    "### English\nCodex CLI authentication is required. Run "
                    "`codex login --device-auth` in a terminal, then try again."
                )
        elif conversation_backend == "OpenAI API" and active_api_key:
            try:
                with st.spinner("OpenAI Agent가 질문과 데이터를 분석하고 있습니다..."):
                    reply, prepared_prompt, used_model = openai_agent_reply(
                        active_api_key,
                        openai_model.strip() or "gpt-4o-mini",
                        prompt,
                        df,
                        st.session_state.candidates,
                        c,
                        diagnostics,
                        st.session_state.messages,
                    )
                st.session_state.last_agent_prompt = prepared_prompt
                st.session_state.openai_connection_status = (
                    f"연결 성공: `{used_model}` 모델을 사용했습니다."
                )
            except Exception as exc:
                local_reply = assistant_reply(
                    prompt, df, st.session_state.candidates, c, diagnostics
                )
                error_detail = safe_openai_error(exc)
                reply = (
                    f"{local_reply}\n\n---\n"
                    "### OpenAI API 연결 진단\n"
                    f"OpenAI 연결에 실패하여 로컬 응답을 사용했습니다. `{error_detail}`\n\n"
                    "- HTTP 401: API 키와 프로젝트 권한을 확인하세요.\n"
                    "- HTTP 403/404: 계정에서 사용할 수 있는 모델명인지 확인하세요.\n"
                    "- HTTP 429: API 크레딧, 사용 한도 또는 요청 제한을 확인하세요.\n"
                    "- 연결/시간 초과: 방화벽과 인터넷 연결을 확인하세요."
                )
        else:
            reply = assistant_reply(
                prompt, df, st.session_state.candidates, c, diagnostics
            )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

with tabs[6]:
    st.subheader("Embedded physics/chemistry-informed hydration simulation")
    st.warning(
        "These reduced-order models are research-screening tools. They do not reproduce or "
        "replace GEMS/CemGEMS, CEMHYD3D or HYMOSTRUC3D, and all default constants require "
        "calibration for each raw-material lot."
    )

    hydration_source = st.selectbox(
        "Experimental mixture used as simulation input",
        df["ExperimentID"].astype(str).tolist(),
        key="embedded_hydration_source",
    )
    hydration_row = df.loc[
        df["ExperimentID"].astype(str) == hydration_source
    ].iloc[0]
    hydration_mix = hydration_row.to_dict()

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    with hcol1:
        hydration_mix["Blaine_m2kg"] = st.number_input(
            "Blaine fineness (m²/kg)", 100.0, 1200.0, 450.0, 10.0,
            key="embedded_blaine",
        )
    with hcol2:
        hydration_mix["D50_um"] = st.number_input(
            "Median particle size D50 (µm)", 1.0, 100.0, 12.0, 1.0,
            key="embedded_d50",
        )
    with hcol3:
        hydration_mix["ReactiveSlag_frac"] = st.number_input(
            "Reactive slag fraction", 0.0, 1.0, 0.76, 0.01,
            key="embedded_reactive_slag",
        )
    with hcol4:
        hydration_mix["ReactiveFlyAsh_frac"] = st.number_input(
            "Reactive fly-ash fraction", 0.0, 1.0, 0.32, 0.01,
            key="embedded_reactive_flyash",
        )

    embedded_model_name = st.selectbox(
        "Hydration model",
        list(MODEL_INFO),
        key="embedded_hydration_model",
    )
    embedded_info = MODEL_INFO[embedded_model_name]
    st.markdown(
        f"**Model classification:** {embedded_info.get('classification', 'Not specified')}"
    )
    basis_tab, equations_tab, calibration_tab, scope_tab = st.tabs([
        "Physical and chemical basis",
        "Governing equations / state variables",
        "Calibration and validation",
        "Applicability and limitations",
    ])
    with basis_tab:
        st.markdown("**Basis**")
        st.write(embedded_info["basis"])
        st.markdown("**Required input**")
        st.write(embedded_info["inputs"])
        st.markdown("**Outputs**")
        st.write(embedded_info["outputs"])
    with equations_tab:
        st.markdown("**Governing equations or numerical relations**")
        for equation in embedded_info.get("governing_equations", []):
            st.code(equation, language="text")
        st.markdown("**State variables**")
        st.write(embedded_info.get("state_variables", "Not specified"))
        st.markdown("**Numerical scheme**")
        st.write(embedded_info.get("numerical_scheme", "Not specified"))
    with calibration_tab:
        st.markdown("**Calibration basis**")
        st.write(embedded_info.get("calibration", "Not specified"))
        st.markdown("**Independent validation**")
        st.write(embedded_info.get("validation", "Not specified"))
    with scope_tab:
        st.markdown("**Recommended applicability**")
        st.write(embedded_info.get("applicability", "Not specified"))
        st.markdown("**Limitations**")
        st.write(embedded_info["limitation"])
        st.markdown("**Reference basis**")
        for reference in embedded_info.get("references", []):
            st.write("- " + reference)

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        max_simulation_age = st.selectbox(
            "Maximum simulation age",
            [72.0, 168.0, 672.0, 2160.0],
            index=2,
            format_func=lambda value: f"{value:g} h ({value / 24:g} d)",
            key="embedded_max_age",
        )
    with control_col2:
        simulation_points = st.slider(
            "Time-grid points", 12, 120, 48, 4,
            key="embedded_time_points",
        )

    input_issues = validate_mix(hydration_mix)
    if input_issues:
        for issue in input_issues:
            st.error(issue)
    else:
        st.success("Mixture input is within the embedded model's screening checks.")

    if st.button(
        "Run embedded hydration simulation",
        type="primary",
        key="run_embedded_hydration",
    ):
        try:
            simulation_times = np.unique(np.concatenate([
                np.array([0.0]),
                np.geomspace(0.1, max_simulation_age, simulation_points),
            ]))
            simulation_result = run_model(
                embedded_model_name,
                hydration_mix,
                simulation_times,
            )
            standard_ages = [
                age for age in (24.0, 72.0, 168.0, 672.0, 2160.0)
                if age <= max_simulation_age
            ]
            if max_simulation_age not in standard_ages:
                standard_ages.append(max_simulation_age)
            st.session_state.embedded_hydration_run = {
                "model": embedded_model_name,
                "source": hydration_source,
                "mix": hydration_mix,
                "result": simulation_result,
                "summary": summary_at_ages(hydration_mix, standard_ages),
                "recipe": gems_recipe(hydration_mix),
            }
        except Exception as exc:
            st.session_state.embedded_hydration_run = None
            st.exception(exc)

    embedded_run = st.session_state.get("embedded_hydration_run")
    if embedded_run:
        result = embedded_run["result"]
        st.markdown(
            f"### Result: {embedded_run['model']} — {embedded_run['source']}"
        )
        numeric_outputs = [
            column for column in result.select_dtypes(include=[np.number]).columns
            if column != "Time_h"
        ]
        st.markdown("#### Graphical model results")
        st.caption(
            "Outputs are separated by physical meaning so quantities with incompatible units "
            "are not superimposed. Hover over a curve to inspect its value and age."
        )
        time_axis_mode = st.radio(
            "Time-axis scale",
            ["Logarithmic", "Linear"],
            horizontal=True,
            key="embedded_time_axis_mode",
        )
        plot_result = result.loc[
            result["Time_h"] > 0
        ].copy() if time_axis_mode == "Logarithmic" else result.copy()

        output_groups = {
            "Reaction kinetics": [
                c for c in numeric_outputs
                if c == "Alpha" or c.startswith("Alpha_") or "Rate_" in c
            ],
            "Heat evolution": [
                c for c in numeric_outputs if "Heat" in c
            ],
            "Hydrate / phase assemblage": [
                c for c in numeric_outputs
                if c.endswith("_pct") and c not in {"Time_h"}
            ],
            "Pore structure / connectivity": [
                c for c in numeric_outputs
                if any(token in c for token in (
                    "Porosity", "Volume_frac", "Connectivity", "GelSpace",
                    "Contact", "ParticleFraction", "Percolation",
                ))
            ],
            "Strength": [
                c for c in numeric_outputs if "Strength" in c
            ],
            "Solution chemistry": [
                c for c in numeric_outputs
                if c.endswith("_proxy") and "Strength" not in c
            ],
        }
        assigned_outputs = {
            column for columns in output_groups.values() for column in columns
        }
        other_outputs = [c for c in numeric_outputs if c not in assigned_outputs]
        if other_outputs:
            output_groups["Other model outputs"] = other_outputs
        output_groups = {name: columns for name, columns in output_groups.items() if columns}

        chart_tabs = st.tabs(list(output_groups))
        for chart_tab, (group_name, group_columns) in zip(chart_tabs, output_groups.items()):
            with chart_tab:
                chart_data = plot_result[["Time_h", *group_columns]].melt(
                    id_vars="Time_h", var_name="Output", value_name="Value"
                )
                chart = px.line(
                    chart_data,
                    x="Time_h",
                    y="Value",
                    color="Output",
                    markers=len(plot_result) <= 30,
                    log_x=time_axis_mode == "Logarithmic",
                    title=f"{group_name} over curing time",
                )
                chart.update_layout(
                    xaxis_title="Curing age (h)",
                    yaxis_title="Model output",
                    legend_title_text="Output",
                    hovermode="x unified",
                )
                st.plotly_chart(chart, width="stretch")

        st.markdown("#### Interactive 3-D hydration-product structure")
        snapshot_age = st.select_slider(
            "3-D snapshot age (h)",
            options=result["Time_h"].astype(float).tolist(),
            value=float(result["Time_h"].iloc[-1]),
            format_func=lambda age: f"{age:g}",
            key="embedded_3d_snapshot_age",
        )
        snapshot_row = result.iloc[(result["Time_h"] - snapshot_age).abs().argmin()]
        snapshot_alpha = float(snapshot_row.get("Alpha", 0.0))
        particle_cloud = particle_snapshot(embedded_run["mix"], snapshot_alpha)
        core_scale = np.clip(1.0 - 0.48 * snapshot_alpha / max(
            reaction_capacity(embedded_run["mix"]), 1e-8
        ), 0.20, 1.0)
        core_sizes = 800.0 * particle_cloud["radius"] * core_scale
        shell_sizes = particle_cloud["shell_diameter_plot"]

        structure_3d = go.Figure()
        structure_3d.add_trace(go.Scatter3d(
            x=particle_cloud["x"], y=particle_cloud["y"], z=particle_cloud["z"],
            mode="markers",
            name="Hydration-product shell",
            marker={
                "size": shell_sizes,
                "color": snapshot_alpha,
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": max(reaction_capacity(embedded_run["mix"]), 1e-8),
                "opacity": 0.24,
                "line": {"width": 0},
            },
            customdata=np.column_stack([
                particle_cloud["diameter_um"],
                np.full(len(particle_cloud), snapshot_alpha),
            ]),
            hovertemplate=(
                "Hydration-product shell<br>Initial particle D=%{customdata[0]:.1f} µm"
                "<br>Reaction degree=%{customdata[1]:.2f}<extra></extra>"
            ),
        ))
        structure_3d.add_trace(go.Scatter3d(
            x=particle_cloud["x"], y=particle_cloud["y"], z=particle_cloud["z"],
            mode="markers",
            name="Unreacted precursor core",
            marker={
                "size": core_sizes,
                "color": particle_cloud["diameter_um"],
                "colorscale": "YlOrBr",
                "opacity": 0.82,
                "line": {"color": "#4e342e", "width": 0.5},
                "colorbar": {"title": "Initial D (µm)", "x": 1.02},
            },
            hovertemplate=(
                "Unreacted precursor core<br>Initial particle D=%{marker.color:.1f} µm"
                "<extra></extra>"
            ),
        ))
        structure_3d.update_layout(
            title=(
                f"Statistical 3-D hydrate-shell structure at {snapshot_age:g} h "
                f"(α={snapshot_alpha:.2f})"
            ),
            scene={
                "xaxis_title": "Normalized X",
                "yaxis_title": "Normalized Y",
                "zaxis_title": "Normalized Z",
                "aspectmode": "cube",
            },
            legend={"orientation": "h", "y": 1.02, "x": 0.0},
            margin={"l": 0, "r": 0, "b": 0, "t": 70},
            height=700,
        )
        st.plotly_chart(structure_3d, width="stretch")
        st.caption(
            "Transparent colored spheres represent modeled hydration-product shells; brown "
            "inner spheres represent unreacted precursor cores. Rotate, zoom and hover to "
            "inspect the structure. This is a statistical conceptual visualization driven by "
            "the modeled reaction degree, not measured 3-D tomography or a phase-resolved "
            "CEMHYD3D reconstruction."
        )

        st.markdown("#### Standard-age cross-model summary")
        show_dataframe(
            embedded_run["summary"].round(5),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Selected-model raw output")
        show_dataframe(result, width="stretch", hide_index=True)

        st.markdown("#### Hydration-model sensitivity")
        st.caption(
            "This section perturbs mixture, curing, fineness, particle size and reactive "
            "fractions while preserving binder mass balance. The result is specific to the "
            "selected reduced-order model, output and age."
        )
        reference_mix = pd.Series({
            key: _get(embedded_run["mix"], key, 0.0)
            for key in HYDRATION_SENSITIVITY_INPUTS
        }, dtype=float)
        reference_binder_total = float(reference_mix[BINDER_COMPONENTS].sum())
        if reference_binder_total > 1e-12:
            reference_mix.loc[BINDER_COMPONENTS] = (
                100.0 * reference_mix[BINDER_COMPONENTS] / reference_binder_total
            )
        reference_units = {
            **{key: "wt.% of normalized dry binder" for key in BINDER_COMPONENTS},
            "WB": "mass ratio",
            "CuringTemp_C": "°C",
            "Blaine_m2kg": "m²/kg",
            "D50_um": "µm",
            "ReactiveSlag_frac": "fraction",
            "ReactiveFlyAsh_frac": "fraction",
        }
        st.info(
            f"Reference mixture = experimental mixture ‘{embedded_run['source']}’ used for the "
            f"current {embedded_run['model']} simulation, including the fineness, particle-size "
            "and reactive-fraction values entered above. Before sensitivity calculations, its "
            "six dry-binder components are normalized to a total of 100 wt.%. Every sensitivity "
            "bar is calculated from a fresh copy of this same reference; perturbations are not "
            "accumulated between variables."
        )
        with st.expander("Reference mixture used in sensitivity calculations", expanded=True):
            reference_table = pd.DataFrame({
                "Input": reference_mix.index,
                "Reference value": reference_mix.values,
                "Unit / basis": [reference_units.get(key, "-") for key in reference_mix.index],
            })
            show_dataframe(
                reference_table.round({"Reference value": 5}),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Original six-component binder total before normalization: "
                f"{reference_binder_total:.5g}. Normalized binder total used by sensitivity: "
                f"{reference_mix[BINDER_COMPONENTS].sum():.5g} wt.%. The selected closure rule "
                "controls how companion binder components change only when one binder component "
                "is perturbed."
            )
        hydration_numeric_outputs = [
            column for column in result.select_dtypes(include=[np.number]).columns
            if column != "Time_h"
        ]
        hs_col1, hs_col2, hs_col3 = st.columns(3)
        with hs_col1:
            hydration_sensitivity_output = st.selectbox(
                "Hydration output for sensitivity",
                hydration_numeric_outputs,
                index=(
                    hydration_numeric_outputs.index("HybridStrength_MPa")
                    if "HybridStrength_MPa" in hydration_numeric_outputs
                    else 0
                ),
                key="hydration_sensitivity_output",
            )
        with hs_col2:
            hydration_sensitivity_age = st.number_input(
                "Sensitivity age (h)",
                min_value=0.1,
                max_value=float(max_simulation_age),
                value=float(min(max_simulation_age, 72.0)),
                step=1.0,
                key="hydration_sensitivity_age",
            )
        with hs_col3:
            hydration_balance_mode = st.selectbox(
                "Hydration sensitivity closure",
                SENSITIVITY_BALANCE_OPTIONS,
                key="hydration_sensitivity_balance",
            )

        if st.button(
            "Run hydration-model sensitivity",
            key="run_hydration_sensitivity",
        ):
            hs_summary, hs_detail = hydration_sensitivity_analysis(
                json.dumps(embedded_run["mix"], sort_keys=True),
                embedded_run["model"],
                hydration_sensitivity_output,
                hydration_sensitivity_age,
                balance_mode=hydration_balance_mode,
            )
            st.session_state.hydration_sensitivity_result = {
                "model": embedded_run["model"],
                "output": hydration_sensitivity_output,
                "age": hydration_sensitivity_age,
                "balance_mode": hydration_balance_mode,
                "summary": hs_summary,
                "detail": hs_detail,
            }

        hs_state = st.session_state.get("hydration_sensitivity_result")
        if (
            hs_state
            and hs_state.get("model") == embedded_run["model"]
            and hs_state.get("output") == hydration_sensitivity_output
            and float(hs_state.get("age")) == float(hydration_sensitivity_age)
            and hs_state.get("balance_mode") == hydration_balance_mode
        ):
            hs_summary = hs_state["summary"]
            hs_detail = hs_state["detail"]
            if hs_summary.empty:
                st.warning("No valid hydration sensitivity perturbations were generated.")
            else:
                hs_fig = px.bar(
                    hs_summary,
                    x="Relative_range_effect",
                    y="Variable",
                    color="Direction",
                    orientation="h",
                    hover_data=[
                        "Low", "Reference", "High",
                        "Output_at_low", "Baseline_output", "Output_at_high",
                        "Local_normalized_slope", "Nonlinearity_index",
                        "Monotonic_fraction", "Mechanistic_interpretation",
                    ],
                    title=(
                        f"{embedded_run['model']} sensitivity of "
                        f"{hydration_sensitivity_output} at {hydration_sensitivity_age:g} h"
                    ),
                )
                hs_fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(hs_fig, width="stretch")

                st.markdown("#### Response relative to the reference mixture")
                st.info(
                    "This chart changes one selected input at a time while applying the chosen "
                    "binder-balance rule. A bar above zero means the modeled output is higher "
                    "than the reference-mixture prediction; a bar below zero means it is lower. "
                    "The vertical value is an absolute model-output difference, not a percentage. "
                    "Range position −100% is the tested lower input bound, 0% is the reference "
                    "mixture, and +100% is the tested upper bound."
                )
                response_variable = st.selectbox(
                    "Input variable shown in the response chart",
                    hs_summary["Variable"].tolist(),
                    key="hydration_response_variable",
                )
                response_data = hs_detail.loc[
                    hs_detail["Variable"] == response_variable
                ].copy()
                response_data["Response"] = np.where(
                    response_data["Change_from_reference"] >= 0,
                    "Increase from reference",
                    "Decrease from reference",
                )
                hs_response = px.bar(
                    response_data,
                    x="Range_position_pct",
                    y="Change_from_reference",
                    color="Response",
                    color_discrete_map={
                        "Increase from reference": "#2e7d32",
                        "Decrease from reference": "#c62828",
                    },
                    hover_data={
                        "Variable_value": ":.5g",
                        "Prediction": ":.5g",
                        "Balance_rule": True,
                        "Range_position_pct": ":.0f",
                        "Change_from_reference": ":.5g",
                        "Response": False,
                    },
                    title=(
                        f"{hydration_sensitivity_output} response to {response_variable} "
                        "relative to the reference mixture"
                    ),
                    labels={
                        "Range_position_pct": "Input range position (%)",
                        "Change_from_reference": "Change from reference prediction",
                    },
                )
                hs_response.add_hline(y=0.0, line_color="black", line_width=1)
                hs_response.update_layout(
                    bargap=0.18,
                    hovermode="x unified",
                    legend_title_text="Modeled response",
                )
                st.plotly_chart(hs_response, width="stretch")
                st.caption(
                    f"Reference prediction: {response_data['Prediction'].iloc[(response_data['Range_position_pct'].abs()).argmin()]:.5g} "
                    f"for {hydration_sensitivity_output} at {hydration_sensitivity_age:g} h. "
                    "Use the bar pattern to compare direction and magnitude; use the sensitivity "
                    "table below to review nonlinearity and monotonicity."
                )
                show_dataframe(
                    hs_summary.round(5),
                    width="stretch",
                    hide_index=True,
                )
                st.warning(
                    "A large sensitivity can arise from a model prior or empirical coefficient. "
                    "Confirm the direction using calorimetry, XRD/TG, pore structure and strength "
                    "before changing the formulation."
                )


        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download hydration result CSV",
                result.to_csv(index=False, float_format="%.2f").encode("utf-8-sig"),
                f"{embedded_run['source']}_{embedded_run['model']}_hydration.csv".replace("/", "-"),
                "text/csv",
            )
        with export_col2:
            st.download_button(
                "Download GEMS/CemGEMS recipe JSON",
                json.dumps(embedded_run["recipe"], ensure_ascii=False, indent=2).encode("utf-8"),
                f"{embedded_run['source']}_gems_recipe.json",
                "application/json",
            )

with tabs[7]:
    st.subheader("Candidate chemical state → engineering performance")
    st.caption(
        "This window connects each candidate's predicted XRD/TG/DTG state to its predicted "
        "engineering results. Links are measured-data associations used as screening evidence; "
        "the performance surrogate remains independently calculated from mixture/process inputs."
    )

    linkage_candidates = st.session_state.candidates
    if linkage_candidates is None or linkage_candidates.empty:
        st.info("Generate a candidate batch in 3 Decide to activate this connection view.")
    else:
        linkage = chemical_performance_linkage(df)
        selected_link_id = st.selectbox(
            "Candidate mixture", linkage_candidates["CandidateID"].tolist(),
            key="chemistry_performance_candidate",
        )
        selected_link_row = linkage_candidates.loc[
            linkage_candidates["CandidateID"] == selected_link_id
        ].iloc[0]

        st.markdown("#### 1. Same-candidate prediction chain")
        mix_col, arrow_col1, chemistry_col, arrow_col2, performance_col = st.columns(
            [1.35, 0.25, 1.7, 0.25, 1.7]
        )
        with mix_col:
            st.markdown("**Candidate mixture**")
            st.write(f"**{selected_link_id}**")
            st.caption(str(selected_link_row.get("Purpose", "Screening candidate")))
            st.metric("W/B", f"{selected_link_row['WB']:.3f}")
            st.metric("Curing", f"{selected_link_row['CuringTemp_C']:.0f} °C")
        with arrow_col1:
            st.markdown("## →")
        with chemistry_col:
            st.markdown("**Predicted chemical state**")
            chemical_groups = {
                "XRD": XRD_TARGETS,
                "TG": TG_TARGETS,
                "DTG": DTG_TARGETS,
            }
            available_chem_count = 0
            for group_name, group_targets in chemical_groups.items():
                st.markdown(f"**{group_name}**")
                group_rows = []
                for chemical in group_targets:
                    value = selected_link_row.get(f"Pred_{chemical}", np.nan)
                    if pd.isna(value):
                        continue
                    std = selected_link_row.get(f"Std_{chemical}", np.nan)
                    group_rows.append({
                        "Descriptor": chemical.replace(f"{group_name}_", "").replace("_", " "),
                        "Predicted": float(value),
                        "Std": float(std) if pd.notna(std) else np.nan,
                    })
                available_chem_count += len(group_rows)
                if group_rows:
                    show_dataframe(
                        pd.DataFrame(group_rows).round(2),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption(
                        f"{group_name}: no fitted targets (at least eight valid measured rows required)."
                    )
            if available_chem_count == 0:
                st.warning("No chemical surrogate has enough measured rows for this candidate.")
        with arrow_col2:
            st.markdown("## →")
        with performance_col:
            st.markdown("**Predicted engineering performance**")
            available_perf = [
                c for c in PERFORMANCE_TARGETS
                if pd.notna(selected_link_row.get(f"Pred_{c}", np.nan))
            ]
            for performance in available_perf:
                value = selected_link_row[f"Pred_{performance}"]
                std = selected_link_row.get(f"Std_{performance}", np.nan)
                suffix = f" ± {std:.2f}" if pd.notna(std) and std > 0 else ""
                st.write(f"{performance.replace('_', ' ')}: **{value:.2f}{suffix}**")

        st.markdown("#### 2. Evidence linking chemical state and performance")
        if linkage.empty:
            st.warning(
                "At least eight paired chemical and performance measurements are required. "
                "Add matched XRD/TG/DTG and engineering test results."
            )
        else:
            evidence = candidate_linkage_evidence(selected_link_row, df, linkage)
            heatmap_data = linkage.pivot(
                index="Chemical_state", columns="Engineering_performance", values="Spearman_rho"
            )
            heatmap = px.imshow(
                heatmap_data, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                aspect="auto", labels={"color": "Spearman ρ"},
                title="Measured-data chemical–performance association map",
            )
            heatmap.update_layout(height=max(430, 24 * len(heatmap_data)))
            st.plotly_chart(heatmap, width="stretch")

            if not evidence.empty:
                evidence_plot = evidence.head(16).copy()
                evidence_plot["Link"] = (
                    evidence_plot["Chemical_state"].str.replace("_", " ")
                    + " → "
                    + evidence_plot["Engineering_performance"].str.replace("_", " ")
                )
                direction_fig = px.bar(
                    evidence_plot.sort_values("Directional_link_index"),
                    x="Directional_link_index", y="Link", orientation="h",
                    color="Measured_Spearman_rho", color_continuous_scale="RdBu_r",
                    range_color=[-1, 1],
                    hover_data=["Candidate_chemical_prediction", "Chemical_deviation_z",
                                "Measured_Spearman_rho", "Paired_experiments"],
                    title="Candidate-specific linkage: chemical deviation × measured association",
                )
                direction_fig.add_vline(x=0, line_color="black", line_width=1)
                st.plotly_chart(direction_fig, width="stretch")
                st.caption(
                    "Positive bars indicate association with a higher performance-metric value; "
                    "negative bars indicate a lower value. Higher cost, CO₂, absorption or "
                    "expansion is not necessarily desirable."
                )
                show_dataframe(evidence.round(4), width="stretch", hide_index=True)
                st.download_button(
                    "Download selected candidate linkage CSV",
                    evidence.to_csv(index=False, float_format="%.5f").encode("utf-8-sig"),
                    f"{selected_link_id}_chemical_performance_linkage.csv", "text/csv",
                )

        st.warning(
            "Interpretation boundary: association does not establish a hydration mechanism or "
            "causal path. Confirm important links with matched-age calorimetry, XRD/TG, pore "
            "structure and engineering tests before approving the candidate."
        )

st.divider()
st.caption(
    "OODA-C3 v3.2. Demonstration data are synthetic. "
    "Replace proxy chemistry, price and carbon factors with verified project-specific values."
)
