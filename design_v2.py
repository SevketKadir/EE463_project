"""EE463 pre-report (v2) design calculations and simulations.

Two-phase interleaved synchronous buck, LTC3890-1 controller (both channels paralleled
onto one 12 V output), two discrete 33 uH inductors (Coilcraft XAL1510-333MED),
four DMT10H015LPS logic-level MOSFETs, 250 kHz per phase.

Outputs: figures/*.png and results.json (all numbers quoted in the report come from here).

Sources of the parameters are tagged in the comments:
  [DS]   manufacturer datasheet value
  [DSc]  read from a typical-performance curve of the datasheet
  [A]    assumption / engineering estimate (listed in the report, to be verified)
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.formatter.useoffset": False})
C1, C2, C3, C4, CG = "#2a6fb0", "#d9822b", "#3a9a5b", "#8c6bb1", "#555555"

# ------------------------------------------------------------------ specification
Vo, Io_nom, Io_ovl = 12.0, 5.0, 7.5
VIN = [20.0, 24.0, 36.0, 48.0]
R_nom, R_ovl = Vo / Io_nom, Vo / Io_ovl          # 2.4 ohm, 1.6 ohm
N_PH = 2

# ------------------------------------------------------------------ LTC3890-1 [DS]
fs = 250e3                    # per-phase switching frequency (design choice)
T = 1 / fs
VREF = 0.800                  # 0.792 ... 0.808 V
VSENSE_MAX = (0.064, 0.075, 0.085)   # min / typ / max, fixed on the -1 version (no ILIM pin)
GM_EA = 2e-3                  # error-amplifier transconductance per channel
GM_TOT = 2 * GM_EA            # ITH1/ITH2 and VFB1/VFB2 tied -> two amplifiers in parallel
I_SS = 1e-6                   # TRACK/SS pull-up per pin (0.7 ... 1.4 uA)
VRUN_ON = (1.21, 1.25)        # RUN1 / RUN2 rising thresholds (typ)
I_RUN = 7e-6 + 0.5e-6         # RUN1 + RUN2 internal pull-up currents (pins tied)
T_ON_MIN = 95e-9
T_DEAD = 30e-9                # TG/BG adaptive delay, each edge (typ, 3300 pF)
R_TG_UP, R_TG_DN = 2.5, 1.5   # driver output resistances
V_INTVCC = 5.1
# ITH -> current-sense-threshold transfer, forced continuous mode [DSc]:
#   VSENSE = 75 mV/V * (VITH - 0.4 V); -30 mV at VITH = 0 V, 75 mV at 1.4 V
G_CS, V_ITH0 = 0.075, 0.40
# Maximum sense threshold vs duty cycle (typical curve) [DSc], used as a ratio to 75 mV
_D_CURVE = np.array([0, .15, .20, .25, .30, .40, .50, .60, .70, .80, .90])
_V_CURVE = np.array([75, 75, 74, 73.2, 73, 72.6, 72, 71.3, 70.5, 70, 69.2])


def vsense_derate(D):
    return float(np.interp(D, _D_CURVE, _V_CURVE)) / 75.0


# ------------------------------------------------------------------ power stage
L = 33e-6                     # XAL1510-333MED, +/-20 % [DS]
DCR = (18.7e-3, 20.0e-3)      # typ / max at 25 C [DS]
L_ISAT, L_IRMS20 = 16.7, 8.6  # A [DS]
TEMP_CU = 1.10                # winding about +25 C above ambient [A]
RS = 12e-3                    # proposed sense resistor (see section 6 of the report)
# DMT10H015LPS [DS]: RDS(on) 18 typ / 25 max mOhm at VGS = 4.5 V (ID = 5 A)
RDS45 = (18e-3, 25e-3)
TEMP_FET = (1.20, 1.30)       # hot factor, Tj ~ 60-80 C [A, from the datasheet trend]
QGS, QGD, RG_INT = 6.9e-9, 5.1e-9, 0.75   # [DS] (Qgd at VDS = 50 V)
COSS50 = 261e-12              # [DS] at VDS = 50 V
QG5 = 17e-9                   # total gate charge at 5.1 V [A, from the Qg curve]
V_PLATEAU = 3.0               # Miller plateau at 2-4 A [A]
QRR10 = 51.9e-9               # [DS] at IF = 10 A, di/dt = 100 A/us
VSD = 0.8                     # body diode at 2-4 A [A]
R_PCB = (4e-3, 6e-3)          # board copper + connectors in the output path [A]
I_IC = 3e-3                   # LTC3890-1 operating current drawn from EXTVCC [A]

# capacitors
C_CER_OUT, ESR_CER_OUT = 4 * 22e-6 * 0.5, 3e-3 / 4    # 4 x 22 uF/25 V X7R 1210, ~50 % at 12 V [A]
C_EL_OUT, ESR_EL_OUT = 470e-6, 0.10                  # 470 uF/25 V low-ESR electrolytic [A]
C_CER_IN = 4 * 4.7e-6 * 0.55                         # 4 x 4.7 uF/100 V X7S 1210, ~55 % at 48 V [A]
C_EL_IN, ESR_EL_IN = 100e-6, 0.3                     # 100 uF/63 V electrolytic [A]

# feedback, frequency, RUN and soft-start networks (0805 parts)
RB, RA = 140e3, 10e3
R_FREQ = 42.2e3
R_RUN_TOP, R_RUN_BOT = 100e3, 7.5e3
C_SS = 22e-9

res = {"spec": dict(Vo=Vo, Io=Io_nom, Io_ovl=Io_ovl, VIN=VIN),
       "design": dict(fs=fs, L=L, RS=RS, RB=RB, RA=RA, R_FREQ=R_FREQ, C_SS=C_SS,
                      C_out_eff=C_CER_OUT + C_EL_OUT, C_in_cer_eff=C_CER_IN)}

# ------------------------------------------------------------------ 1. operating point
def ripple_phase(Vin, L_=L):
    D = Vo / Vin
    return Vo * (1 - D) / (L_ * fs)


def ripple_sum(Vin, L_=L):
    """peak-to-peak ripple of i1 + i2 (two phases, 180 deg), ideal duty"""
    D = Vo / Vin
    dI = ripple_phase(Vin, L_)
    return dI * (1 - 2 * D) / (1 - D) if D <= 0.5 else dI * (2 * D - 1) / D


rows = []
for Vin in VIN:
    D = Vo / Vin
    dI = ripple_phase(Vin)
    row = dict(Vin=Vin, D=D, dI_ph=dI, dI_ratio=dI / (Io_nom / 2), dI_sum=abs(ripple_sum(Vin)))
    for tag, Io in (("5", Io_nom), ("7.5", Io_ovl)):
        Iph = Io / 2
        row["Ipk_" + tag] = Iph + dI / 2
        row["Irms_" + tag] = float(np.sqrt(Iph ** 2 + dI ** 2 / 12))
    rows.append(row)
res["ripple_table"] = rows

# inductance sweep at 48 V
res["L_sweep"] = [dict(L_uH=l, dI48=ripple_phase(48, l * 1e-6), pct=100 * ripple_phase(48, l * 1e-6) / 2.5,
                       Ipk=2.5 + ripple_phase(48, l * 1e-6) / 2) for l in (15, 22, 33, 47, 68)]
# inductance tolerance (-20 %)
res["L_tol"] = dict(Lmin=0.8 * L, dI48=ripple_phase(48, 0.8 * L), Ipk_5=2.5 + ripple_phase(48, 0.8 * L) / 2,
                    Ipk_75=3.75 + ripple_phase(48, 0.8 * L) / 2)

# ------------------------------------------------------------------ 2. frequency, divider, RUN, SS
f_rfreq = 105e3 + (R_FREQ - 25e3) * (440e3 - 105e3) / (65e3 - 25e3)   # straight line between [DS] anchors
Vo_set = VREF * (1 + RB / RA)
Vo_lo = 0.792 * (1 + RB * 0.999 / (RA * 1.001))
Vo_hi = 0.808 * (1 + RB * 1.001 / (RA * 0.999))
R_par = R_RUN_TOP * R_RUN_BOT / (R_RUN_TOP + R_RUN_BOT)
Vin_on = (VRUN_ON[0] - I_RUN * R_par) * (R_RUN_TOP + R_RUN_BOT) / R_RUN_BOT
t_ss = C_SS * VREF / (2 * I_SS)          # two pull-up sources charge the shared capacitor
res["settings"] = dict(f_rfreq=f_rfreq, Vo_set=Vo_set, Vo_lo=Vo_lo, Vo_hi=Vo_hi,
                       tol_pct=100 * (Vo_hi - Vo) / Vo, Vin_on=Vin_on, Vin_hyst=0.05 * (R_RUN_TOP + R_RUN_BOT) / R_RUN_BOT,
                       t_ss=t_ss, Dmin=Vo / 48, t_on_48=Vo / 48 / fs, I_div=Vo / (RA + RB))

# ------------------------------------------------------------------ 3. current limit
def current_limit(Rs, Vin, which):
    """total output current at which both phases hit the peak-current clamp"""
    D = Vo / Vin
    v = VSENSE_MAX[which] * vsense_derate(D)
    return 2 * (v / Rs - ripple_phase(Vin) / 2)


cl = []
for Rs_ in (10e-3, 12e-3, 15e-3):
    entry = dict(Rs_mOhm=Rs_ * 1e3)
    for Vin in VIN:
        entry[f"min_{int(Vin)}"] = current_limit(Rs_, Vin, 0)
        entry[f"typ_{int(Vin)}"] = current_limit(Rs_, Vin, 1)
        entry[f"max_{int(Vin)}"] = current_limit(Rs_, Vin, 2)
    dI_sc = T_ON_MIN * 48 / L
    entry["Isc"] = [2 * (0.45 * VSENSE_MAX[k] / Rs_ - dI_sc / 2) for k in range(3)]
    entry["P_rs_5"] = 2 * (2.5 ** 2 + ripple_phase(48) ** 2 / 12) * Rs_
    entry["P_rs_75"] = 2 * (3.75 ** 2 + ripple_phase(48) ** 2 / 12) * Rs_
    entry["Rmax_for_7A5"] = min(VSENSE_MAX[0] * vsense_derate(Vo / v) / (3.75 + ripple_phase(v) / 2) for v in VIN)
    cl.append(entry)
res["current_limit"] = cl

# ------------------------------------------------------------------ 4. loss model
def losses(Vin, Io, case=0):
    """case 0 = typical, case 1 = worst (max RDS, max DCR, x2 on estimated dynamic terms)"""
    k = 1.0 if case == 0 else 2.0
    Iph = Io / 2
    Rds = RDS45[case] * TEMP_FET[case]
    Rl = DCR[case] * TEMP_CU
    Rph = Rds + Rl + RS
    D = (Vo + Iph * Rph) / Vin                                  # duty with conduction drops
    dI = Vo * (1 - D) / (L * fs) if D < 1 else 0
    Irms = np.sqrt(Iph ** 2 + dI ** 2 / 12)
    I_on, I_off = Iph - dI / 2, Iph + dI / 2                    # HS turn-on (valley) / turn-off (peak)
    p = {}
    p["HS conduction"] = 2 * D * Irms ** 2 * Rds
    p["LS conduction"] = 2 * (1 - D) * Irms ** 2 * Rds
    Qsw = QGD + QGS / 2
    ig_on = (V_INTVCC - V_PLATEAU) / (R_TG_UP + RG_INT)
    ig_off = V_PLATEAU / (R_TG_DN + RG_INT)
    t_on, t_off = Qsw / ig_on, Qsw / ig_off
    p["HS switching"] = 2 * 0.5 * Vin * (I_on * t_on + I_off * t_off) * fs * k
    Qoss = 2 * COSS50 * 50 * np.sqrt(Vin / 50)                  # C ~ V^-1/2 -> Q = 2 C(V) V
    p["Coss"] = 2 * Qoss * Vin * fs * k
    p["Qrr"] = 2 * QRR10 * max(I_on, 0) / 10 * Vin * fs * k
    p["Dead time"] = 2 * VSD * (abs(I_on) + I_off) * T_DEAD * fs * k
    p["Inductor Cu"] = 2 * Irms ** 2 * Rl
    p["Inductor core"] = 2 * 0.05 * (dI / ripple_phase(48)) ** 2.2 * k   # 50 mW each at 48 V [A]
    p["Sense resistors"] = 2 * Irms ** 2 * RS
    p["Gate drive + IC (EXTVCC)"] = (4 * QG5 * fs + I_IC) * Vo * (1 if case == 0 else 1.3)
    Icin = cin_rms(Vin, Io)
    p["Caps + PCB"] = Io ** 2 * R_PCB[case] + Icin ** 2 * 0.004
    P = sum(p.values())
    Pout = Vo * Io
    fet_hs = (p["HS conduction"] + p["HS switching"] + p["Coss"] / 2 + p["Qrr"]) / 2
    fet_ls = (p["LS conduction"] + p["Coss"] / 2 + p["Dead time"]) / 2
    return dict(D=D, dI=dI, Irms=Irms, t_on=t_on, t_off=t_off, parts=p, P=P, eff=Pout / (Pout + P),
                fet_hs=fet_hs, fet_ls=fet_ls)


def cin_rms(Vin, Io):
    """RMS ripple current in the input capacitors (flat-top phase currents, 180 deg)"""
    D = Vo / Vin
    t = np.linspace(0, 1, 4000, endpoint=False)
    s1 = (t < D).astype(float)
    s2 = (((t + 0.5) % 1) < D).astype(float)
    iin = (s1 + s2) * Io / 2
    return float(np.sqrt(np.mean((iin - iin.mean()) ** 2)))


loss_rows = []
for Vin in VIN:
    for Io in (Io_nom, Io_ovl):
        lt, lw = losses(Vin, Io, 0), losses(Vin, Io, 1)
        loss_rows.append(dict(Vin=Vin, Io=Io, typ=lt, worst=lw))
res["losses"] = loss_rows
eta_typ = [r["typ"]["eff"] for r in loss_rows if r["Io"] == 5]
eta_w = [r["worst"]["eff"] for r in loss_rows if r["Io"] == 5]
res["eta_mean"] = dict(typ=float(np.mean(eta_typ)), worst=float(np.mean(eta_w)),
                       bonus_typ=100 * (np.mean(eta_typ) - 0.8), bonus_worst=100 * (np.mean(eta_w) - 0.8))

# IC power with and without EXTVCC (gate-drive current through the INTVCC LDO)
I_drive = 4 * QG5 * fs + I_IC
res["ic_power"] = [dict(Vin=v, from_vin=I_drive * (v - V_INTVCC), from_extvcc=I_drive * (Vo - V_INTVCC))
                   for v in VIN]
res["ic_power_note"] = dict(I_drive=I_drive, theta_ja=90)

# thermal (worst-case loss, R_thJA assumed 40 C/W for a 5x6 mm power package on ~1 in^2 2-oz copper)
TH_FET = 40.0
th = []
for r in loss_rows:
    lw = r["worst"]
    th.append(dict(Vin=r["Vin"], Io=r["Io"], P_hs=lw["fet_hs"], P_ls=lw["fet_ls"],
                   dT_hs=lw["fet_hs"] * TH_FET, dT_ls=lw["fet_ls"] * TH_FET,
                   P_L=lw["parts"]["Inductor Cu"] / 2 + lw["parts"]["Inductor core"] / 2))
res["thermal"] = th

# input / output capacitor checks
def cin_dv(Vin, Io):
    """peak-to-peak voltage ripple on the input ceramics (bulk electrolytic ignored)"""
    D = Vo / Vin
    t = np.linspace(0, 1, 4000, endpoint=False)
    iin = ((t < D).astype(float) + (((t + 0.5) % 1) < D).astype(float)) * Io / 2
    q = np.cumsum(iin.mean() - iin) * (T / len(t))
    return float(np.ptp(q) / C_CER_IN)


res["cin"] = [dict(Vin=v, single=float(Io_nom * np.sqrt(Vo * (v - Vo)) / v), two=cin_rms(v, Io_nom),
                   two_ovl=cin_rms(v, Io_ovl), dv_pp=cin_dv(v, Io_nom), dv_pp_ovl=cin_dv(v, Io_ovl))
              for v in VIN]

# ------------------------------------------------------------------ 5. small-signal loop (peak-current mode)
def Zout(s, Rl):
    zc = ESR_CER_OUT + 1 / (s * C_CER_OUT)
    ze = ESR_EL_OUT + 1 / (s * C_EL_OUT)
    return 1 / (1 / Rl + 1 / zc + 1 / ze)


def He(s, Vin):
    """sampling gain (Ridley), with an equivalent compensating ramp Se = 0.5 * off-slope"""
    D = Vo / Vin
    Sn = (Vin - Vo) / L * RS
    Se = 0.5 * Vo / L * RS
    mc = 1 + Se / Sn
    Q = 1 / (np.pi * (mc * (1 - D) - 0.5))
    wn = np.pi * fs
    return 1 / (1 + s / (wn * Q) + s ** 2 / wn ** 2), Se


K_I = N_PH * G_CS / RS          # A of total inductor current per V of ITH
H_FB = RA / (RA + RB)


def Zc(s, comp):
    Rc, Cc, Cc2 = comp
    return (1 + s * Rc * Cc) / (s * (Cc + Cc2) * (1 + s * Rc * Cc * Cc2 / (Cc + Cc2)))


def loop(f, Vin, comp, Rl=R_nom):
    s = 2j * np.pi * f
    return H_FB * GM_TOT * Zc(s, comp) * K_I * Zout(s, Rl) * He(s, Vin)[0]


def design_comp(fc, fz_ratio=4.0, fp=None, Vin=24.0):
    s = 2j * np.pi * fc
    plant = abs(H_FB * GM_TOT * K_I * Zout(s, R_nom) * He(s, Vin)[0])
    Rc = 1 / plant
    Cc = 1 / (2 * np.pi * Rc * fc / fz_ratio)
    Cc2 = 1 / (2 * np.pi * Rc * (fp if fp else 5 * fc))
    return Rc, Cc, Cc2


def margins(Vin, comp, Rl=R_nom):
    f = np.logspace(1, np.log10(fs / 2), 4000)
    Tm = loop(f, Vin, comp, Rl)
    mag = np.abs(Tm)
    i = np.where(mag < 1)[0][0]
    fc = float(np.interp(1, [mag[i], mag[i - 1]], [f[i], f[i - 1]]))
    ph = float(np.degrees(np.angle(loop(fc, Vin, comp, Rl))))
    return fc, 180 + ph


def gain_margin(Vin, comp, Rl=R_nom):
    f = np.logspace(3, np.log10(fs / 2), 4000)
    Tm = loop(f, Vin, comp, Rl)
    ph = np.degrees(np.unwrap(np.angle(Tm)))
    j = np.where(ph < -180)[0]
    if len(j) == 0:
        return float(-20 * np.log10(abs(Tm[-1]))), float(f[-1])
    return float(-20 * np.log10(abs(Tm[j[0]]))), float(f[j[0]])


E12 = np.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2])


def e12(x):
    d = 10 ** np.floor(np.log10(x))
    return float(E12[np.argmin(abs(E12 * d - x))] * d)


compA_raw = design_comp(20e3, 4.0, 100e3)
compB_raw = design_comp(5e3, 4.0, 40e3)
compA = tuple(e12(x) for x in compA_raw)
compB = tuple(e12(x) for x in compB_raw)
res["comp"] = {}
for name, cp, raw in (("A", compA, compA_raw), ("B", compB, compB_raw)):
    d = dict(Rc=cp[0], Cc=cp[1], Cc2=cp[2], raw=list(raw), fz=1 / (2 * np.pi * cp[0] * cp[1]),
             fp=(cp[1] + cp[2]) / (2 * np.pi * cp[0] * cp[1] * cp[2]))
    for Vin in VIN:
        fc, pm = margins(Vin, cp)
        d[f"fc_{int(Vin)}"], d[f"pm_{int(Vin)}"] = fc, pm
        d[f"gm_{int(Vin)}"], d[f"f180_{int(Vin)}"] = gain_margin(Vin, cp)
    fc, pm = margins(24, cp, R_ovl)
    d["fc_24_ovl"], d["pm_24_ovl"] = fc, pm
    res["comp"][name] = d
res["loop_consts"] = dict(K_I=K_I, H_FB=H_FB, GM_TOT=GM_TOT, Se=He(1j, 24)[1],
                          f_esr_el=1 / (2 * np.pi * ESR_EL_OUT * C_EL_OUT),
                          f_esr_cer_vs_el=1 / (2 * np.pi * ESR_EL_OUT * C_CER_OUT))

# ------------------------------------------------------------------ 6. behavioural closed-loop switching simulation
def simulate(vin_fn, rload_fn, t_end, comp, Io0, Vin0, steps=400, rec=None, rec_every=1):
    """Cycle-by-cycle peak-current-mode model of the paralleled LTC3890-1 channels.
    Clock: phase 2 lags phase 1 by T/2. Top switch on at the clock edge, off when
    i*RS + ramp >= VSENSE command (= G_CS*(VITH-0.4), clamped at 75 mV). Forced continuous.
    Error amplifiers (gm_tot) drive the ITH network Rc+Cc // Cc2. Output: ceramic and
    electrolytic branches with ESR, resistive load. Dead time and gate drive are not modelled."""
    Rc, Cc, Cc2 = comp
    dt = T / steps
    Rph = RDS45[0] * TEMP_FET[0] + DCR[0] * TEMP_CU + RS
    Se = 0.5 * Vo / L * RS
    i = [Io0 / 2, Io0 / 2]
    D0 = (Vo + Io0 / 2 * Rph) / Vin0
    vith = V_ITH0 + (RS * (Io0 / 2 + ripple_phase(Vin0) / 2) + Se * max(0.0, D0 - 0.2) * T) / G_CS
    vcc = vith
    vce = vel = Vo
    hs = [False, False]
    pos_prev = [1.0, 1.0]
    gce, gel = 1 / ESR_CER_OUT, 1 / ESR_EL_OUT
    a, b = C_CER_OUT / dt, C_EL_OUT / dt
    Rl_cur, Minv = None, None
    out = []
    n = int(round(t_end / dt))
    rec_start = rec if rec is not None else 0.0
    for k in range(n):
        t = k * dt
        vin = vin_fn(t)
        Rl = rload_fn(t)
        if Rl != Rl_cur:
            Rl_cur = Rl
            G = 1 / Rl + gce + gel
            m11 = a + gce - gce * gce / G
            m22 = b + gel - gel * gel / G
            m12 = -gce * gel / G
            det = m11 * m22 - m12 * m12
            Minv = (m22 / det, -m12 / det, m11 / det, G)
        itot = i[0] + i[1]
        G = Minv[3]
        vo = (itot + vce * gce + vel * gel) / G
        vfb = vo * H_FB
        igm = GM_TOT * (VREF - vfb)
        vith += (igm - (vith - vcc) / Rc) / Cc2 * dt
        vcc += (vith - vcc) / (Rc * Cc) * dt
        vith = min(max(vith, 0.0), 2.2)
        vcmd = min(G_CS * (vith - V_ITH0), VSENSE_MAX[1])
        sw = [0.0, 0.0]
        for ph in range(2):
            pos = (t / T + 0.5 * ph) % 1.0
            if pos < pos_prev[ph]:
                hs[ph] = True
            pos_prev[ph] = pos
            if hs[ph]:
                ramp = Se * max(0.0, pos - 0.2) * T
                if i[ph] * RS + ramp >= vcmd or pos > 0.99:
                    hs[ph] = False
            sw[ph] = vin if hs[ph] else 0.0
            i[ph] += (sw[ph] - vo - i[ph] * Rph) / L * dt
        itot = i[0] + i[1]
        r1 = a * vce + itot * gce / G
        r2 = b * vel + itot * gel / G
        m22i, m12i, m11i = Minv[0], Minv[1], Minv[2]
        vce, vel = m22i * r1 + m12i * r2, m12i * r1 + m11i * r2
        if t >= rec_start and k % rec_every == 0:
            out.append((t, i[0], i[1], vo, float(hs[0]), float(hs[1]), vith, vin, vo / Rl))
    return np.array(out)


ss = {}
for Vin in VIN:
    for Io in (Io_nom, Io_ovl):
        r = simulate(lambda t, v=Vin: v, lambda t, R=Vo / Io: R, 1.6e-3, compA, Io, Vin, rec=1.6e-3 - 4 * T)
        w = r[-2 * 400:]
        iin = w[:, 4] * w[:, 1] + w[:, 5] * w[:, 2]
        key = f"{int(Vin)}_{Io}"
        ss[key] = r
        res.setdefault("sim_ss", []).append(dict(
            Vin=Vin, Io=Io, Vo_avg=float(w[:, 3].mean()), Vpp=float(np.ptp(w[:, 3])),
            dI1=float(np.ptp(w[:, 1])), dIsum=float(np.ptp(w[:, 1] + w[:, 2])),
            I1=float(w[:, 1].mean()), I2=float(w[:, 2].mean()), Ipk=float(max(w[:, 1].max(), w[:, 2].max())),
            D1=float(w[:, 4].mean()), Iin=float(iin.mean()),
            Icin_rms=float(np.sqrt(np.mean((iin - iin.mean()) ** 2))), vith=float(w[:, 6].mean())))
        print(f"SS Vin={Vin} Io={Io}: Vo={w[:, 3].mean():.4f} Vpp={np.ptp(w[:, 3])*1e3:.2f} mV "
              f"dI1={np.ptp(w[:, 1]):.3f} dIsum={np.ptp(w[:, 1] + w[:, 2]):.3f} D={w[:, 4].mean():.3f}")

# load steps 5 A -> 7.5 A -> 5 A, and a line step 24 V -> 36 V -> 24 V
T_STEP1, T_STEP2, T_END = 0.6e-3, 2.4e-3, 4.0e-3


def load_profile(t):
    return R_ovl if T_STEP1 <= t < T_STEP2 else R_nom


tr = {}
for Vin in (24.0, 48.0):
    for name, cp in (("A", compA), ("B", compB)):
        r = simulate(lambda t, v=Vin: v, load_profile, T_END, cp, Io_nom, Vin, steps=200, rec=0.3e-3, rec_every=4)
        tr[(Vin, name)] = r
        t, vo = r[:, 0], r[:, 3]


def step_metrics(r, t0, t1, settle_band=0.02):
    t, vo = r[:, 0], r[:, 3]
    pre = vo[(t > t0 - 0.2e-3) & (t < t0)].mean()
    seg = (t >= t0) & (t < t1)
    post = vo[(t > t1 - 0.3e-3) & (t < t1)].mean()
    dev = vo[seg] - post
    vmin, vmax = vo[seg].min(), vo[seg].max()
    # settling: last time |vo - post| > 0.5 % * 12 V (60 mV)
    out_band = np.where(np.abs(vo[seg] - post) > 0.06)[0]
    ts = (t[seg][out_band[-1]] - t0) if len(out_band) else 0.0
    return dict(pre=float(pre), post=float(post), vmin=float(vmin), vmax=float(vmax), ts=float(ts))


res["load_step"] = []
for (Vin, name), r in tr.items():
    up = step_metrics(r, T_STEP1, T_STEP2)
    dn = step_metrics(r, T_STEP2, T_END)
    res["load_step"].append(dict(Vin=Vin, comp=name, up=up, down=dn))
    print("load step", Vin, name, up, dn)

LS1, LS2 = 0.6e-3, 2.4e-3


def line_profile(t):
    return 36.0 if LS1 <= t < LS2 else 24.0


res["line_step"] = []
for name, cp in (("A", compA), ("B", compB)):
    r = simulate(line_profile, lambda t: R_nom, T_END, cp, Io_nom, 24.0, steps=200, rec=0.3e-3, rec_every=4)
    tr[("line", name)] = r
    up = step_metrics(r, LS1, LS2)
    dn = step_metrics(r, LS2, T_END)
    res["line_step"].append(dict(comp=name, up=up, down=dn))
    print("line step", name, up, dn)

# ------------------------------------------------------------------ 7. figures
# 7.1 ripple cancellation vs duty
Dg = np.linspace(0.2, 0.65, 300)
Vg = Vo / Dg
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(Dg, [ripple_phase(v) for v in Vg], color=C1, label="Per phase, ΔI$_L$")
ax[0].plot(Dg, [abs(ripple_sum(v)) for v in Vg], color=C3, label="Sum i$_{L1}$+i$_{L2}$")
ax[0].plot(Dg, [2 * ripple_phase(v) * 1 for v in Vg], color=C2, ls="--", label="Single phase, L/2 (same total)")
ax[0].set_xlabel("Duty cycle D"); ax[0].set_ylabel("Ripple current (A p-p)"); ax[0].legend(fontsize=7)
kd = [abs(ripple_sum(v)) / ripple_phase(v) for v in Vg]
ax[1].plot(Dg, kd, color=C4)
ax[1].set_xlabel("Duty cycle D"); ax[1].set_ylabel("ΔI$_{sum}$ / ΔI$_{phase}$")
for a_ in ax:
    a_.axvspan(12 / 48, 12 / 20, color="#cccccc", alpha=.3)
fig.tight_layout(); fig.savefig("figures/ripple_vs_D.png"); plt.close(fig)


def fig_wave(key, title, fname):
    r = ss[key][-2 * 400:]
    t = (r[:, 0] - r[0, 0]) * 1e6
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=True)
    ax[0].plot(t, r[:, 4] + 1.3, color=C1, lw=1, label="TG1 (phase 1)")
    ax[0].plot(t, r[:, 5], color=C2, lw=1, label="TG2 (phase 2)")
    ax[0].set_yticks([]); ax[0].legend(loc="upper right", fontsize=7, ncol=2); ax[0].set_title(title, fontsize=9)
    ax[1].plot(t, r[:, 1], color=C1, lw=1.2, label="i$_{L1}$")
    ax[1].plot(t, r[:, 2], color=C2, lw=1.2, label="i$_{L2}$")
    ax[1].plot(t, r[:, 1] + r[:, 2], color=C3, lw=1.2, label="i$_{L1}$+i$_{L2}$")
    ax[1].set_ylabel("Current (A)"); ax[1].legend(loc="upper right", fontsize=7, ncol=3)
    ax[2].plot(t, r[:, 3] * 1e3 - 12000, color=CG, lw=1.2)
    ax[2].set_ylabel("V$_{out}$ − 12 V (mV)"); ax[2].set_xlabel("Time (µs)")
    fig.savefig(f"figures/{fname}.png"); plt.close(fig)


for Vin in VIN:
    fig_wave(f"{int(Vin)}_5.0", f"Closed loop, V$_{{in}}$ = {int(Vin)} V, I$_o$ = 5 A", f"wave_{int(Vin)}V_5A")
fig_wave("20_7.5", "Closed loop, V$_{in}$ = 20 V, I$_o$ = 7.5 A (90 W)", "wave_20V_7A5")
fig_wave("48_7.5", "Closed loop, V$_{in}$ = 48 V, I$_o$ = 7.5 A (90 W)", "wave_48V_7A5")

# 7.2 current-limit window
fig, ax = plt.subplots(figsize=(6.0, 2.8))
vv = np.linspace(20, 48, 50)
for Rs_, col in ((10e-3, C1), (12e-3, C3), (15e-3, C2)):
    lo = [current_limit(Rs_, v, 0) for v in vv]
    ty = [current_limit(Rs_, v, 1) for v in vv]
    hi = [current_limit(Rs_, v, 2) for v in vv]
    ax.fill_between(vv, lo, hi, color=col, alpha=.18)
    ax.plot(vv, ty, color=col, label=f"R$_{{SENSE}}$ = {Rs_*1e3:.0f} mΩ")
ax.axhline(7.5, color="k", ls="--", lw=1); ax.text(20.5, 7.65, "7.5 A overload point", fontsize=7)
ax.set_xlabel("Input voltage (V)"); ax.set_ylabel("Output current limit (A)")
ax.legend(fontsize=7, loc="upper right"); ax.set_title("Two-phase current limit, LTC3890-1 (64/75/85 mV)", fontsize=9)
fig.savefig("figures/current_limit.png"); plt.close(fig)

# 7.3 loss breakdown
cats = list(loss_rows[0]["typ"]["parts"].keys())
pal = ["#2a6fb0", "#7aa6d6", "#d9822b", "#f0b97a", "#b5462f", "#8c6bb1", "#3a9a5b", "#9bcf9f", "#e0c341", "#555555",
       "#bbbbbb"]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bottom = np.zeros(4)
rows5 = [r for r in loss_rows if r["Io"] == 5]
for c, col in zip(cats, pal):
    vals = np.array([r["typ"]["parts"][c] for r in rows5])
    ax.bar([f"{int(r['Vin'])} V" for r in rows5], vals, bottom=bottom, color=col, label=c)
    bottom += vals
for x, (b_, r) in enumerate(zip(bottom, rows5)):
    ax.text(x, b_ + 0.03, f"η = {100 * r['typ']['eff']:.1f} %", ha="center", fontsize=7)
ax.set_ylabel("Loss (W)"); ax.set_title("Estimated loss at 60 W (typical parameters)", fontsize=9)
ax.set_ylim(0, bottom.max() * 1.18)
ax.legend(fontsize=6.5, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.savefig("figures/loss_breakdown.png"); plt.close(fig)

# 7.4 Bode plots
f = np.logspace(1.5, np.log10(fs / 2), 600)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
for name, cp, col in (("A", compA, C1), ("B", compB, C2)):
    for Vin, ls in ((20, "-"), (48, ":")):
        Tm = loop(f, Vin, cp)
        ax[0].semilogx(f, 20 * np.log10(np.abs(Tm)), color=col, ls=ls, label=f"Comp. {name}, {Vin} V")
        ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(Tm))), color=col, ls=ls)
ax[0].axhline(0, color="k", lw=.8); ax[1].axhline(-180, color="k", lw=.8)
ax[0].set_ylabel("|T| (dB)"); ax[1].set_ylabel("∠T (deg)"); ax[1].set_xlabel("Frequency (Hz)")
ax[0].legend(fontsize=7, ncol=2); ax[0].set_ylim(-40, 60)
fig.savefig("figures/bode.png"); plt.close(fig)

# 7.5 load-step and line-step responses (signals averaged over one switching period for readability)
N_AVG = 50   # samples per switching period in the transient records (steps=200, rec_every=4)


def pavg(x):
    return np.convolve(x, np.ones(N_AVG) / N_AVG, mode="same")


fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
for j, Vin in enumerate((24.0, 48.0)):
    for name, col in (("A", C1), ("B", C2)):
        r = tr[(Vin, name)]
        ax[0, j].plot(r[:, 0] * 1e3, pavg(r[:, 3]), color=col, lw=1, label=f"Comp. {name}")
        ax[1, j].plot(r[:, 0] * 1e3, pavg(r[:, 1] + r[:, 2]), color=col, lw=1)
    ax[1, j].plot(r[:, 0] * 1e3, r[:, 8], color="k", lw=1, label="Load current")
    ax[0, j].set_title(f"Load step 5 A → 7.5 A → 5 A, V$_{{in}}$ = {int(Vin)} V", fontsize=8)
    ax[0, j].axhspan(11.76, 12.24, color="#cccccc", alpha=.3)
    ax[1, j].set_xlabel("Time (ms)"); ax[0, j].set_xlim(0.4, 3.9); ax[1, j].set_xlim(0.4, 3.9)
    ax[0, j].set_ylim(11.75, 12.25); ax[1, j].set_ylim(4, 8.5)
ax[0, 0].set_ylabel("V$_{out}$ (V)"); ax[1, 0].set_ylabel("i$_{L1}$+i$_{L2}$ (A)")
ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/load_step.png"); plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True)
for name, col in (("A", C1), ("B", C2)):
    r = tr[("line", name)]
    ax[0].plot(r[:, 0] * 1e3, pavg(r[:, 3]), color=col, lw=1, label=f"Comp. {name}")
ax[1].plot(r[:, 0] * 1e3, r[:, 7], color="k", lw=1)
ax[0].set_ylabel("V$_{out}$ (V)"); ax[1].set_ylabel("V$_{in}$ (V)"); ax[1].set_xlabel("Time (ms)")
ax[0].set_xlim(0.4, 3.9); ax[0].set_ylim(11.98, 12.02)
ax[0].legend(fontsize=7); ax[0].set_title("Line step 24 V → 36 V → 24 V at 5 A (ideal step edges)", fontsize=9)
fig.savefig("figures/line_step.png"); plt.close(fig)

json.dump(res, open("results.json", "w"), indent=1, default=float)
print("eta", res["eta_mean"])
print("comp", res["comp"])
print("settings", res["settings"])
