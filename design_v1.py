"""EE463 pre-report design calculations + power-stage simulation.
Two-phase interleaved synchronous buck with an inversely coupled inductor.
Outputs: figures/*.png, results.json
"""
import json, os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.formatter.useoffset": False})
C_BLUE, C_ORNG, C_GRN, C_GRY = "#2a6fb0", "#d9822b", "#3a9a5b", "#555555"

# ---------------- specification & design choices ----------------
Vo, Io_nom, Io_ovl = 12.0, 5.0, 7.5
VIN = [20, 24, 36, 48]
fs = 250e3; T = 1 / fs                     # per-phase switching frequency
L, alpha = 27e-6, 0.5                      # self inductance, coupling coeff (inverse)
M = alpha * L
Ltr, Ldiff = L - M, L + M                  # common-mode (transient) / differential
# MOSFET: TI CSD19534Q5A (100 V, 12.6 mOhm typ / 15.1 max @10 V, Qg 17 nC, Qgd 3.2 nC,
# Qgs 5.1 nC, Qg(th) 3.3 nC, Qoss 44 nC @50 V, Qrr 134 nC @10 A, Vsd 0.8 V, RthJA 50 C/W)
Rds_hot = 15.1e-3 * 1.35                   # max RDS(on) x hot factor (~100 C)
Qg, Qgd, Qgs, Qgth, Qoss50, Qrr10 = 17e-9, 3.2e-9, 5.1e-9, 3.3e-9, 44e-9, 134e-9
Vdrv, Vpl, Rg_int, Rg_ext, Rdrv = 10.0, 4.5, 1.1, 2.0, 1.5   # assumed driver (pending)
t_dead, Vsd = 30e-9, 0.8
DCR = 10e-3                                 # per winding, hot (from magnetic design)
ESR_out = 2e-3                              # 4 x 22 uF X7R in parallel
C_out_eff = 4 * 11e-6                       # 4 x 22 uF 25 V X7R 1210, ~50 % DC-bias derating @12 V
Rpol = 8e-3                                 # reverse-polarity MOSFET (placeholder)
P_aux = 0.30                                # controller + drivers + aux supply (placeholder, pending)

res = {"design": dict(fs=fs, L=L, alpha=alpha, M=M, Ltr=Ltr, Ldiff=Ldiff,
                      C_out_eff=C_out_eff, DCR=DCR, Rds_hot=Rds_hot)}

# ---------------- coupled-inductor analytic relations ----------------
def Leq_ss(D, L=L, a=alpha):
    """steady-state equivalent inductance seen by one phase (inverse coupling)"""
    Dm = D if D <= 0.5 else 1 - D
    return L * (1 - a**2) / (1 - a * Dm / (1 - Dm))

def ripple_phase(Vin, L=L, a=alpha):
    D = Vo / Vin
    return Vo * (1 - D) * T / Leq_ss(D, L, a)

def ripple_sum(Vin, Ltr_=Ltr):
    D = Vo / Vin
    if D <= 0.5:
        return Vo * (1 - 2 * D) * T / Ltr_
    return Vo * (2 * D - 1) * (1 - D) / D * T / Ltr_   # (Vin-Vo)(2D-1)T / Ltr ... normalised

# ---------------- loss model ----------------
def losses(Vin, Io, ripple_ph, Bpp=0.1):
    Iph = Io / 2
    D = Vo / Vin
    for _ in range(5):  # iterate D with conduction drops
        D = (Vo + Iph * (Rds_hot + DCR)) / (Vin - Io * Rpol / (1))  # simple correction
    Irms_ph = np.sqrt(Iph**2 + ripple_ph**2 / 12)
    P_cond_hs = Irms_ph**2 * D * Rds_hot
    P_cond_ls = Irms_ph**2 * (1 - D) * Rds_hot
    Ig_on = (Vdrv - Vpl) / (Rdrv + Rg_ext + Rg_int)
    Ig_off = Vpl / (Rdrv + Rg_ext + Rg_int)
    Qsw = Qgd + (Qgs - Qgth)
    t_on, t_off = Qsw / Ig_on, Qsw / Ig_off
    I_on, I_off = Iph - ripple_ph / 2, Iph + ripple_ph / 2
    P_sw = 0.5 * Vin * (I_on * t_on + I_off * t_off) * fs
    P_oss = Qoss50 * np.sqrt(Vin / 50) * Vin * 0.5 * fs * 2   # HS+LS output charge (approx.)
    P_rr = Qrr10 * (I_on / 10) * Vin * fs                      # Qrr scaled with current
    P_dead = Vsd * (I_on + I_off) * t_dead * fs
    P_gate = 2 * Qg * Vdrv * fs
    P_cu = 2 * Irms_ph**2 * DCR / 2  # per phase share computed below (one winding)
    per_phase = dict(cond_hs=P_cond_hs, cond_ls=P_cond_ls, sw=P_sw, oss=P_oss, rr=P_rr,
                     dead=P_dead, gate=P_gate, cu=Irms_ph**2 * DCR)
    tot_phase = sum(per_phase.values())
    Iin = Vo * Io / Vin / 0.94
    P_pol = Iin**2 * Rpol
    # N87 Steinmetz estimate anchored to TDK E25/13/7 data (375 kW/m3 @ 200 mT,100 kHz,100 C), a=1.3, b=2.6
    P_core = 375e3 * ((Bpp / 2) / 0.2)**2.6 * (fs / 100e3)**1.3 * 3.02e-6
    P_cap = 0.02
    Ploss = 2 * tot_phase + P_pol + P_core + P_cap + P_aux
    Pout = Vo * Io
    return dict(D=D, Irms_ph=Irms_ph, per_phase=per_phase, phase_total=tot_phase,
                P_pol=P_pol, P_core=P_core, P_cap=P_cap, P_aux=P_aux, Ploss=Ploss,
                eff=Pout / (Pout + Ploss), t_on=t_on, t_off=t_off,
                Tj_hs=25 + 40 + (P_cond_hs + P_sw + P_oss / 2 + P_rr) * 35,
                P_fet_hs=P_cond_hs + P_sw + P_oss / 2 + P_rr,
                P_fet_ls=P_cond_ls + P_oss / 2 + P_dead)

# ---------------- time-domain power-stage simulation ----------------
def simulate(Vin, Io, L_=L, a=alpha, C=C_out_eff, n_per=1400, steps=500, D=None,
             step_load=None):
    M_ = a * L_
    Rl = Vo / Io
    Lm = np.array([[L_, -M_], [-M_, L_]]); Li = np.linalg.inv(Lm)
    R = Rds_hot + DCR
    dt = T / steps
    def mats(Rl_):
        k = 1 / (1 + ESR_out / Rl_)
        # x = [i1, i2, vc];  vo = k (vc + ESR isum)
        A = np.zeros((3, 3)); A[0:2, 0:2] = -Li * R
        vo_i = k * ESR_out; vo_v = k
        for r in range(2):
            A[r, 0] += -(Li[r, 0] + Li[r, 1]) * vo_i
            A[r, 1] += -(Li[r, 0] + Li[r, 1]) * vo_i
            A[r, 2] += -(Li[r, 0] + Li[r, 1]) * vo_v
        A[2, 0] = A[2, 1] = (1 - vo_i / Rl_) / C
        A[2, 2] = -vo_v / Rl_ / C
        Bm = np.zeros((3, 2)); Bm[0:2, :] = Li
        Ad = expm(A * dt)
        Bd = np.linalg.solve(A, (Ad - np.eye(3))) @ Bm
        return Ad, Bd, k
    if D is None:
        D = (Vo + Io / 2 * R) / Vin
    x = np.array([Io / 2, Io / 2, Vo])
    Ad, Bd, k = mats(Rl)
    rec = []
    for p in range(n_per):
        if step_load and p == step_load[0]:
            Rl = Vo / step_load[1]; Ad, Bd, k = mats(Rl)
        for s in range(steps):
            ph = s / steps
            s1 = ph < D; s2 = ((ph + .5) % 1) < D
            u = np.array([s1 * Vin, s2 * Vin])
            if p >= n_per - 4 or step_load:
                vo = k * (x[2] + ESR_out * (x[0] + x[1]))
                rec.append((p * T + s * dt, x[0], x[1], vo, s1 * x[0] + s2 * x[1], s1, s2))
            x = Ad @ x + Bd @ u
    return np.array(rec), D

def steady(Vin, Io, **kw):
    D = None
    for _ in range(3):
        r, D = simulate(Vin, Io, D=D, **kw)
        vavg = r[:, 3].mean()
        D = D * Vo / vavg if D else None
    r, D = simulate(Vin, Io, D=D, **kw)
    return r, D

# ---------------- run ----------------
table = []
sims = {}
for Vin in VIN:
    for Io in (Io_nom, Io_ovl):
        r, D = steady(Vin, Io)
        w = r[-2 * 500:]  # last two periods
        i1, i2, vo, iin = w[:, 1], w[:, 2], w[:, 3], w[:, 4]
        row = dict(Vin=Vin, Io=Io, D_ideal=Vo / Vin, D_sim=D,
                   dI_ph_calc=ripple_phase(Vin), dI_ph_sim=float(np.ptp(i1)),
                   dI_sum_calc=abs(ripple_sum(Vin)), dI_sum_sim=float(np.ptp(i1 + i2)),
                   Vo_avg=float(vo.mean()), Vpp_sim=float(np.ptp(vo)),
                   Vpp_calc=abs(ripple_sum(Vin)) / (8 * 2 * fs * C_out_eff) + ESR_out * abs(ripple_sum(Vin)),
                   Iph_pk=float(max(i1.max(), i2.max())), Iph_rms=float(np.sqrt((i1**2).mean())),
                   Iin_avg=float(iin.mean()), Iin_rms=float(np.sqrt((iin**2).mean())),
                   Icin_rms=float(np.sqrt(((iin - iin.mean())**2).mean())))
        phi1 = (L * w[:, 1] - M * w[:, 2]) / 10
        row["B_pp_outer"] = float(np.ptp(phi1) / 26.25e-6)
        L_ = losses(Vin, Io, row["dI_ph_calc"], row["B_pp_outer"])
        row.update(eff=L_["eff"], Ploss=L_["Ploss"], loss=L_)
        table.append(row)
        sims[(Vin, Io)] = r
        print(f"Vin={Vin} Io={Io}: D={D:.3f} dIph {row['dI_ph_calc']:.2f}/{row['dI_ph_sim']:.2f} "
              f"dIsum {row['dI_sum_calc']:.2f}/{row['dI_sum_sim']:.2f} Vo={row['Vo_avg']:.3f} "
              f"Vpp={row['Vpp_sim']*1e3:.1f}mV (calc {row['Vpp_calc']*1e3:.1f}) Ipk={row['Iph_pk']:.2f} "
              f"Icin={row['Icin_rms']:.2f} eff={L_['eff']*100:.1f}% Pfet_hs={L_['P_fet_hs']:.2f}")
res["table"] = table
eta = [r["eff"] for r in table if r["Io"] == 5]
res["eta_avg"] = float(np.mean(eta)); res["bonus_eff"] = 100 * (np.mean(eta) - 0.8)
print("eta avg", res["eta_avg"], "bonus", res["bonus_eff"])

# ---------------- uncoupled comparison (same transient inductance) ----------------
cmp_rows = []
for Vin in VIN:
    cmp_rows.append(dict(Vin=Vin, coupled=ripple_phase(Vin), uncoupled_sameLtr=ripple_phase(Vin, Ltr, 0),
                         uncoupled_sameL=ripple_phase(Vin, L, 0), sum_coupled=abs(ripple_sum(Vin)),
                         sum_uncoupled_sameL=abs(ripple_sum(Vin, L))))
res["compare"] = cmp_rows

# ---------------- magnetic design (E25/13/7, N87) ----------------
mu0 = 4e-7 * np.pi
Ac, Ao = 52.5e-6, 26.25e-6     # centre leg (Ae) and outer leg (~Ae/2)
N = 10
Ro = N**2 / Ldiff               # outer-leg reluctance
Rc = (N**2 / Ltr - Ro) / 2      # centre-leg reluctance
g_o, g_c = Ro * mu0 * Ao, Rc * mu0 * Ac
# flux from overload simulation at worst ripple (48 V) and at 20 V
Bmax = {}
for Vin in (20, 48):
    r = sims[(Vin, Io_ovl)][-1000:]
    i1, i2 = r[:, 1], r[:, 2]
    phi1 = (L * i1 - M * i2) / N
    phic = (L - M) * (i1 + i2) / N
    Bmax[Vin] = dict(B_outer_pk=float(phi1.max() / Ao), B_outer_pp=float(np.ptp(phi1) / Ao),
                     B_centre_pk=float(phic.max() / Ac))
# winding: 10 turns, 2 x 0.8 mm (AWG20-ish) parallel -> Cu area
Acu = 2 * np.pi * (0.8e-3 / 2)**2
MLT = 38e-3
Rdc20 = 1.72e-8 * N * MLT / Acu
res["magnetic"] = dict(core="TDK/EPCOS E 25/13/7, N87", Ac=Ac, Ao=Ao, N=N, g_outer=g_o, g_centre=g_c,
                       Bmax=Bmax, Acu=Acu, MLT=MLT, Rdc20=Rdc20, Rdc100=Rdc20 * 1.31,
                       J_ovl=float(np.sqrt(3.75**2 + 0.25) / (Acu * 1e6)))
print("magnetic", res["magnetic"])

# ---------------- figures ----------------
def fig_wave(Vin, Io, fname):
    r = sims[(Vin, Io)][-2 * 500:]
    t = (r[:, 0] - r[0, 0]) * 1e6
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    ax[0].plot(t, r[:, 5] * 1 + 1.2, color=C_BLUE, lw=1, label="Phase A gate (HS)")
    ax[0].plot(t, r[:, 6] * 1, color=C_ORNG, lw=1, label="Phase B gate (HS)")
    ax[0].set_yticks([]); ax[0].legend(loc="upper right", fontsize=7, ncol=2); ax[0].set_title(
        f"Vin = {Vin} V, Io = {Io} A (D = {Vo/Vin:.2f} ideal)", fontsize=9)
    ax[1].plot(t, r[:, 1], color=C_BLUE, lw=1.2, label="iL1")
    ax[1].plot(t, r[:, 2], color=C_ORNG, lw=1.2, label="iL2")
    ax[1].plot(t, r[:, 1] + r[:, 2], color=C_GRN, lw=1.2, label="iL1 + iL2")
    ax[1].set_ylabel("Current (A)"); ax[1].legend(loc="upper right", fontsize=7, ncol=3)
    ax[2].plot(t, r[:, 3], color=C_GRY, lw=1.2)
    ax[2].set_ylabel("Vout (V)"); ax[2].set_xlabel("Time (µs)")
    fig.savefig(f"figures/{fname}.png"); plt.close(fig)
for Vin in VIN:
    fig_wave(Vin, Io_nom, f"wave_{Vin}V_5A")
fig_wave(20, Io_ovl, "wave_20V_7A5")

# ripple vs duty plot (analytic)
Dg = np.linspace(0.2, 0.65, 200)
Vg = Vo / Dg
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
ax[0].plot(Dg, [ripple_phase(v) for v in Vg], color=C_BLUE, label=f"Coupled (L={L*1e6:.0f} µH, α={alpha})")
ax[0].plot(Dg, [ripple_phase(v, Ltr, 0) for v in Vg], color=C_ORNG, ls="--", label=f"Uncoupled, L={Ltr*1e6:.1f} µH (same L_tr)")
ax[0].set_xlabel("Duty cycle D"); ax[0].set_ylabel("ΔI per phase (A p-p)"); ax[0].legend(fontsize=7)
ax[1].plot(Dg, [abs(ripple_sum(v)) for v in Vg], color=C_GRN)
ax[1].set_xlabel("Duty cycle D"); ax[1].set_ylabel("Δ(iL1+iL2) (A p-p)")
for a_ in ax:
    a_.axvspan(12 / 48, 12 / 20, color="#cccccc", alpha=.25)
fig.tight_layout(); fig.savefig("figures/ripple_vs_D.png"); plt.close(fig)

# loss breakdown bar
fig, ax = plt.subplots(figsize=(6.4, 2.9))
cats = ["cond_hs", "cond_ls", "sw", "oss", "rr", "dead", "gate", "cu"]
names = ["HS cond.", "LS cond.", "HS switching", "Coss", "Qrr", "Dead-time", "Gate drive", "Winding Cu"]
bottom = np.zeros(4)
pal = ["#2a6fb0", "#7aa6d6", "#d9822b", "#f0b97a", "#b5462f", "#8c6bb1", "#3a9a5b", "#9bcf9f"]
rows5 = [r for r in table if r["Io"] == 5]
for c, n, col in zip(cats, names, pal):
    vals = np.array([2 * r["loss"]["per_phase"][c] for r in rows5])
    ax.bar([str(v) for v in VIN], vals, bottom=bottom, color=col, label=n); bottom += vals
other = np.array([r["loss"]["P_pol"] + r["loss"]["P_core"] + r["loss"]["P_cap"] + r["loss"]["P_aux"] for r in rows5])
ax.bar([str(v) for v in VIN], other, bottom=bottom, color="#bbbbbb", label="Core+caps+rev.pol.+aux")
ax.set_xlabel("Input voltage (V)"); ax.set_ylabel("Loss (W)"); ax.set_title("Estimated loss at 60 W", fontsize=9)
ax.legend(fontsize=6.5, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.savefig("figures/loss_breakdown.png"); plt.close(fig)

# open-loop load step 2.5 A -> 5 A at 24 V (power-stage dynamics only)
r, D = simulate(24, 2.5, D=(Vo + 1.25 * (Rds_hot + DCR)) / 24, n_per=900, steps=100, step_load=(500, 5.0))
fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True)
r = r[r[:, 0] > 380 * T]
t = r[:, 0] * 1e3
ax[0].plot(t, r[:, 1] + r[:, 2], color=C_GRN, lw=.8); ax[0].set_ylabel("iL1+iL2 (A)")
ax[1].plot(t, r[:, 3], color=C_GRY, lw=.8); ax[1].set_ylabel("Vout (V)"); ax[1].set_xlabel("Time (ms)")
ax[0].set_title("Open-loop load step 2.5 A → 5 A at Vin = 24 V (fixed duty)", fontsize=9)
fig.savefig("figures/openloop_step.png"); plt.close(fig)
res["openloop_step"] = dict(Vo_before=float(r[(r[:, 0] > 480 * T) & (r[:, 0] < 500 * T), 3].mean()),
                            Vo_after=float(r[r[:, 0] > 850 * T, 3].mean()),
                            Vmin=float(r[r[:, 0] > 500 * T, 3].min()))
print("step", res["openloop_step"])

json.dump(res, open("results.json", "w"), indent=1, default=float)
