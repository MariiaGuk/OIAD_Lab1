import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані (варіант 3)
N0 = 3400          # чисельність соціальної групи, осіб
T_MAX = 100        # час спостереження, год
N_STAR = 0.4 * N0  # поріг біфуркації: 0.4·N0 = 1360 осіб

TAU = 24           # завдання 1: час до перших змін стану, год
K1 = 0.6           # завдання 1: коефіцієнт емоційної складової

NP = 80            # завдання 2: початкова кількість агентів
TAU2 = 36          # завдання 2: час до перших змін стану, год (tau1 = tau2)
K2 = 0.55          # завдання 2: коефіцієнт емоційної складової (k1 = k2)

K3 = -0.3          # завдання 3: коефіцієнт емоційної складової

t = np.linspace(0.5, T_MAX, 600)
T_ROWS = (10, 20, 30, 40, 50, 60, 80, 100)   # рядки таблиць у звіті


# Формули
def n_one(t, k):
    """Один канал, вираз (2): N(t) = e^(-tau/t) · N0 · k."""
    return np.exp(-TAU / t) * N0 * k


def n_two(t):
    """Два канали, вираз (3): N(t) = Nп + N0·(1 - (1 - k·p)^2), p = e^(-tau/t)."""
    p = np.exp(-TAU2 / t)
    return NP + N0 * (1 - (1 - K2 * p) ** 2)


# Межі при t -> ∞ (p = e^(-tau/t) -> 1)
def n_inf_one(k):
    """Один канал: N(∞) = N0·k."""
    return N0 * k


def n_inf_two():
    """Два канали: N(∞) = Nп + N0·(1 - (1 - k)^2)."""
    return NP + N0 * (1 - (1 - K2) ** 2)


def p_exp(t, tau):
    """Імовірність потрапити під вплив: p(t) = e^(-tau/t)."""
    return np.exp(-tau / t)


# Точки біфуркації: N(t*) = 0.4·N0
def t_star_one(k):
    """Формула (4): t* = tau / ln(k / 0.4); існує лише при k > 0.4."""
    return TAU / np.log(k / 0.4) if k > 0.4 else None


def t_star_two():
    """Формули (5), (6): p* = (1 - sqrt(1 - (N* - Nп)/N0)) / k, t* = -tau / ln p*."""
    p_star = (1 - np.sqrt(1 - (N_STAR - NP) / N0)) / K2
    return -TAU2 / np.log(p_star)


def mark_point(ax, ts):
    """Позначає точку біфуркації на графіку."""
    ax.plot([ts], [N_STAR], "ko", zorder=5)
    ax.vlines(ts, 0, N_STAR, color="k", ls=":", lw=1)
    ax.annotate(f"точка біфуркації\nt* ≈ {ts:.1f} год", (ts, N_STAR), xytext=(ts - 2, N_STAR + 170),
                ha="right", fontsize=10, arrowprops=dict(arrowstyle="-", lw=.6))


def finish(fig, ax, name, loc):
    ax.set_xlim(0, T_MAX)
    ax.set_xticks(range(0, T_MAX + 1, 10))
    ax.set_xlabel("t, год")
    ax.set_ylabel("N(t), осіб")
    ax.legend(loc=loc, fontsize=9)
    fig.tight_layout()
    fig.savefig(name, dpi=200)


plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
SIZE = (7.2, 4.4)
print(f"Поріг: N* = 0.4·N0 = {N_STAR:.0f} осіб")

# Завдання 1
ts1 = t_star_one(K1)
inf1 = n_inf_one(K1)
print("\nЗавдання 1 (один канал, k = 0.6)")
for h in T_ROWS:
    print(f"  t = {h:3d}: p = e^(-{TAU}/t) = {p_exp(h, TAU):.4f},  N = {n_one(h, K1):7.0f},  N/N0 = {n_one(h, K1) / N0:.3f}")
print(f"  Межа при t -> ∞: N = {inf1:.0f} осіб,  N/N0 = {inf1 / N0:.2f}")
print(f"  Точка біфуркації: t* = {ts1:.1f} год")

fig, ax = plt.subplots(figsize=SIZE)
ax.plot(t, n_one(t, K1), color="darkred", lw=2, label="N(t), k = 0.6")
ax.axhline(N_STAR, ls="--", color="gray", label=f"поріг 0.4·N₀ = {N_STAR:.0f}")
ax.axhline(inf1, ls="-.", color="lightgray", label=f"межа N₀·k = {inf1:.0f}")
mark_point(ax, ts1)
ax.set_ylim(0, 2200)
finish(fig, ax, "fig1.png", "lower right")

# Завдання 2
ts2 = t_star_two()
inf2 = n_inf_two()  # межа N(t) при t -> ∞ (p -> 1)
print("\nЗавдання 2 (два канали)")
for h in T_ROWS:
    print(f"  t = {h:3d}: p = e^(-{TAU2}/t) = {p_exp(h, TAU2):.4f},  N = {n_two(h):7.0f},  N/N0 = {n_two(h) / N0:.3f}")
print(f"  Межа при t -> ∞: N = {inf2:.0f} осіб,  N/N0 = {inf2 / N0:.2f}")
print(f"  Точка біфуркації: t* = {ts2:.1f} год (у завданні 1 – {ts1:.1f} год)")

fig, ax = plt.subplots(figsize=SIZE)
ax.plot(t, n_one(t, K1), color="gray", ls="--", lw=1.5, label="один канал (завдання 1)")
ax.plot(t, n_two(t), color="darkgreen", lw=2, label="два канали (завдання 2)")
ax.axhline(N_STAR, ls="--", color="gray", lw=1, label=f"поріг 0.4·N₀ = {N_STAR:.0f}")
ax.axhline(inf2, ls="-.", color="lightgray", label=f"межа N(∞) = {inf2:.0f}")
mark_point(ax, ts2)
ax.set_ylim(0, 3000)
finish(fig, ax, "fig2.png", "lower right")

# Завдання 3: k = -0.3
inf3 = n_inf_one(K3)
print("\nЗавдання 3 (k = -0.3)")
for h in T_ROWS:
    print(f"  t = {h:3d}: p = e^(-{TAU}/t) = {p_exp(h, TAU):.4f},  N = {n_one(h, K3):7.0f},  N/N0 = {n_one(h, K3) / N0:.3f}")
print(f"  Межа при t -> ∞: N = {inf3:.0f} осіб,  N/N0 = {inf3 / N0:.2f}")
print("  Точка біфуркації: не досягається (N(t) < 0 при всіх t)")

fig, ax = plt.subplots(figsize=SIZE)
ax.plot(t, n_one(t, K3), color="navy", lw=2, label="N(t), k = −0.3")
ax.axhline(0, color="k", lw=.8)
ax.axhline(inf3, ls="-.", color="lightgray", label=f"межа N₀·k = {inf3:.0f}".replace("-", "−"))
ax.annotate(f"N(100) ≈ {n_one(100, K3):.0f}".replace("-", "−"), (100, n_one(100, K3)),
            xytext=(70, -200), arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.text(5, -950, "точка біфуркації не досягається", fontsize=10)
ax.set_ylim(-1100, 100)
finish(fig, ax, "fig3.png", "upper right")

# Завдання 3 (продовження): поверхня N(t, k), k від -0.2 до 0.3
print("\nN(t, k), осіб")
print(f"  N(t, k) = N0·k·p(t), p(t) = e^(-{TAU}/t)")
print(f"  p(t):  t=20: {p_exp(20, TAU):.4f}   t=50: {p_exp(50, TAU):.4f}   t=100: {p_exp(100, TAU):.4f}   t->∞: 1")
print("      k   t=20  t=50  t=100  t->∞")
for k in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
    print(f"  {k:+.1f}  {n_one(20, k):5.0f} {n_one(50, k):5.0f} {n_one(100, k):6.0f} {n_inf_one(k):6.0f}")
print("  k ≤ 0.3 < 0.4: поріг не досягається ні для одного k з діапазону")

T, K = np.meshgrid(np.linspace(0.5, T_MAX, 120), np.linspace(-0.2, 0.3, 60))
fig = plt.figure(figsize=(7.2, 5.2))
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(T, K, n_one(T, K), cmap="coolwarm", edgecolor="none", alpha=.95)
ax.set_xlabel("t, год")
ax.set_xticks(range(0, T_MAX + 1, 10))
ax.set_ylabel("k")
ax.set_zlabel("N(t, k), осіб", labelpad=8, rotation=90)
ax.zaxis.set_rotate_label(False)
ax.view_init(elev=24, azim=-125)
fig.colorbar(surf, shrink=.6, pad=.1)
fig.tight_layout()
fig.savefig("fig4.png", dpi=200)

plt.show()