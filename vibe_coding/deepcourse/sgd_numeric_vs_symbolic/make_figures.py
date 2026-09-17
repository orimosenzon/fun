"""
יוצר את האיורים הסטטיים של השיעור לתוך img/.
הרצה:  python3 make_figures.py
דורש: numpy, matplotlib, python-bidi (לכיתוב עברי בגרפים).
אם CIFAR-10 כבר הורד לתיקייה cifar10_data/, איור הצינור ישתמש בתמונת חתול אמיתית.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

try:
    from bidi.algorithm import get_display as heb
except ImportError:            # נפילה רכה: היפוך פשוט של המחרוזת
    def heb(s):
        return s[::-1]

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, 'img')
os.makedirs(IMG, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# צבעים אחידים לכל האיורים
C_INPUT = '#78909C'
C_PARAM = '#1E88E5'
C_OP    = '#FB8C00'
C_LOSS  = '#E53935'
C_FWD   = '#37474F'
C_BWD   = '#C62828'
C_GOOD  = '#2E7D32'
BG      = '#FAFAFA'


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print('נכתב', path)


def load_sample_cat():
    """תמונת חתול אחת מ-CIFAR-10 אם הנתונים קיימים, אחרת None."""
    path = os.path.join(HERE, 'cifar10_data', 'cifar-10-batches-py', 'data_batch_1')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    labels = np.array(d[b'labels'])
    idx = np.where(labels == 3)[0][20]         # חתול
    return d[b'data'][idx].reshape(3, 32, 32).transpose(1, 2, 0)


def box(ax, xy, w, h, text, fc, fontsize=12, tc='white', lw=1.5):
    r = mpatches.FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=fc, edgecolor='#263238', linewidth=lw, zorder=3)
    ax.add_patch(r)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=tc, zorder=4)


def arrow(ax, p0, p1, color=C_FWD, style='-|>', lw=1.8, ls='-', rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, color=color,
                        linewidth=lw, linestyle=ls, zorder=2,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# איור 1: הצינור המלא, מתמונה להפסד
# ---------------------------------------------------------------------------
def fig_pipeline():
    img = load_sample_cat()
    if img is None:
        rng = np.random.default_rng(0)
        img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    # התמונה
    ax.imshow(img, extent=(0.3, 2.1, 1.5, 3.3), zorder=3, interpolation='nearest')
    ax.add_patch(mpatches.Rectangle((0.3, 1.5), 1.8, 1.8, fill=False, ec='#263238', lw=1.5, zorder=4))
    ax.text(1.2, 1.2, heb('תמונה 32×32, שלושה ערוצי צבע'), ha='center', va='top', fontsize=10)
    ax.text(1.2, 3.55, heb('קלט'), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # הווקטור x, צבוע לפי ערכי הפיקסלים האמיתיים
    flat = img.reshape(-1, 3) / 255.0                      # 1024 שלשות RGB
    strip = flat[::8][:, None, :]                          # דגימה כדי שהפס יהיה קריא
    ax.imshow(strip, extent=(3.35, 3.75, 0.6, 4.2), zorder=3, aspect='auto', interpolation='nearest')
    ax.add_patch(mpatches.Rectangle((3.35, 0.6), 0.4, 3.6, fill=False, ec='#263238', lw=1.2, zorder=4))
    ax.text(3.55, 4.3, r'$\mathbf{x}\in\mathbb{R}^{3072}$', ha='center', va='bottom', fontsize=13)
    ax.text(3.55, 0.45, heb('וקטור פיקסלים'), ha='center', va='top', fontsize=11)
    arrow(ax, (2.25, 2.4), (3.25, 2.4))
    ax.text(2.75, 2.6, heb('פריסה'), ha='center', va='bottom', fontsize=10, color=C_FWD)

    # הנוירון
    box(ax, (4.6, 1.75), 2.6, 1.3, r'$z=\mathbf{w}^{\top}\mathbf{x}+b$', C_PARAM, fontsize=14)
    ax.text(5.9, 3.3, heb('נוירון אחד: משקולות w והטיה b'), ha='center', va='bottom', fontsize=11)
    ax.text(5.9, 1.45, heb('3073 פרמטרים נלמדים'), ha='center', va='top', fontsize=10, color=C_PARAM)
    arrow(ax, (3.85, 2.4), (4.5, 2.4))

    # סיגמואיד
    box(ax, (7.9, 1.75), 2.0, 1.3, r'$\hat{y}=\sigma(z)$', C_OP, fontsize=14)
    ax.text(8.9, 3.3, heb('סיגמואיד: הסתברות'), ha='center', va='bottom', fontsize=11)
    ax.text(8.9, 1.45, r'$\hat{y}\in(0,1)$', ha='center', va='top', fontsize=11, color=C_OP)
    arrow(ax, (7.3, 2.4), (7.8, 2.4))

    # הפסד
    box(ax, (10.6, 1.75), 2.9, 1.3, r'$\ell(\hat{y},y)$', C_LOSS, fontsize=14)
    ax.text(12.05, 3.3, heb('הפסד: כמה טעינו?'), ha='center', va='bottom', fontsize=11)
    ax.text(12.05, 1.45, heb('התווית y: חתול=0, כלב=1'), ha='center', va='top', fontsize=10, color=C_LOSS)
    arrow(ax, (10.0, 2.4), (10.5, 2.4))

    save(fig, 'pipeline.png')


# ---------------------------------------------------------------------------
# איור 2: משיק מול מיתר, הפרש קדמי מול הפרש מרכזי
# ---------------------------------------------------------------------------
def fig_secant_tangent():
    f = lambda w: 0.3 * (w - 1.0) ** 2 + 0.08 * (w - 1.0) ** 3 + 1.0
    df = lambda w: 0.6 * (w - 1.0) + 0.24 * (w - 1.0) ** 2
    w0, eps = 1.4, 0.9
    ws = np.linspace(-0.6, 3.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, mode in zip(axes, ['forward', 'central']):
        ax.set_facecolor(BG)
        ax.plot(ws, f(ws), color=C_FWD, lw=2.2, label='$J(w)$  ' + heb('הפונקציה'))
        # המשיק (הנגזרת האמיתית)
        slope = df(w0)
        ax.plot(ws, f(w0) + slope * (ws - w0), '--', color=C_GOOD, lw=2,
                label=f'{slope:.3f}  ' + heb('משיק, הנגזרת האמיתית:'))
        if mode == 'forward':
            a, b = w0, w0 + eps
            sec = (f(b) - f(a)) / eps
            title = heb('הפרש קדמי: שגיאה מסדר ε')
            lab = f'{sec:.3f}  ' + heb('מיתר, הפרש קדמי:')
            ax.set_title(title, fontsize=14)
            ax.text(0.97, 0.62, r'$\frac{J(w+\varepsilon)-J(w)}{\varepsilon}$', transform=ax.transAxes,
                    ha='right', fontsize=18)
        else:
            a, b = w0 - eps, w0 + eps
            sec = (f(b) - f(a)) / (2 * eps)
            title = heb('הפרש מרכזי: שגיאה מסדר ε²')
            lab = f'{sec:.3f}  ' + heb('מיתר, הפרש מרכזי:')
            ax.set_title(title, fontsize=14)
            ax.text(0.97, 0.62, r'$\frac{J(w+\varepsilon)-J(w-\varepsilon)}{2\varepsilon}$', transform=ax.transAxes,
                    ha='right', fontsize=18)
        ax.plot(ws, f(a) + sec * (ws - a), color=C_LOSS, lw=2, label=lab)
        ax.plot([a, b], [f(a), f(b)], 'o', color=C_LOSS, ms=8, zorder=5)
        ax.plot([w0], [f(w0)], 'o', color=C_GOOD, ms=9, zorder=6, mec='white')
        ax.vlines([a, b], 0.3, [f(a), f(b)], colors='#9E9E9E', linestyles=':', lw=1.2)
        ax.annotate('', xy=(b, 0.42), xytext=(a, 0.42), arrowprops=dict(arrowstyle='<->', color='#616161'))
        ax.text((a + b) / 2, 0.47, ('ε' if mode == 'forward' else '2ε'), ha='center', fontsize=13, color='#616161')
        ax.set_xlabel('w', fontsize=13)
        ax.set_ylim(0.3, 3.0)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('J(w)', fontsize=13)
    save(fig, 'secant_tangent.png')


# ---------------------------------------------------------------------------
# איור 3: תזוזה של קואורדינטה אחת בכל פעם
# ---------------------------------------------------------------------------
def fig_nudge():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    labels = [r'$w_1$', r'$w_2$', r'$w_3$', r'$\cdots$', r'$w_j$', r'$\cdots$', r'$w_{n}$', r'$b$']
    x0, bw, y0 = 0.6, 0.85, 2.0
    for i, lab in enumerate(labels):
        x = x0 + i * (bw + 0.12)
        hot = (lab == r'$w_j$')
        fc = C_LOSS if hot else '#CFD8DC'
        tc = 'white' if hot else '#263238'
        box(ax, (x, y0), bw, 0.8, lab, fc, fontsize=13, tc=tc, lw=2.2 if hot else 1.0)
        if hot:
            xj = x + bw / 2
            arrow(ax, (xj, y0 + 0.85), (xj, y0 + 1.55), color=C_GOOD)
            ax.text(xj, y0 + 1.65, r'$+\varepsilon$', ha='center', va='bottom', fontsize=15, color=C_GOOD)
            ax.text(xj, y0 + 2.2, r'$J_{+}=J(\boldsymbol{\theta}+\varepsilon\,\mathbf{e}_j)$', ha='center',
                    va='bottom', fontsize=13)
            arrow(ax, (xj, y0 - 0.05), (xj, y0 - 0.75), color=C_BWD)
            ax.text(xj, y0 - 0.85, r'$-\varepsilon$', ha='center', va='top', fontsize=15, color=C_BWD)
            ax.text(xj, y0 - 1.4, r'$J_{-}=J(\boldsymbol{\theta}-\varepsilon\,\mathbf{e}_j)$', ha='center',
                    va='top', fontsize=13)
    ax.text(x0 + 1.5 * (bw + 0.12), y0 - 0.55, heb('כל שאר הפרמטרים נשארים קפואים'), ha='center', va='top',
            fontsize=11, color='#546E7A')

    # הנוסחה מימין
    ax.text(11.3, 3.0, r'$\dfrac{\partial J}{\partial \theta_j}\;\approx\;\dfrac{J_{+}-J_{-}}{2\varepsilon}$',
            ha='center', va='center', fontsize=20)
    ax.text(11.3, 1.55, heb('שני חישובים מלאים של J'), ha='center', fontsize=12)
    ax.text(11.3, 1.15, heb('עבור מספר אחד בגרדיאנט'), ha='center', fontsize=12)
    ax.text(11.3, 0.45, heb('חישובי J לגרדיאנט שלם') + '  $2(n+1)$  ' + heb('ובסך הכל:'), ha='center',
            fontsize=12, fontweight='bold', color=C_LOSS)
    save(fig, 'nudge.png')


# ---------------------------------------------------------------------------
# איור 4: גרף החישוב וכלל השרשרת
# ---------------------------------------------------------------------------
def fig_chain_graph():
    fig, ax = plt.subplots(figsize=(15, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    def circ(x, y, text, fc):
        c = mpatches.Circle((x, y), 0.42, facecolor=fc, edgecolor='#263238', lw=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, text, ha='center', va='center', fontsize=14, color='white', zorder=4)

    yc = 3.7
    circ(1.0, yc + 1.2, r'$\mathbf{x}$', C_INPUT)
    circ(1.0, yc, r'$\mathbf{w}$', C_PARAM)
    circ(1.0, yc - 1.2, r'$b$', C_PARAM)
    box(ax, (2.6, yc - 0.55), 2.6, 1.1, r'$z=\mathbf{w}^{\top}\mathbf{x}+b$', C_OP, fontsize=14)
    box(ax, (6.5, yc - 0.55), 2.0, 1.1, r'$\hat{y}=\sigma(z)$', C_OP, fontsize=14)
    box(ax, (9.9, yc - 0.55), 3.6, 1.1,
        r'$\ell=-[\,y\ln\hat{y}+(1-y)\ln(1-\hat{y})\,]$', C_LOSS, fontsize=12)
    circ(12.9, yc + 1.2, r'$y$', C_INPUT)

    # חצים קדימה
    for yy in (yc + 1.2, yc, yc - 1.2):
        arrow(ax, (1.45, yy), (2.55, yc + (yy - yc) * 0.25))
    arrow(ax, (5.25, yc), (6.45, yc))
    arrow(ax, (8.55, yc), (9.85, yc))
    arrow(ax, (12.9, yc + 0.75), (12.9, yc + 0.6))
    ax.text(7.5, yc + 0.75, heb('קדימה: מחשבים ערכים, משמאל לימין'), ha='center', fontsize=12,
            color=C_FWD, fontweight='bold')

    # נגזרות מקומיות מתחת לכל קופסה
    yd = yc - 1.15
    ax.text(3.9, yd, r'$\frac{\partial z}{\partial \mathbf{w}}=\mathbf{x}\;,\quad\frac{\partial z}{\partial b}=1$',
            ha='center', va='top', fontsize=14, color=C_BWD)
    ax.text(7.5, yd, r'$\frac{\partial \hat{y}}{\partial z}=\hat{y}(1-\hat{y})$',
            ha='center', va='top', fontsize=14, color=C_BWD)
    ax.text(11.7, yd, r'$\frac{\partial \ell}{\partial \hat{y}}=-\frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}}$',
            ha='center', va='top', fontsize=14, color=C_BWD)

    # חץ אחורה ארוך, מההפסד עד הפרמטרים
    yb = 1.45
    arrow(ax, (13.4, yb), (1.6, yb), color=C_BWD, ls='--', lw=2.2)
    ax.text(7.5, yb - 0.12, heb('אחורה: מכפילים את הנגזרות המקומיות זו בזו, מימין לשמאל (כלל השרשרת)'),
            ha='center', va='top', fontsize=12, color=C_BWD, fontweight='bold')

    # התוצאה המצטברת
    ax.text(7.5, 0.1,
            r'$\frac{\partial \ell}{\partial z}=\hat{y}-y \qquad\Longrightarrow\qquad '
            r'\frac{\partial \ell}{\partial \mathbf{w}}=(\hat{y}-y)\,\mathbf{x},\quad '
            r'\frac{\partial \ell}{\partial b}=\hat{y}-y$',
            ha='center', va='bottom', fontsize=15, color=C_GOOD,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_GOOD, lw=1.5))
    save(fig, 'chain_graph.png')


# ---------------------------------------------------------------------------
# איור 5: ירידת גרדיאנט מלאה מול סטוכסטית, על בעיית צעצוע דו-ממדית
# ---------------------------------------------------------------------------
def fig_gd_vs_sgd():
    rng = np.random.default_rng(3)
    N = 200
    D = rng.standard_normal((N, 2)) * np.array([1.6, 0.9]) + np.array([2.0, 1.0])
    A = np.array([[1.0, 0.0], [0.0, 3.0]])              # "עקמומיות" שונה בשני הצירים
    mu = D.mean(axis=0)

    def J(th):                                            # ההפסד המלא
        d = th - D
        return 0.5 * np.mean(np.einsum('ij,jk,ik->i', d, A, d))

    def grad(th, idx):                                    # גרדיאנט על תת-קבוצה idx
        d = th[None, :] - D[idx]
        return (d @ A).mean(axis=0)

    th0 = np.array([-2.5, 3.6])
    lr, steps = 0.12, 40
    paths = {}
    for name, m in [('GD', N), ('mini-batch (m=16)', 16), ('SGD (m=1)', 1)]:
        th = th0.copy()
        path = [th.copy()]
        for _ in range(steps):
            idx = np.arange(N) if m == N else rng.choice(N, m, replace=False)
            th = th - lr * grad(th, idx)
            path.append(th.copy())
        paths[name] = np.array(path)

    g1, g2 = np.meshgrid(np.linspace(-3.2, 5.2, 200), np.linspace(-1.5, 4.2, 200))
    Z = np.zeros_like(g1)
    for i in range(g1.shape[0]):
        for j in range(g1.shape[1]):
            Z[i, j] = J(np.array([g1[i, j], g2[i, j]]))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.contour(g1, g2, Z, levels=18, cmap='Greys', linewidths=0.9, alpha=0.8)
    colors = {'GD': C_GOOD, 'mini-batch (m=16)': C_PARAM, 'SGD (m=1)': C_LOSS}
    desc = {'GD': heb('כל 200 הדוגמאות בכל צעד'),
            'mini-batch (m=16)': heb('16 דוגמאות אקראיות בכל צעד'),
            'SGD (m=1)': heb('דוגמה אקראית אחת בכל צעד')}
    for name, p in paths.items():
        ax.plot(p[:, 0], p[:, 1], '-o', color=colors[name], ms=3.5, lw=1.6, alpha=0.9,
                label=f'{name}:  ' + desc[name])
    ax.plot(*th0, 's', color='black', ms=9, zorder=5)
    ax.text(th0[0] + 0.15, th0[1] + 0.15, heb('התחלה'), fontsize=11)
    ax.plot(*mu, '*', color='gold', mec='black', ms=18, zorder=5)
    ax.text(mu[0] + 0.2, mu[1] - 0.35, heb('המינימום'), fontsize=11)
    ax.set_title(heb('40 צעדים באותו קצב למידה: מלא, מיני-אצווה, ודוגמה בודדת'), fontsize=14)
    ax.set_xlabel(r'$\theta_1$', fontsize=13)
    ax.set_ylabel(r'$\theta_2$', fontsize=13)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.set_aspect('equal')
    save(fig, 'gd_vs_sgd.png')


if __name__ == '__main__':
    fig_pipeline()
    fig_secant_tangent()
    fig_nudge()
    fig_chain_graph()
    fig_gd_vs_sgd()
