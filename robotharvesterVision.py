# ============================================================
#  КОМП'ЮТЕРНИЙ ЗІР ДЛЯ РОБОТА-ЗБИРАЧА
#  Самостійна робота — Системи ШІ
#  Алгоритми: Perceptron, MLP
#  Автор: Мельник М.О.
# ============================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import threading
import time
from datetime import datetime

# ── Стиль matplotlib ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d1117',
    'axes.facecolor':    '#161b22',
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   '#8b949e',
    'axes.titlecolor':   '#e6edf3',
    'xtick.color':       '#8b949e',
    'ytick.color':       '#8b949e',
    'grid.color':        '#21262d',
    'grid.linewidth':    0.8,
    'lines.linewidth':   2.2,
    'legend.facecolor':  '#161b22',
    'legend.edgecolor':  '#30363d',
    'legend.labelcolor': '#e6edf3',
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Палітра кольорів (GitHub Dark inspired) ──────────────────
BG_DEEP   = '#0d1117'
BG_PANEL  = '#161b22'
BG_CARD   = '#1c2128'
BG_INPUT  = '#21262d'
BORDER    = '#30363d'
FG_MAIN   = '#e6edf3'
FG_DIM    = '#8b949e'
FG_HINT   = '#484f58'

CYAN    = '#79c0ff'
GREEN   = '#56d364'
RED     = '#ff7b72'
YELLOW  = '#e3b341'
ORANGE  = '#f0883e'
PURPLE  = '#bc8cff'
PINK    = '#f778ba'
TEAL    = '#39d353'
LIME    = '#7ee787'
WHITE   = '#f0f6fc'

# ── Клас-каталог об'єктів ────────────────────────────────────
OBJ_CLASSES = ['Яблуко', 'Груша', 'Томат', 'Огірок', 'Перець', 'Баклажан',
               'Листя', 'Ґрунт', 'Камінь', 'Сміття']
OBJ_COLORS  = [RED, ORANGE, '#e53e3e', GREEN, LIME,
               PURPLE, TEAL, '#a0522d', FG_DIM, YELLOW]

FEATURE_NAMES = ['Червоний R', 'Зелений G', 'Синій B',
                 'Округлість', 'Розмір', 'Яскравість', 'Текстура', 'Відтінок']

# ── Іконки класів ────────────────────────────────────────────
OBJ_ICONS = {'Яблуко': '🍎', 'Груша': '🍐', 'Томат': '🍅', 'Огірок': '🥒',
             'Перець': '🌶', 'Баклажан': '🍆', 'Листя': '🍃',
             'Ґрунт': '🪨', 'Камінь': '⬛', 'Сміття': '🗑'}


# ============================================================
# 1. ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ
# ============================================================

def generate_harvest_dataset(n_samples=600, n_classes=6, seed=42):
    rng = np.random.default_rng(seed)
    selected = OBJ_CLASSES[:n_classes]
    profiles = {
        'Яблуко':   {'mean': [220,40,40,0.90,0.65,0.80,0.20,0.05], 'std': [20,15,15,0.05,0.10,0.08,0.05,0.02]},
        'Груша':    {'mean': [200,200,40,0.70,0.60,0.82,0.18,0.15], 'std': [15,20,15,0.06,0.08,0.07,0.05,0.03]},
        'Томат':    {'mean': [230,30,20,0.95,0.45,0.75,0.15,0.02],  'std': [15,12,10,0.03,0.07,0.08,0.04,0.02]},
        'Огірок':   {'mean': [60,180,40,0.30,0.75,0.60,0.30,0.25],  'std': [15,20,15,0.05,0.10,0.07,0.06,0.03]},
        'Перець':   {'mean': [200,150,20,0.60,0.55,0.78,0.22,0.10], 'std': [25,25,15,0.07,0.08,0.08,0.05,0.04]},
        'Баклажан': {'mean': [100,20,140,0.65,0.50,0.35,0.25,0.40], 'std': [20,10,25,0.06,0.08,0.07,0.05,0.05]},
        'Листя':    {'mean': [50,150,40,0.40,0.85,0.55,0.45,0.30],  'std': [20,25,15,0.10,0.15,0.10,0.08,0.05]},
        'Ґрунт':    {'mean': [120,90,60,0.20,0.90,0.40,0.60,0.20],  'std': [25,20,15,0.08,0.12,0.10,0.10,0.04]},
        'Камінь':   {'mean': [140,140,140,0.55,0.70,0.55,0.70,0.02],'std': [30,30,30,0.10,0.12,0.10,0.10,0.02]},
        'Сміття':   {'mean': [160,140,80,0.35,0.60,0.60,0.50,0.15], 'std': [35,35,25,0.12,0.15,0.12,0.12,0.05]},
    }
    per_class = n_samples // n_classes
    X_list, y_list = [], []
    for idx, cls in enumerate(selected):
        p = profiles[cls]
        s = rng.normal(p['mean'], p['std'], size=(per_class, 8))
        s[:, :3] = np.clip(s[:, :3] / 255.0, 0, 1)
        s[:, 3:] = np.clip(s[:, 3:], 0, 1)
        X_list.append(s)
        y_list.extend([idx] * per_class)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], selected


# ============================================================
# 2. АЛГОРИТМИ
# ============================================================

def sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(a): return a * (1.0 - a)
def relu(z):     return np.maximum(0, z)
def relu_d(a):   return (a > 0).astype(np.float32)
def tanh_a(z):   return np.tanh(z)
def tanh_d(a):   return 1.0 - a**2
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

ACTS = {'sigmoid': (sigmoid, sigmoid_d), 'relu': (relu, relu_d), 'tanh': (tanh_a, tanh_d)}


class Perceptron:
    def __init__(self, n_features=8, n_classes=6, lr=0.01, max_iter=150, activation='sigmoid'):
        self.n_features, self.n_classes = n_features, n_classes
        self.lr, self.max_iter = lr, max_iter
        self.act, self.act_d = ACTS.get(activation, ACTS['sigmoid'])
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.1, (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.loss_hist = []; self.acc_hist = []; self.n_iter_ = 0

    def _fwd(self, X): return self.act(X @ self.W + self.b)

    def fit(self, X, y, X_val=None, y_val=None, callback=None, stop_flag=None):
        n = len(X)
        Y = np.zeros((n, self.n_classes), dtype=np.float32)
        Y[np.arange(n), y] = 1.0
        for it in range(self.max_iter):
            A = self._fwd(X)
            diff = A - Y
            loss = float(np.mean(diff**2))
            dZ = diff * self.act_d(A)
            self.W -= self.lr * (X.T @ dZ) / n
            self.b -= self.lr * dZ.mean(axis=0)
            acc = float(np.mean(np.argmax(A, 1) == y))
            self.loss_hist.append(loss); self.acc_hist.append(acc)
            self.n_iter_ = it + 1
            val_acc = float(np.mean(self.predict(X_val) == y_val)) if X_val is not None else None
            if callback: callback(it + 1, loss, acc, val_acc)
            if stop_flag and stop_flag[0]: break
        return self

    def predict_proba(self, X): return self._fwd(X)
    def predict(self, X): return np.argmax(self._fwd(X), axis=1)


class MLP:
    def __init__(self, layer_sizes, lr=0.01, max_iter=200, activation='relu', momentum=0.9, l2=1e-4):
        self.layer_sizes = layer_sizes
        self.lr, self.max_iter = lr, max_iter
        self.momentum, self.l2 = momentum, l2
        self.act, self.act_d = ACTS.get(activation, ACTS['relu'])
        rng = np.random.default_rng(42)
        self.weights, self.biases, self.vel_w, self.vel_b = [], [], [], []
        for i in range(len(layer_sizes) - 1):
            W = rng.normal(0, np.sqrt(2.0 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i+1])).astype(np.float32)
            b = np.zeros(layer_sizes[i+1], dtype=np.float32)
            self.weights.append(W); self.biases.append(b)
            self.vel_w.append(np.zeros_like(W)); self.vel_b.append(np.zeros_like(b))
        self.loss_hist = []; self.acc_hist = []; self.val_acc_hist = []; self.n_iter_ = 0

    def _fwd(self, X):
        acts = [X]; a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = softmax(z) if i == len(self.weights) - 1 else self.act(z)
            acts.append(a)
        return acts

    def _bwd(self, acts, Y, n):
        gw = [None] * len(self.weights); gb = [None] * len(self.weights)
        delta = acts[-1] - Y
        for i in range(len(self.weights) - 1, -1, -1):
            gw[i] = (acts[i].T @ delta) / n + self.l2 * self.weights[i]
            gb[i] = delta.mean(axis=0)
            if i > 0: delta = (delta @ self.weights[i].T) * self.act_d(acts[i])
        return gw, gb

    def fit(self, X, y, X_val=None, y_val=None, batch_size=64, callback=None, stop_flag=None):
        n = len(X)
        Y = np.zeros((n, self.layer_sizes[-1]), dtype=np.float32)
        Y[np.arange(n), y] = 1.0
        rng = np.random.default_rng(42)
        for it in range(self.max_iter):
            idx = rng.permutation(n); total_loss = 0.0
            n_b = max(1, n // batch_size)
            for s in range(0, n, batch_size):
                bi = idx[s:s+batch_size]; Xb, Yb = X[bi], Y[bi]; nb = len(Xb)
                acts = self._fwd(Xb)
                total_loss += -float(np.sum(Yb * np.log(acts[-1] + 1e-12))) / nb
                gw, gb = self._bwd(acts, Yb, nb)
                for i in range(len(self.weights)):
                    self.vel_w[i] = self.momentum * self.vel_w[i] + self.lr * gw[i]
                    self.vel_b[i] = self.momentum * self.vel_b[i] + self.lr * gb[i]
                    self.weights[i] -= self.vel_w[i]; self.biases[i] -= self.vel_b[i]
            loss = total_loss / n_b
            acc = float(np.mean(self.predict(X) == y))
            self.loss_hist.append(loss); self.acc_hist.append(acc); self.n_iter_ = it + 1
            val_acc = None
            if X_val is not None:
                val_acc = float(np.mean(self.predict(X_val) == y_val))
                self.val_acc_hist.append(val_acc)
            if callback: callback(it + 1, loss, acc, val_acc)
            if stop_flag and stop_flag[0]: break
        return self

    def predict_proba(self, X): return self._fwd(X)[-1]
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# 3. МЕТРИКИ
# ============================================================

def accuracy(yt, yp): return float(np.mean(yt == yp))

def confusion_matrix(yt, yp, nc):
    cm = np.zeros((nc, nc), dtype=np.int32)
    for t, p in zip(yt, yp): cm[t, p] += 1
    return cm

def prf(cm):
    n = cm.shape[0]; pr, re, f1 = [], [], []
    for i in range(n):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
        pr.append(p); re.append(r); f1.append(2*p*r/(p+r+1e-9))
    return np.array(pr), np.array(re), np.array(f1)


# ============================================================
# 4. ГОЛОВНИЙ GUI
# ============================================================

class RobotVisionApp:
    AUTHOR = "Мельник М.О."

    def __init__(self, root):
        self.root = root
        self.root.title(f"🤖 Комп'ютерний зір для робота-збирача  |  {self.AUTHOR}")
        self.root.geometry("1600x960")
        self.root.configure(bg=BG_DEEP)
        self.root.minsize(1200, 760)

        self._closed = False
        self._after_id = None
        self._lock = threading.Lock()
        self._running = False
        self._stop_flag = [False]

        self._n_cls_var = tk.IntVar(value=6)
        self.X_train = self.y_train = None
        self.X_val   = self.y_val   = None
        self.X_test  = self.y_test  = None
        self.classes = []
        self.model   = None
        self.result  = {}

        self.pd = {'iter': 0, 'max_iter': 100,
                   'loss_hist': [], 'acc_hist': [], 'val_acc_hist': []}

        self._generate_data()
        self._build_ui()
        self._schedule_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────
    def _generate_data(self, seed=42):
        n = max(300, self._n_cls_var.get() * 80)
        X, y, cls = generate_harvest_dataset(n, self._n_cls_var.get(), seed)
        self.classes = cls
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        t1, t2 = int(len(X)*0.70), int(len(X)*0.85)
        self.X_train, self.y_train = X[idx[:t1]],  y[idx[:t1]]
        self.X_val,   self.y_val   = X[idx[t1:t2]], y[idx[t1:t2]]
        self.X_test,  self.y_test  = X[idx[t2:]],   y[idx[t2:]]
        self.model = None; self.result = {}

    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────
        hdr = tk.Frame(self.root, bg='#090d14', height=62)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        # Left branding
        left_hdr = tk.Frame(hdr, bg='#090d14')
        left_hdr.pack(side='left', padx=18, pady=0, fill='y')

        tk.Label(left_hdr, text="🤖  ROBOT HARVESTER VISION",
                 font=('Courier', 16, 'bold'), fg=CYAN, bg='#090d14'
                 ).pack(side='left', pady=18)
        tk.Label(left_hdr, text="  AI · MLP · Perceptron · ComputerVision",
                 font=('Courier', 9), fg=FG_DIM, bg='#090d14'
                 ).pack(side='left', pady=18)

        # Right: status + author
        right_hdr = tk.Frame(hdr, bg='#090d14')
        right_hdr.pack(side='right', padx=18, fill='y')

        self.status_lbl = tk.Label(right_hdr, text="● Готово",
                                   font=('Courier', 10, 'bold'), fg=GREEN, bg='#090d14')
        self.status_lbl.pack(anchor='e', pady=(10, 2))
        tk.Label(right_hdr, text=f"© 2026 {self.AUTHOR}",
                 font=('Courier', 8), fg=FG_HINT, bg='#090d14').pack(anchor='e')

        # Separator line
        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill='x')

        # ── Notebook ────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('App.TNotebook',
                        background=BG_DEEP, borderwidth=0, tabmargins=[0,0,0,0])
        style.configure('App.TNotebook.Tab',
                        background=BG_CARD, foreground=FG_DIM,
                        font=('Courier', 10, 'bold'),
                        padding=[18, 8], borderwidth=0)
        style.map('App.TNotebook.Tab',
                  background=[('selected', BG_PANEL)],
                  foreground=[('selected', CYAN)])

        nb = ttk.Notebook(self.root, style='App.TNotebook')
        nb.pack(fill='both', expand=True, padx=0, pady=0)

        tabs = {}
        for key, label in [('train', '  🧠 Навчання  '),
                            ('cmp',   '  📊 Порівняння  '),
                            ('test',  '  🔍 Тестування  '),
                            ('help',  '  📋 Довідка  ')]:
            f = tk.Frame(nb, bg=BG_DEEP)
            nb.add(f, text=label)
            tabs[key] = f

        self._build_tab_train(tabs['train'])
        self._build_tab_compare(tabs['cmp'])
        self._build_tab_test(tabs['test'])
        self._build_tab_help(tabs['help'])

        # ── Footer ──────────────────────────────────────────
        ft = tk.Frame(self.root, bg='#090d14', height=24)
        ft.pack(fill='x', side='bottom')
        ft.pack_propagate(False)
        tk.Label(ft,
                 text=f"© 2026 {self.AUTHOR}  ·  Комп'ютерний зір для робота-збирача  ·  Perceptron & MLP  ·  Самостійна робота — Системи ШІ",
                 font=('Courier', 7), fg=FG_HINT, bg='#090d14').pack(pady=5)

    # =========================================================
    # ВКЛАДКА 1 — НАВЧАННЯ
    # =========================================================
    def _build_tab_train(self, parent):
        left = tk.Frame(parent, bg=BG_DEEP, width=308)
        left.pack(side='left', fill='y', padx=(8,4), pady=6)
        left.pack_propagate(False)

        right = tk.Frame(parent, bg=BG_DEEP)
        right.pack(side='left', fill='both', expand=True, padx=(0,8), pady=6)

        # ── Algo card ──
        ac = self._card(left, "⚡  АЛГОРИТМ")
        ac.pack(fill='x', pady=(0,6))
        self.algo_var = tk.StringVar(value='MLP')
        for a, desc in [('Perceptron','Одношаровий · дельта-правило'),
                        ('MLP',       'Багатошаровий · backpropagation')]:
            rb = tk.Radiobutton(ac, text=f"{a}\n{desc}",
                                variable=self.algo_var, value=a,
                                font=('Courier', 9), fg=CYAN, bg=BG_CARD,
                                selectcolor=BG_INPUT, justify='left',
                                activebackground=BG_CARD, activeforeground=CYAN,
                                anchor='w')
            rb.pack(anchor='w', padx=4, pady=3, fill='x')

        # ── Hyperparams ──
        hp = self._card(left, "⚙  ГІПЕРПАРАМЕТРИ")
        hp.pack(fill='x', pady=(0,6))
        self._params = {}
        defs = [
            ("Прихованi шари MLP",   'hidden',  '64,32'),
            ("Швидкість навч. η",    'lr',      '0.01'),
            ("Максимум епох",        'epochs',  '150'),
            ("Активація",            'act',     'relu'),
            ("Momentum µ (MLP)",     'momentum','0.9'),
            ("L2 λ (MLP)",           'l2',      '0.0001'),
            ("Розмір батча (MLP)",   'batch',   '64'),
        ]
        for label, key, default in defs:
            row = tk.Frame(hp, bg=BG_CARD)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, font=('Courier', 8), fg=FG_DIM,
                     bg=BG_CARD, width=22, anchor='w').pack(side='left', padx=4)
            e = tk.Entry(row, font=('Courier', 9), bg=BG_INPUT, fg=YELLOW,
                         insertbackground=FG_MAIN, relief='flat', width=9,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=CYAN)
            e.insert(0, default)
            e.pack(side='right', padx=4)
            self._params[key] = e
        tk.Label(hp, text="Активації: relu  sigmoid  tanh",
                 font=('Courier', 7), fg=FG_HINT, bg=BG_CARD).pack(anchor='w', padx=4, pady=(2,0))

        # ── Dataset card ──
        dc = self._card(left, "🌿  НАБІР ДАНИХ")
        dc.pack(fill='x', pady=(0,6))
        tk.Label(dc, text="Кількість класів об'єктів:",
                 font=('Courier', 8), fg=FG_DIM, bg=BG_CARD).pack(anchor='w')
        cf = tk.Frame(dc, bg=BG_CARD)
        cf.pack(fill='x', pady=2)
        for v in (3, 4, 5, 6, 8, 10):
            tk.Radiobutton(cf, text=str(v), variable=self._n_cls_var, value=v,
                           font=('Courier', 9, 'bold'), fg=FG_DIM, bg=BG_CARD,
                           selectcolor=BG_INPUT, activebackground=BG_CARD
                           ).pack(side='left', padx=2)

        self._btn(dc, "🔄  Новий набір даних", PURPLE,
                  lambda: (self._generate_data(int(time.time()) % 9999),
                           self._draw_dataset())).pack(fill='x', pady=4)
        self.data_lbl = tk.Label(dc, text="", font=('Courier', 8), fg=FG_DIM, bg=BG_CARD)
        self.data_lbl.pack(anchor='w')
        self._refresh_data_lbl()

        # ── Controls ──
        cc = self._card(left, "🎛  КЕРУВАННЯ")
        cc.pack(fill='x', pady=(0,6))
        self.btn_run  = self._btn(cc, "▶  НАВЧАТИ", GREEN,  self._run_training)
        self.btn_run.pack(fill='x', pady=3)
        self.btn_stop = self._btn(cc, "⏹  ЗУПИНИТИ", ORANGE, self._stop_training, state='disabled')
        self.btn_stop.pack(fill='x', pady=3)
        self.btn_reset = self._btn(cc, "↺  СКИНУТИ", PURPLE, self._reset)
        self.btn_reset.pack(fill='x', pady=3)

        # Progress
        self.prog_var = tk.DoubleVar(value=0)
        ps = ttk.Style()
        ps.configure('Prog.Horizontal.TProgressbar',
                     troughcolor=BG_INPUT, background=CYAN, thickness=6)
        ttk.Progressbar(cc, variable=self.prog_var, maximum=100,
                        style='Prog.Horizontal.TProgressbar').pack(fill='x', pady=4)
        self.iter_lbl = tk.Label(cc, text="Епоха: —",
                                 font=('Courier', 8), fg=FG_DIM, bg=BG_CARD)
        self.iter_lbl.pack(anchor='w')

        # ── Results ──
        rc = self._card(left, "📊  РЕЗУЛЬТАТИ")
        rc.pack(fill='both', expand=True)
        self.res_vars = {}
        for label, key, color in [
            ("Алгоритм",   'algo',      CYAN),
            ("Епох",       'niters',    FG_MAIN),
            ("Train Acc",  'train_acc', GREEN),
            ("Val Acc",    'val_acc',   YELLOW),
            ("Test Acc",   'test_acc',  WHITE),
            ("Макро F1",   'f1',        ORANGE),
            ("Час (с)",    'time',      PINK),
        ]:
            row = tk.Frame(rc, bg=BG_CARD)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, font=('Courier', 9), fg=FG_DIM,
                     bg=BG_CARD, width=14, anchor='w').pack(side='left', padx=6)
            v = tk.StringVar(value="—")
            self.res_vars[key] = v
            tk.Label(row, textvariable=v, font=('Courier', 11, 'bold'),
                     fg=color, bg=BG_CARD).pack(side='right', padx=8)

        self._build_train_graphs(right)

    def _build_train_graphs(self, parent):
        self.fig_train = plt.figure(figsize=(12, 8.5), facecolor=BG_DEEP)
        gs = GridSpec(2, 3, figure=self.fig_train,
                      hspace=0.46, wspace=0.34,
                      left=0.06, right=0.97, top=0.95, bottom=0.08)
        self.ax_data    = self.fig_train.add_subplot(gs[0, 0])
        self.ax_loss    = self.fig_train.add_subplot(gs[0, 1])
        self.ax_acc     = self.fig_train.add_subplot(gs[0, 2])
        self.ax_cm      = self.fig_train.add_subplot(gs[1, 0])
        self.ax_scatter = self.fig_train.add_subplot(gs[1, 1])
        self.ax_feat    = self.fig_train.add_subplot(gs[1, 2])

        for ax, t in [(self.ax_data,    'Розподіл даних (R vs G)'),
                      (self.ax_loss,    'Крива втрат'),
                      (self.ax_acc,     'Точність навчання'),
                      (self.ax_cm,      'Матриця похибок'),
                      (self.ax_scatter, 'Scatter: R vs Округлість'),
                      (self.ax_feat,    'Важливість ознак')]:
            ax.set_title(t, fontsize=9, pad=8)
            ax.grid(True, alpha=0.25)

        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=parent)
        self.canvas_train.get_tk_widget().pack(fill='both', expand=True)
        self.canvas_train.draw()
        self._draw_dataset()

    def _draw_dataset(self):
        self._refresh_data_lbl()
        if self.X_train is None: return
        ax = self.ax_data; ax.cla()
        ax.set_title('Розподіл даних (R vs G)', fontsize=9, pad=8)
        ax.grid(True, alpha=0.25)
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.concatenate([self.y_train, self.y_val, self.y_test])
        for ki, cls in enumerate(self.classes):
            mask = y == ki
            ax.scatter(X[mask,0], X[mask,1], s=7, color=OBJ_COLORS[ki%len(OBJ_COLORS)],
                       alpha=0.45, label=f"{OBJ_ICONS.get(cls,'●')} {cls}")
        ax.set_xlabel('R (норм.)', fontsize=7)
        ax.set_ylabel('G (норм.)', fontsize=7)
        ax.legend(fontsize=6, markerscale=3, loc='upper right',
                  framealpha=0.5, ncol=1)
        ax.tick_params(labelsize=7)
        self.canvas_train.draw_idle()

    def _refresh_data_lbl(self):
        if self.X_train is not None:
            self.data_lbl.config(
                text=f"Train / Val / Test : {len(self.y_train)} / {len(self.y_val)} / {len(self.y_test)}")

    def _draw_curves(self):
        pd = self.pd
        self.ax_loss.cla()
        self.ax_loss.set_title('Крива втрат', fontsize=9, pad=8)
        self.ax_loss.grid(True, alpha=0.25)
        if pd['loss_hist']:
            xs = range(1, len(pd['loss_hist'])+1)
            self.ax_loss.plot(xs, pd['loss_hist'], color=RED, lw=2)
            self.ax_loss.fill_between(xs, pd['loss_hist'], alpha=0.12, color=RED)
        self.ax_loss.tick_params(labelsize=7)

        self.ax_acc.cla()
        self.ax_acc.set_title('Точність навчання', fontsize=9, pad=8)
        self.ax_acc.grid(True, alpha=0.25)
        if pd['acc_hist']:
            xs = range(1, len(pd['acc_hist'])+1)
            self.ax_acc.plot(xs, pd['acc_hist'], color=GREEN, lw=2, label='Train')
            if pd['val_acc_hist']:
                xv = range(1, len(pd['val_acc_hist'])+1)
                self.ax_acc.plot(xv, pd['val_acc_hist'], color=YELLOW, lw=2,
                                 linestyle='--', label='Val')
            self.ax_acc.set_ylim(0, 1.05)
            self.ax_acc.legend(fontsize=7)
        self.ax_acc.tick_params(labelsize=7)
        self.canvas_train.draw_idle()

    def _draw_final(self):
        if self.model is None or self.X_test is None: return
        preds  = self.model.predict(self.X_test)
        n_cls  = len(self.classes)
        cm     = confusion_matrix(self.y_test, preds, n_cls)
        short  = [c[:6] for c in self.classes]

        # Confusion matrix
        self.ax_cm.cla()
        self.ax_cm.set_title('Матриця похибок (Test)', fontsize=9, pad=8)
        self.ax_cm.imshow(cm, cmap='viridis', aspect='auto', interpolation='nearest')
        self.ax_cm.set_xticks(range(n_cls)); self.ax_cm.set_yticks(range(n_cls))
        self.ax_cm.set_xticklabels(short, rotation=35, ha='right', fontsize=6)
        self.ax_cm.set_yticklabels(short, fontsize=6)
        for i in range(n_cls):
            for j in range(n_cls):
                self.ax_cm.text(j, i, str(cm[i,j]), ha='center', va='center',
                                fontsize=6, color='white' if cm[i,j] > cm.max()*0.4 else FG_DIM)
        self.ax_cm.tick_params(labelsize=6)

        # Scatter
        self.ax_scatter.cla()
        self.ax_scatter.set_title('R vs Округлість (✓=правильно ×=помилка)', fontsize=8, pad=8)
        self.ax_scatter.grid(True, alpha=0.25)
        for ki, cls in enumerate(self.classes):
            clr = OBJ_COLORS[ki % len(OBJ_COLORS)]
            ok  = (self.y_test==ki) & (preds==ki)
            bad = (self.y_test==ki) & (preds!=ki)
            if ok.any():  self.ax_scatter.scatter(self.X_test[ok,0],  self.X_test[ok,3],  s=10, color=clr, alpha=0.55, marker='o')
            if bad.any(): self.ax_scatter.scatter(self.X_test[bad,0], self.X_test[bad,3], s=35, color=clr, alpha=0.9,  marker='x', linewidths=1.5)
        self.ax_scatter.set_xlabel('R', fontsize=7)
        self.ax_scatter.set_ylabel('Округлість', fontsize=7)
        self.ax_scatter.tick_params(labelsize=7)

        # Feature importance
        self.ax_feat.cla()
        self.ax_feat.set_title('Важливість ознак (|ваги W₁|)', fontsize=9, pad=8)
        self.ax_feat.grid(True, alpha=0.25)
        if hasattr(self.model, 'weights') and self.model.weights:
            imp = np.abs(self.model.weights[0]).mean(axis=1)
        elif hasattr(self.model, 'W'):
            imp = np.abs(self.model.W).mean(axis=1)
        else:
            imp = np.ones(8)
        imp = imp[:8] / (imp[:8].max() + 1e-9)
        cols = [CYAN, GREEN, PURPLE, YELLOW, ORANGE, RED, PINK, WHITE]
        bars = self.ax_feat.barh(range(len(imp)), imp, color=cols[:len(imp)], alpha=0.85, height=0.65)
        self.ax_feat.set_yticks(range(len(FEATURE_NAMES)))
        self.ax_feat.set_yticklabels(FEATURE_NAMES, fontsize=7)
        for bar, val in zip(bars, imp):
            self.ax_feat.text(min(val+0.02, 0.98), bar.get_y()+bar.get_height()/2,
                              f'{val:.2f}', va='center', fontsize=7, color=FG_DIM)
        self.ax_feat.set_xlim(0, 1.15)
        self.ax_feat.tick_params(labelsize=7)
        self.canvas_train.draw_idle()

    # =========================================================
    # ВКЛАДКА 2 — ПОРІВНЯННЯ
    # =========================================================
    def _build_tab_compare(self, parent):
        top = tk.Frame(parent, bg=BG_PANEL, height=52)
        top.pack(fill='x', padx=8, pady=6)
        top.pack_propagate(False)

        self._btn(top, "⚡  Порівняти Perceptron vs MLP", CYAN,
                  self._run_comparison).pack(side='left', padx=8, pady=8)
        self.cmp_lbl = tk.Label(top, text="Натисніть для порівняльного аналізу обох моделей",
                                font=('Courier', 9), fg=FG_DIM, bg=BG_PANEL)
        self.cmp_lbl.pack(side='left', padx=10)

        self.fig_cmp = plt.figure(figsize=(13.5, 8), facecolor=BG_DEEP)
        gs = GridSpec(2, 3, figure=self.fig_cmp,
                      hspace=0.5, wspace=0.36,
                      left=0.06, right=0.97, top=0.93, bottom=0.10)
        self.ax_cl = self.fig_cmp.add_subplot(gs[0,0])
        self.ax_ca = self.fig_cmp.add_subplot(gs[0,1])
        self.ax_cb = self.fig_cmp.add_subplot(gs[0,2])
        self.ax_cp = self.fig_cmp.add_subplot(gs[1,0])
        self.ax_cm2 = self.fig_cmp.add_subplot(gs[1,1])
        self.ax_cf  = self.fig_cmp.add_subplot(gs[1,2])
        for ax, t in [(self.ax_cl,'Крива втрат'), (self.ax_ca,'Точність Train'),
                      (self.ax_cb,'Підсумок метрик'), (self.ax_cp,'CM: Perceptron'),
                      (self.ax_cm2,'CM: MLP'), (self.ax_cf,'F1 по класах')]:
            ax.set_title(t, fontsize=9, pad=8); ax.grid(True, alpha=0.25)

        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=parent)
        self.canvas_cmp.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=4)
        self.canvas_cmp.draw()

    def _run_comparison(self):
        self.cmp_lbl.config(text="⏳ Порівняння виконується (~15–30 с)...", fg=YELLOW)
        threading.Thread(target=self._cmp_thread, daemon=True).start()

    def _cmp_thread(self):
        n_cls = len(self.classes)
        res = {}
        for name in ('Perceptron', 'MLP'):
            t0 = time.time()
            if name == 'Perceptron':
                m = Perceptron(8, n_cls, lr=0.01, max_iter=150)
                m.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            else:
                m = MLP([8,64,32,n_cls], lr=0.01, max_iter=150, activation='relu', momentum=0.9, l2=1e-4)
                m.fit(self.X_train, self.y_train, self.X_val, self.y_val, batch_size=64)
            t = time.time() - t0
            p = m.predict(self.X_test)
            a = accuracy(self.y_test, p)
            cm = confusion_matrix(self.y_test, p, n_cls)
            _, _, f1 = prf(cm)
            res[name] = {'acc':a,'cm':cm,'f1':f1,'time':t,
                         'loss':m.loss_hist,'train_acc':m.acc_hist}
        self.root.after(0, lambda: self._cmp_plot(res))

    def _cmp_plot(self, res):
        pr, mr = res['Perceptron'], res['MLP']
        n_cls = len(self.classes); short = [c[:5] for c in self.classes]

        self.ax_cl.cla()
        self.ax_cl.set_title('Крива втрат', fontsize=9, pad=8)
        self.ax_cl.plot(pr['loss'], color=RED,  lw=2, label='Perceptron')
        self.ax_cl.plot(mr['loss'], color=CYAN, lw=2, label='MLP')
        self.ax_cl.legend(fontsize=7); self.ax_cl.grid(True, alpha=0.25)

        self.ax_ca.cla()
        self.ax_ca.set_title('Точність Train', fontsize=9, pad=8)
        self.ax_ca.plot(pr['train_acc'], color=RED,  lw=2, label='Perceptron')
        self.ax_ca.plot(mr['train_acc'], color=CYAN, lw=2, label='MLP')
        self.ax_ca.legend(fontsize=7); self.ax_ca.grid(True, alpha=0.25)

        self.ax_cb.cla()
        self.ax_cb.set_title('Test Accuracy', fontsize=9, pad=8)
        accs = [pr['acc'], mr['acc']]
        x = np.arange(2)
        bars = self.ax_cb.bar(x, accs, color=[RED, CYAN], alpha=0.85, width=0.5)
        self.ax_cb.set_xticks(x)
        self.ax_cb.set_xticklabels(['Perceptron', 'MLP'], fontsize=9)
        self.ax_cb.set_ylim(0, 1.1)
        for bar, v in zip(bars, accs):
            self.ax_cb.text(bar.get_x()+bar.get_width()/2, v+0.02,
                            f'{v*100:.1f}%', ha='center', fontsize=10, color=WHITE, fontweight='bold')
        # Time twin axis
        ax2 = self.ax_cb.twinx()
        ax2.bar(x + 0.0, [pr['time'], mr['time']], color=[RED, CYAN], alpha=0.3, width=0.2)
        ax2.set_ylabel('Час (с)', color=FG_DIM, fontsize=7)
        ax2.tick_params(axis='y', labelcolor=FG_DIM, labelsize=7)
        self.ax_cb.grid(True, alpha=0.25)

        for ax, cm_data, cmap, title in [
            (self.ax_cp, pr['cm'], 'Reds',  'CM: Perceptron'),
            (self.ax_cm2, mr['cm'], 'Blues', 'CM: MLP'),
        ]:
            ax.cla(); ax.set_title(title, fontsize=9, pad=8)
            ax.imshow(cm_data, cmap=cmap, aspect='auto', interpolation='nearest')
            ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
            ax.set_xticklabels(short, rotation=30, fontsize=6)
            ax.set_yticklabels(short, fontsize=6)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm_data[i,j]), ha='center', va='center',
                            fontsize=6, color='white' if cm_data[i,j] > cm_data.max()*0.4 else FG_DIM)

        self.ax_cf.cla()
        self.ax_cf.set_title('F1 Score за класами', fontsize=9, pad=8)
        xs = np.arange(n_cls)
        self.ax_cf.bar(xs-0.2, pr['f1'], width=0.35, color=RED,  alpha=0.8, label='Perceptron')
        self.ax_cf.bar(xs+0.2, mr['f1'], width=0.35, color=CYAN, alpha=0.8, label='MLP')
        self.ax_cf.set_xticks(xs); self.ax_cf.set_xticklabels(short, rotation=25, fontsize=6)
        self.ax_cf.set_ylim(0, 1.1); self.ax_cf.legend(fontsize=7)
        self.ax_cf.grid(True, alpha=0.25)

        self.canvas_cmp.draw()
        self.cmp_lbl.config(
            text=f"✅ Завершено:  Perceptron {pr['acc']*100:.1f}%  |  MLP {mr['acc']*100:.1f}%  "
                 f"|  Час: {pr['time']:.1f}с / {mr['time']:.1f}с",
            fg=GREEN)

    # =========================================================
    # ВКЛАДКА 3 — ТЕСТУВАННЯ
    # =========================================================
    def _build_tab_test(self, parent):
        # Header
        hdr_test = tk.Frame(parent, bg=BG_PANEL)
        hdr_test.pack(fill='x', padx=8, pady=6)
        tk.Label(hdr_test, text="🔍  ІНТЕРАКТИВНИЙ КЛАСИФІКАТОР — введіть ознаки об'єкта",
                 font=('Courier', 11, 'bold'), fg=CYAN, bg=BG_PANEL).pack(anchor='w', padx=10, pady=8)

        # Sliders area
        sliders_area = tk.Frame(parent, bg=BG_CARD)
        sliders_area.pack(fill='x', padx=8, pady=(0,6))

        self.test_sliders = {}
        demo = [('R (червоний)',0), ('G (зелений)',1), ('Округлість',3), ('Розмір',4),
                ('Яскравість',5), ('Текстура',6)]

        cols_per_row = 3
        for i, (name, idx) in enumerate(demo):
            row_frame = sliders_area if i % cols_per_row != 0 else None
            if i % cols_per_row == 0:
                row_frame = tk.Frame(sliders_area, bg=BG_CARD)
                row_frame.pack(fill='x', padx=8, pady=2)
            f = tk.Frame(row_frame, bg=BG_CARD)
            f.pack(side='left', expand=True, fill='x', padx=6, pady=4)
            tk.Label(f, text=name, font=('Courier', 8, 'bold'), fg=FG_DIM, bg=BG_CARD).pack(anchor='w')
            s = tk.Scale(f, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                         bg=BG_CARD, fg=CYAN, highlightthickness=0, troughcolor=BG_INPUT,
                         activebackground=CYAN, font=('Courier', 8))
            s.set(0.5)
            s.pack(fill='x')
            s.bind("<ButtonRelease-1>", lambda e: self._predict_sample())
            s.bind("<B1-Motion>",       lambda e: self._predict_sample())
            self.test_sliders[idx] = s

        # Prediction display
        pred_frame = tk.Frame(parent, bg=BG_CARD, bd=0)
        pred_frame.pack(fill='both', expand=True, padx=8, pady=(0,6))

        self.pred_lbl = tk.Label(pred_frame,
                                 text="Спочатку навчіть модель на вкладці «Навчання»",
                                 font=('Courier', 15, 'bold'), fg=ORANGE, bg=BG_CARD)
        self.pred_lbl.pack(pady=20)

        # Confidence bar chart
        self.fig_prob = plt.figure(figsize=(11, 3.5), facecolor=BG_CARD)
        self.ax_prob = self.fig_prob.add_subplot(111)
        self.ax_prob.set_title('Впевненість нейромережі за класами (%)',
                               fontsize=10, pad=8)
        self.ax_prob.grid(True, alpha=0.25)
        self.canvas_prob = FigureCanvasTkAgg(self.fig_prob, master=pred_frame)
        self.canvas_prob.get_tk_widget().pack(fill='both', expand=True, padx=16, pady=6)
        self.canvas_prob.draw()

    def _predict_sample(self):
        if self.model is None: return
        sample = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.6, 0.3, 0.1], dtype=np.float32)
        for idx, s in self.test_sliders.items():
            sample[idx] = s.get()
        proba = self.model.predict_proba(sample.reshape(1, -1))[0]
        ci    = int(np.argmax(proba))
        clr   = OBJ_COLORS[ci % len(OBJ_COLORS)]
        icon  = OBJ_ICONS.get(self.classes[ci], '●')
        conf  = proba[ci] * 100

        self.pred_lbl.config(
            text=f"{icon}  {self.classes[ci]}   —   {conf:.1f}%  впевненості",
            fg=clr)

        self.ax_prob.cla()
        self.ax_prob.set_title('Розподіл ймовірностей класів (%)', fontsize=10, pad=8)
        self.ax_prob.grid(True, alpha=0.25)
        y_pos = np.arange(len(self.classes))
        colors = [OBJ_COLORS[i % len(OBJ_COLORS)] for i in range(len(self.classes))]
        bars   = self.ax_prob.barh(y_pos, proba*100, color=colors, alpha=0.75, height=0.65)
        # Highlight winner
        bars[ci].set_alpha(1.0)
        bars[ci].set_linewidth(2)
        bars[ci].set_edgecolor(WHITE)
        self.ax_prob.set_yticks(y_pos)
        labels = [f"{OBJ_ICONS.get(c,'●')} {c}" for c in self.classes]
        self.ax_prob.set_yticklabels(labels, fontsize=9)
        self.ax_prob.set_xlim(0, 108)
        for bar, v in zip(bars, proba*100):
            if v > 2:
                self.ax_prob.text(v+1, bar.get_y()+bar.get_height()/2,
                                  f'{v:.1f}%', va='center', fontsize=8, color=FG_DIM)
        self.canvas_prob.draw()

    # =========================================================
    # ВКЛАДКА 4 — ДОВІДКА
    # =========================================================
    def _build_tab_help(self, parent):
        nb2 = ttk.Notebook(parent)
        nb2.pack(fill='both', expand=True, padx=8, pady=6)

        for title, content in [
            ("📖 Про програму", self._help_about()),
            ("🧮 Алгоритми",    self._help_algo()),
            ("📋 Інструкція",   self._help_manual()),
            ("📝 Про автора",   self._help_author()),
        ]:
            f = tk.Frame(nb2, bg=BG_CARD)
            nb2.add(f, text=title)
            txt = scrolledtext.ScrolledText(f, bg=BG_CARD, fg=FG_MAIN,
                                            insertbackground=FG_MAIN,
                                            font=('Courier', 10), wrap='word', bd=0,
                                            selectbackground=BORDER)
            txt.pack(fill='both', expand=True, padx=4, pady=4)
            txt.insert('1.0', content)
            txt.configure(state='disabled')

    def _help_about(self):
        return f"""╔══════════════════════════════════════════════════════════════════════╗
║         КОМП'ЮТЕРНИЙ ЗІР ДЛЯ РОБОТА-ЗБИРАЧА                         ║
║         Система класифікації агро-об'єктів засобами ШІ               ║
╚══════════════════════════════════════════════════════════════════════╝

МЕТА:
  Класифікація об'єктів на конвеєрі робота-збирача (плоди, листя,
  сторонні предмети) за 8 візуальними ознаками з використанням
  алгоритмів Perceptron та MLP.

ЗАВДАННЯ СИСТЕМИ:
  • Навчання двох нейромережевих класифікаторів на синтетичних даних.
  • Порівняльний аналіз моделей за Accuracy, F1-score та часом навчання.
  • Інтерактивна класифікація об'єктів за ознаками, заданими вручну.

КЛАСИ ОБ'ЄКТІВ (до 10):
  🍎 Яблуко   🍐 Груша   🍅 Томат    🥒 Огірок
  🌶 Перець   🍆 Баклажан  🍃 Листя  🪨 Ґрунт   ⬛ Камінь   🗑 Сміття

ВХІДНІ ОЗНАКИ (8-вимірний вектор):
  [R, G, B]      — нормалізовані колірні канали (0–1)
  [Округлість]   — відношення площі до периметру² (0–1)
  [Розмір]       — площа об'єкта в пікселях (норм.)
  [Яскравість]   — середня інтенсивність відбиття (0–1)
  [Текстура]     — ступінь шорсткості поверхні (0–1)
  [Відтінок]     — спектральний зсув HSV (0–1)

ТЕХНІЧНИЙ СТЕК:
  Python 3.12 · NumPy · Matplotlib · Tkinter
  Реалізація алгоритмів — чиста NumPy (без sklearn/pytorch)

Автор:  {self.AUTHOR}
Дата:   {datetime.now().strftime('%Y-%m-%d')}
"""

    def _help_algo(self):
        return """══════════════════════════════════════════════════════════════
АЛГОРИТМ 1 — PERCEPTRON (ОДНОШАРОВИЙ ПЕРСЕПТРОН)
══════════════════════════════════════════════════════════════

Математична модель:
  z = X · W + b
  A = f(z)       [функція активації: sigmoid/relu/tanh]
  L = (1/n) · ‖A − Y‖²    [MSE]

Оновлення ваг (дельта-правило):
  δ  = (A − Y) · f'(A)
  dW = Xᵀ · δ / n
  W ← W − η · dW
  b ← b − η · mean(δ)

Особливості:
  • Ініціалізація: W ~ N(0, 0.1), b = 0
  • Мультикласовість: one-hot encoding
  • Обмеження: лінійне розділення класів

══════════════════════════════════════════════════════════════
АЛГОРИТМ 2 — MLP (БАГАТОШАРОВИЙ ПЕРСЕПТРОН)
══════════════════════════════════════════════════════════════

Архітектура за замовчуванням:  [8 → 64 → 32 → K]

Forward pass:
  zᵢ = aᵢ₋₁ · Wᵢ + bᵢ
  aᵢ = f(zᵢ)   [прихований шар]
  aₗ = softmax(zₗ)   [вихідний шар]

Функція втрат (крос-ентропія + L2):
  L = −(1/n)·ΣΣ Yᵢₖ · log(Aᵢₖ + ε)  +  λ·Σ‖W‖²

Backpropagation:
  δₗ = Aₗ − Y          [вихідний шар]
  δᵢ = (δᵢ₊₁·Wᵢ₊₁ᵀ) · f'(aᵢ)   [прихований шар]
  ∇Wᵢ = aᵢ₋₁ᵀ · δᵢ / n  +  λ·Wᵢ

Momentum SGD:
  vᵢ ← μ · vᵢ + η · ∇Wᵢ
  Wᵢ ← Wᵢ − vᵢ

Ініціалізація He:
  W ~ N(0, √(2/fan_in))

══════════════════════════════════════════════════════════════
МЕТРИКИ ЯКОСТІ
══════════════════════════════════════════════════════════════
  Accuracy  = (TP + TN) / N
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1-score  = 2·P·R / (P + R)   [макро-усереднення]
"""

    def _help_manual(self):
        return """══════════════════════════════════════════════════════════════
ІНСТРУКЦІЯ З ВИКОРИСТАННЯ ПРОГРАМИ
══════════════════════════════════════════════════════════════

ВКЛАДКА «НАВЧАННЯ»:
─────────────────────────────────────────────────────────────
1. Оберіть алгоритм: Perceptron або MLP.

2. Налаштуйте гіперпараметри:
   • Прихованi шари MLP  — вкажіть розміри через кому, напр.: 64,32
   • Швидкість навч. η   — рекомендовано: 0.001–0.05
   • Максимум епох       — кількість ітерацій (50–500)
   • Активація           — relu (за замовч.), sigmoid, tanh
   • Momentum µ          — 0.8–0.99 (лише для MLP)
   • L2 λ                — регуляризація (0.0001–0.01)
   • Розмір батча        — 32, 64, 128 (лише для MLP)

3. Оберіть кількість класів (3–10).

4. Натисніть «▶ НАВЧАТИ» — система почне навчання у фоновому
   потоці. Прогрес відображається на графіках у реальному часі.

5. Після завершення у лівій панелі з'являться метрики:
   Train Acc / Val Acc / Test Acc / F1.

6. Для скидання натисніть «↺ СКИНУТИ».

ВКЛАДКА «ПОРІВНЯННЯ»:
─────────────────────────────────────────────────────────────
• Натисніть «⚡ Порівняти Perceptron vs MLP».
• Обидва алгоритми запускаються з однаковими даними.
• Результат: порівняння Loss, Accuracy, Матриць похибок, F1.

ВКЛАДКА «ТЕСТУВАННЯ»:
─────────────────────────────────────────────────────────────
• Потребує навченої моделі (вкладка «Навчання»).
• Рухайте слайдери для встановлення 6 ключових ознак об'єкта.
• Нейромережа класифікує об'єкт у реальному часі.
• Гістограма показує впевненість за кожним класом.

ПОРАДИ:
─────────────────────────────────────────────────────────────
• MLP з relu та 150+ епохами дає найкращі результати.
• Якщо Perceptron не збігається — це очікувано: класи
  нелінійно роздільні → тільки MLP здатний їх розрізнити.
• Для демонстрації на захисті: запустіть «Порівняння» та
  поясніть різницю між матрицями похибок.
"""

    def _help_author(self):
        return f"""╔══════════════════════════════════════════════════════════════════════╗
║                         ПРО АВТОРА                                   ║
╚══════════════════════════════════════════════════════════════════════╝

  Автор:        Мельник М.О.
  Дисципліна:   Системи штучного інтелекту
  Робота:       Самостійна 
                Комп'ютерний зір для робота-збирача
  Рік:          2026

  Кафедра:      САП, Львівська політехніка

══════════════════════════════════════════════════════════════════════
РЕАЛІЗОВАНО:
  ✅ Одношаровий персептрон (Perceptron) з дельта-правилом
  ✅ Багатошаровий персептрон (MLP) з backpropagation
  ✅ Momentum SGD + L2 регуляризація
  ✅ He ініціалізація ваг
  ✅ Крос-ентропійна функція втрат + softmax
  ✅ Синтетичний датасет із 10 класів агро-об'єктів
  ✅ Метрики: Accuracy, F1-score, Confusion Matrix
  ✅ Інтерактивна класифікація за слайдерами
  ✅ Порівняльний аналіз двох моделей
  ✅ Темна тема, адаптивний UI, фоновий потік навчання

══════════════════════════════════════════════════════════════════════
ТЕХНОЛОГІЇ:
  Python 3.12  |  NumPy  |  Matplotlib  |  Tkinter
  Всі алгоритми реалізовані вручну без sklearn/PyTorch.

  © 2026  {self.AUTHOR}
  Дата генерації звіту: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # =========================================================
    # ЛОГІКА НАВЧАННЯ
    # =========================================================
    def _run_training(self):
        if self._running: return
        self._running = True
        self._stop_flag[0] = False
        self.btn_run.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.status_lbl.config(text="⏳ Навчання...", fg=YELLOW)
        self.pd['loss_hist'].clear(); self.pd['acc_hist'].clear(); self.pd['val_acc_hist'].clear()
        algo    = self.algo_var.get()
        lr      = float(self._params['lr'].get())
        epochs  = int(self._params['epochs'].get())
        act     = self._params['act'].get().strip()
        self.pd['max_iter'] = epochs
        threading.Thread(target=self._train_thread, args=(algo, lr, epochs, act), daemon=True).start()

    def _train_thread(self, algo, lr, epochs, act):
        n_cls = len(self.classes)
        t0 = time.time()

        def cb(epoch, loss, acc, val_acc):
            with self._lock:
                self.pd['iter'] = epoch
                self.pd['loss_hist'].append(loss)
                self.pd['acc_hist'].append(acc)
                if val_acc is not None: self.pd['val_acc_hist'].append(val_acc)

        if algo == 'Perceptron':
            self.model = Perceptron(8, n_cls, lr=lr, max_iter=epochs, activation=act)
            self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val,
                           callback=cb, stop_flag=self._stop_flag)
        else:
            try: sizes = [int(x.strip()) for x in self._params['hidden'].get().split(',')]
            except: sizes = [64, 32]
            mom   = float(self._params['momentum'].get())
            l2    = float(self._params['l2'].get())
            batch = int(self._params['batch'].get())
            self.model = MLP([8]+sizes+[n_cls], lr=lr, max_iter=epochs, activation=act,
                              momentum=mom, l2=l2)
            self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val,
                           batch_size=batch, callback=cb, stop_flag=self._stop_flag)

        elapsed = time.time() - t0
        preds   = self.model.predict(self.X_test)
        test_acc = accuracy(self.y_test, preds)
        cm_res  = confusion_matrix(self.y_test, preds, n_cls)
        _, _, f1 = prf(cm_res)
        macro_f1 = float(np.mean(f1))
        self.root.after(0, lambda: self._train_done(algo, test_acc, macro_f1, elapsed))

    def _train_done(self, algo, test_acc, macro_f1, elapsed):
        with self._lock:
            pd = self.pd
            ta = pd['acc_hist'][-1] if pd['acc_hist'] else 0.0
            va = pd['val_acc_hist'][-1] if pd['val_acc_hist'] else 0.0
            self.res_vars['algo'].set(algo)
            self.res_vars['niters'].set(str(len(pd['loss_hist'])))
            self.res_vars['train_acc'].set(f"{ta*100:.2f}%")
            self.res_vars['val_acc'].set(f"{va*100:.2f}%")
            self.res_vars['test_acc'].set(f"{test_acc*100:.2f}%")
            self.res_vars['f1'].set(f"{macro_f1:.4f}")
            self.res_vars['time'].set(f"{elapsed:.2f}")
        self._draw_final()
        self._predict_sample()
        self._running = False
        self.btn_run.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.status_lbl.config(text=f"✅ Готово: {test_acc*100:.1f}% Test Acc", fg=GREEN)

    def _stop_training(self):
        self._stop_flag[0] = True
        self.status_lbl.config(text="⏸ Зупинка...", fg=ORANGE)

    def _reset(self):
        self._generate_data()
        self._draw_dataset()
        for v in self.res_vars.values(): v.set("—")
        self.prog_var.set(0)
        self.iter_lbl.config(text="Епоха: —")
        for ax in [self.ax_loss, self.ax_acc, self.ax_cm, self.ax_scatter, self.ax_feat]:
            ax.cla(); ax.grid(True, alpha=0.25)
        self.canvas_train.draw_idle()
        self.status_lbl.config(text="● Готово", fg=GREEN)

    def _schedule_refresh(self):
        if self._closed: return
        try:
            if self._running:
                with self._lock:
                    pd = self.pd
                    if pd['max_iter'] > 0 and pd['loss_hist']:
                        cur = pd['iter']; total = pd['max_iter']
                        self.prog_var.set(cur / total * 100)
                        self.iter_lbl.config(
                            text=f"Епоха {cur}/{total}   Loss: {pd['loss_hist'][-1]:.5f}")
                self._draw_curves()
            self._after_id = self.root.after(280, self._schedule_refresh)
        except tk.TclError:
            pass

    def _on_close(self):
        self._closed = True; self._running = False
        if self._after_id:
            try: self.root.after_cancel(self._after_id)
            except tk.TclError: pass
        plt.close('all')
        self.root.destroy()

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _card(parent, title):
        return tk.LabelFrame(parent, text=title,
                             font=('Courier', 8, 'bold'),
                             fg=CYAN, bg=BG_CARD, bd=1,
                             relief='flat', labelanchor='nw',
                             padx=6, pady=4,
                             highlightthickness=1,
                             highlightbackground=BORDER)

    @staticmethod
    def _btn(parent, text, color, cmd, state='normal'):
        return tk.Button(parent, text=text, command=cmd,
                         font=('Courier', 9, 'bold'),
                         bg=color, fg=BG_DEEP,
                         activebackground=color, activeforeground=BG_DEEP,
                         relief='flat', bd=0, padx=10, pady=7,
                         cursor='hand2', state=state)


# ============================================================
# ТОЧКА ВХОДУ
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = RobotVisionApp(root)
    root.mainloop()