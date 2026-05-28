"""
RT60 Dashboard — Interactive acoustic measurement visualiser
Requires: matplotlib, pandas, numpy
Run: python rt60_dashboard.py [optional: path/to/cleaned_data.csv]
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import CheckButtons, RadioButtons, Button
from dataclasses import dataclass

# ── Globals ───────────────────────────────────────────────────────────────────
DF                   = None
PLOT_LINES           = []
LEGEND_OBJ           = None
CONDITION_STYLE_MAP  = {}

# ── Dashboard colour tokens ───────────────────────────────────────────────────
BG         = '#0d1117'
PANEL      = '#161b22'
BORDER     = '#30363d'
ACCENT     = '#58a6ff'
TEXT       = '#e6edf3'
DIM        = '#8b949e'
SUCCESS    = '#3fb950'
CTRL_COLOR = '#e2c97e'

PALETTE = [
    '#58a6ff', '#3fb950', '#f78166', '#d2a8ff',
    '#ffa657', '#79c0ff', '#56d364', '#ff7b72',
    '#bc8cff', '#c9a96e', '#39d353', '#ff9800',
]

# Print-safe palette for Light/White export
REPORT_PALETTE = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#555555',
    '#bcbd22', '#17becf', '#393b79', '#637939',
]

LOC_LINE_STYLES = ['-', '--', '-.', ':']
MET_LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
MARKERS         = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8']

FREQ_COLS = ['50','63','80','100','125','160','200','250','315','400',
             '500','630','800','1000','1250','1600','2000','2500',
             '3150','4000','5000','6300','8000','10000']

METRIC_LABELS = ['EDT','T20','T30','Topt','T60M','C50','C80','D50','TS']
METRIC_UNITS  = {
    'EDT':'s', 'T20':'s', 'T30':'s', 'Topt':'s', 'T60M':'s',
    'C50':'dB', 'C80':'dB', 'D50':'%', 'TS':'s',
}
LOC_LABELS    = ['ShMv','SvMv','ShM45','SvM45']
COMPARE_MODES = ['Conditions', 'Locations', 'Metrics']
LEGEND_LOCS   = ['best', 'upper right', 'upper left', 'lower right', 'lower left',
                 'upper center', 'lower center', 'center right', 'center left', 'center']
EXPORT_THEMES = ['White (Report)', 'Light', 'Dark']


# ── Chart customisation state ─────────────────────────────────────────────────
@dataclass
class ChartSettings:
    title:        str   = ''
    x_label:      str   = 'Frequency (Hz)'
    y_label:      str   = ''       # blank = auto
    y_auto:       bool  = True
    y_min:        str   = ''
    y_max:        str   = ''
    legend_pos:   str   = 'best'
    legend_title: str   = ''
    line_width:   float = 2.0
    marker_size:  float = 5.0
    font_size:    int   = 11
    show_markers: bool  = True
    show_grid:    bool  = True
    ctrl_label:   str   = 'Avg Control ±σ'
    export_theme: str   = 'White (Report)'
    export_dpi:   int   = 300


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_matplotlib_theme():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    '#111820',
        'axes.edgecolor':    BORDER,
        'axes.labelcolor':   DIM,
        'axes.titlecolor':   TEXT,
        'xtick.color':       DIM,
        'ytick.color':       DIM,
        'grid.color':        '#1e2630',
        'grid.linewidth':    0.6,
        'text.color':        TEXT,
        'legend.facecolor':  PANEL,
        'legend.edgecolor':  BORDER,
        'legend.labelcolor': TEXT,
        'lines.linewidth':   1.8,
        'font.family':       'sans-serif',
        'font.size':         9,
    })


def build_condition_style_map(conditions):
    global CONDITION_STYLE_MAP
    CONDITION_STYLE_MAP = {}
    for i, cond in enumerate(sorted(conditions)):
        CONDITION_STYLE_MAP[cond] = (
            PALETTE[i % len(PALETTE)],
            MARKERS[i % len(MARKERS)],
        )


def load_csv(path):
    global DF
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    try:
        DF = pd.read_csv(path)
        if 'Repeat' in DF.columns:
            DF['Repeat'] = DF['Repeat'].astype(str)
        build_condition_style_map(DF['Condition'].unique())
        return True, path
    except Exception as e:
        return False, str(e)


def get_series(condition, location, metric):
    if DF is None:
        return None, None
    mask = ((DF['Condition'] == condition) &
            (DF['Location']  == location)  &
            (DF['Metric']    == metric))
    rows = DF[mask]
    if rows.empty:
        return None, None
    cols  = [c for c in FREQ_COLS if c in rows.columns]
    vals  = rows[cols].mean().values
    freqs = [float(c) for c in cols]
    return freqs, vals


def get_control_envelope(location, metric):
    if DF is None:
        return None, None, None, None
    ctrl_conds = [c for c in DF['Condition'].unique() if 'control' in c.lower()]
    if not ctrl_conds:
        return None, None, None, None
    arrays = []
    for cond in ctrl_conds:
        _, vals = get_series(cond, location, metric)
        if vals is not None:
            arrays.append(vals)
    if not arrays:
        return None, None, None, None
    cols  = [c for c in FREQ_COLS if c in DF.columns]
    mat   = np.array(arrays)
    mean  = np.nanmean(mat, axis=0)
    std   = np.nanstd(mat, axis=0)
    freqs = [float(c) for c in cols]
    return freqs, mean, mean - std, mean + std


def available_conditions():
    return sorted(DF['Condition'].unique()) if DF is not None else []

def available_locations():
    return sorted(DF['Location'].unique()) if DF is not None else LOC_LABELS

def available_metrics():
    return sorted(DF['Metric'].unique()) if DF is not None else METRIC_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Settings dialog
# ─────────────────────────────────────────────────────────────────────────────

def open_settings_dialog(settings: ChartSettings) -> ChartSettings:
    """Modal Tkinter dialog for chart customisation. Returns updated settings."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return settings

    root = tk.Tk()
    root.title('Chart Settings')
    root.configure(bg='#1a2030')
    root.resizable(False, False)

    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    DB = '#1a2030'
    style.configure('TFrame',           background=DB)
    style.configure('TLabel',           background=DB, foreground='#e6edf3',
                    font=('Segoe UI', 9))
    style.configure('Dim.TLabel',       background=DB, foreground='#8b949e',
                    font=('Segoe UI', 8, 'italic'))
    style.configure('Section.TLabel',   background=DB, foreground='#58a6ff',
                    font=('Segoe UI', 9, 'bold'))
    style.configure('TEntry',           fieldbackground='#0d1117',
                    foreground='#e6edf3', insertcolor='#e6edf3')
    style.configure('TCombobox',        fieldbackground='#0d1117',
                    foreground='#e6edf3', selectbackground='#30363d',
                    selectforeground='#e6edf3')
    style.configure('TCheckbutton',     background=DB, foreground='#e6edf3',
                    font=('Segoe UI', 9))
    style.map('TCheckbutton',           background=[('active', DB)])

    result = [None]
    f = ttk.Frame(root, padding='18 14 18 14')
    f.pack(fill='both', expand=True)
    f.columnconfigure(1, weight=1)

    row = [0]

    def nr():
        v = row[0]; row[0] += 1; return v

    def section(text):
        ttk.Label(f, text=text, style='Section.TLabel').grid(
            row=nr(), column=0, columnspan=2, sticky='w', pady=(12, 2))

    def sep():
        ttk.Separator(f, orient='horizontal').grid(
            row=nr(), column=0, columnspan=2, sticky='ew', pady=(6, 0))

    def row_entry(label, var, width=32, hint=''):
        rr = nr()
        ttk.Label(f, text=label).grid(row=rr, column=0, sticky='w',
                                       padx=(0, 14), pady=2)
        ttk.Entry(f, textvariable=var, width=width).grid(
            row=rr, column=1, sticky='ew', pady=2)
        if hint:
            ttk.Label(f, text=hint, style='Dim.TLabel').grid(
                row=nr(), column=1, sticky='w')

    def row_combo(label, var, values, width=26):
        rr = nr()
        ttk.Label(f, text=label).grid(row=rr, column=0, sticky='w',
                                       padx=(0, 14), pady=2)
        cb = ttk.Combobox(f, textvariable=var, values=values,
                          width=width, state='readonly')
        cb.grid(row=rr, column=1, sticky='w', pady=2)

    def row_spin(label, var, from_, to, inc, width=8):
        rr = nr()
        ttk.Label(f, text=label).grid(row=rr, column=0, sticky='w',
                                       padx=(0, 14), pady=2)
        tk.Spinbox(f, textvariable=var, from_=from_, to=to,
                   increment=inc, width=width,
                   bg='#0d1117', fg='#e6edf3', relief='flat',
                   buttonbackground='#30363d',
                   insertbackground='#e6edf3').grid(
            row=rr, column=1, sticky='w', pady=2)

    def row_check(label, var):
        ttk.Checkbutton(f, text=label, variable=var).grid(
            row=nr(), column=0, columnspan=2, sticky='w', pady=2)

    # ── Variables ─────────────────────────────────────────────────────────────
    v_title    = tk.StringVar(value=settings.title)
    v_xlabel   = tk.StringVar(value=settings.x_label)
    v_ylabel   = tk.StringVar(value=settings.y_label)
    v_yauto    = tk.BooleanVar(value=settings.y_auto)
    v_ymin     = tk.StringVar(value=settings.y_min)
    v_ymax     = tk.StringVar(value=settings.y_max)
    v_legpos   = tk.StringVar(value=settings.legend_pos)
    v_legtitle = tk.StringVar(value=settings.legend_title)
    v_lw       = tk.DoubleVar(value=settings.line_width)
    v_ms       = tk.DoubleVar(value=settings.marker_size)
    v_fs       = tk.IntVar(value=settings.font_size)
    v_markers  = tk.BooleanVar(value=settings.show_markers)
    v_grid     = tk.BooleanVar(value=settings.show_grid)
    v_ctrllab  = tk.StringVar(value=settings.ctrl_label)
    v_theme    = tk.StringVar(value=settings.export_theme)
    v_dpi      = tk.IntVar(value=settings.export_dpi)

    # ── Sections ──────────────────────────────────────────────────────────────
    section('LABELS')
    row_entry('Chart title',  v_title)
    row_entry('X-axis label', v_xlabel)
    row_entry('Y-axis label', v_ylabel, hint='leave blank for auto')

    sep()
    section('Y AXIS')
    row_check('Auto range', v_yauto)
    row_entry('Y minimum',  v_ymin, width=12)
    row_entry('Y maximum',  v_ymax, width=12)

    sep()
    section('LEGEND')
    row_combo('Position',     v_legpos, LEGEND_LOCS)
    row_entry('Legend title', v_legtitle)

    sep()
    section('APPEARANCE')
    row_spin('Line width',  v_lw, 0.5, 6.0,  0.25)
    row_spin('Marker size', v_ms, 0.0, 14.0, 0.5)
    row_spin('Font size',   v_fs, 7,   22,   1)
    row_check('Show markers',           v_markers)
    row_check('Show grid',              v_grid)
    row_entry('Control overlay label',  v_ctrllab)

    sep()
    section('EXPORT')
    row_combo('Theme', v_theme, EXPORT_THEMES)
    row_spin('DPI',    v_dpi, 72, 600, 50)

    sep()

    # ── Action buttons ────────────────────────────────────────────────────────
    bf = ttk.Frame(f)
    bf.grid(row=nr(), column=0, columnspan=2, pady=(10, 2))

    def on_apply():
        result[0] = ChartSettings(
            title        = v_title.get(),
            x_label      = v_xlabel.get(),
            y_label      = v_ylabel.get(),
            y_auto       = v_yauto.get(),
            y_min        = v_ymin.get(),
            y_max        = v_ymax.get(),
            legend_pos   = v_legpos.get(),
            legend_title = v_legtitle.get(),
            line_width   = float(v_lw.get()),
            marker_size  = float(v_ms.get()),
            font_size    = int(v_fs.get()),
            show_markers = v_markers.get(),
            show_grid    = v_grid.get(),
            ctrl_label   = v_ctrllab.get(),
            export_theme = v_theme.get(),
            export_dpi   = int(v_dpi.get()),
        )
        root.destroy()

    tk.Button(bf, text='  Apply  ', command=on_apply,
              bg=ACCENT, fg='#0d1117', font=('Segoe UI', 9, 'bold'),
              relief='flat', cursor='hand2', pady=5).pack(side='left', padx=6)
    tk.Button(bf, text=' Cancel ', command=root.destroy,
              bg='#30363d', fg=TEXT, font=('Segoe UI', 9),
              relief='flat', cursor='hand2', pady=5).pack(side='left', padx=6)

    root.update_idletasks()
    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f'{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}')
    root.grab_set()
    root.wait_window()

    return result[0] if result[0] is not None else settings


# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality export
# ─────────────────────────────────────────────────────────────────────────────

def export_report_figure(series_list, ctrl_data, settings, auto_ylabel, path, fmt):
    """
    Render a standalone publication-quality figure and save it.
    series_list : [(label, orig_color, linestyle, marker, xs, ys), ...]
    ctrl_data   : (freqs, mean, lo, hi)  — any element may be None
    """
    theme = settings.export_theme
    if theme in ('White (Report)', 'Light'):
        bg_col    = 'white' if theme == 'White (Report)' else '#f5f5f5'
        ax_col    = 'white'
        txt_col   = 'black'
        grid_col  = '#cccccc'
        spine_col = '#888888'
        leg_fc    = 'white'
        leg_ec    = '#aaaaaa'
        use_pal   = REPORT_PALETTE
        ctrl_c    = '#b8860b'
    else:
        bg_col    = BG
        ax_col    = '#111820'
        txt_col   = TEXT
        grid_col  = '#1e2630'
        spine_col = BORDER
        leg_fc    = PANEL
        leg_ec    = BORDER
        use_pal   = PALETTE
        ctrl_c    = CTRL_COLOR

    fs = settings.font_size
    lw = settings.line_width
    ms = settings.marker_size if settings.show_markers else 0

    with matplotlib.rc_context({
        'font.family':       'sans-serif',
        'font.size':         fs,
        'axes.labelsize':    fs,
        'xtick.labelsize':   fs - 1,
        'ytick.labelsize':   fs - 1,
        'legend.fontsize':   fs - 1,
        'figure.facecolor':  bg_col,
        'axes.facecolor':    ax_col,
        'text.color':        txt_col,
        'axes.labelcolor':   txt_col,
        'xtick.color':       txt_col,
        'ytick.color':       txt_col,
        'axes.edgecolor':    spine_col,
        'grid.color':        grid_col,
    }):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(bg_col)
        ax.set_facecolor(ax_col)

        for sp in ax.spines.values():
            sp.set_edgecolor(spine_col)
            sp.set_linewidth(0.8)

        # Data series
        for i, (label, _orig_col, ls, mk, xs, ys) in enumerate(series_list):
            color  = use_pal[i % len(use_pal)]
            marker = mk if settings.show_markers else ''
            ax.plot(xs, ys,
                    color=color, linestyle=ls, marker=marker,
                    markersize=ms, linewidth=lw, label=label,
                    markerfacecolor=color,
                    markeredgecolor=bg_col, markeredgewidth=0.5)

        # Control overlay
        ctrl_freqs, ctrl_mean, ctrl_lo, ctrl_hi = ctrl_data
        if ctrl_freqs is not None:
            ax.plot(ctrl_freqs, ctrl_mean,
                    color=ctrl_c, linestyle='--', linewidth=lw * 0.85,
                    marker='o' if settings.show_markers else '',
                    markersize=ms * 0.85,
                    markerfacecolor=ctrl_c, markeredgecolor=bg_col,
                    markeredgewidth=0.5, label=settings.ctrl_label)
            ax.fill_between(ctrl_freqs, ctrl_lo, ctrl_hi,
                            color=ctrl_c, alpha=0.15, linewidth=0)
            for edge_y in (ctrl_lo, ctrl_hi):
                ax.plot(ctrl_freqs, edge_y,
                        color=ctrl_c, linewidth=0.6, linestyle=':', alpha=0.5)

        # Axes formatting
        ax.set_xscale('log')
        ax.set_xlim(45, 12000)
        ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())

        if settings.show_grid:
            ax.grid(True, which='major', color=grid_col, linewidth=0.7, alpha=0.8)

        xlabel = settings.x_label or 'Frequency (Hz)'
        ylabel = settings.y_label if settings.y_label else auto_ylabel
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        if settings.title:
            ax.set_title(settings.title, fontsize=fs + 1, pad=10)

        # Y limits
        all_ys = [ys for *_, ys in series_list if ys is not None]
        if ctrl_mean is not None:
            all_ys += [ctrl_mean, ctrl_lo, ctrl_hi]
        if all_ys and settings.y_auto:
            combined = np.concatenate(all_ys)
            valid    = combined[np.isfinite(combined)]
            if len(valid):
                lo_v, hi_v = valid.min(), valid.max()
                pad = (hi_v - lo_v) * 0.12 if hi_v != lo_v else 0.5
                ax.set_ylim(lo_v - pad, hi_v + pad)
        elif not settings.y_auto:
            try:
                lo_cur, hi_cur = ax.get_ylim()
                ymin = float(settings.y_min) if settings.y_min.strip() else lo_cur
                ymax = float(settings.y_max) if settings.y_max.strip() else hi_cur
                ax.set_ylim(ymin, ymax)
            except (ValueError, AttributeError):
                pass

        # Legend
        if series_list or ctrl_freqs is not None:
            leg = ax.legend(
                loc=settings.legend_pos,
                title=settings.legend_title or None,
                title_fontsize=fs - 1,
                framealpha=0.9,
                edgecolor=leg_ec,
                facecolor=leg_fc,
                fancybox=False,
                handlelength=2.2,
                borderpad=0.8,
            )
            for t in leg.get_texts():
                t.set_color(txt_col)
            if leg.get_title():
                leg.get_title().set_color(txt_col)

        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=settings.export_dpi,
                    bbox_inches='tight', facecolor=bg_col, edgecolor='none')
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class RT60Dashboard:

    def __init__(self, initial_path=None):
        apply_matplotlib_theme()

        self.current_metric    = 'EDT'
        self.current_location  = 'ShMv'
        self.compare_mode      = 'Conditions'
        self.active_conditions = set()
        self.show_full         = False
        self.show_avg_control  = False
        self.csv_path          = ''
        self._condition_labels = []
        self.settings          = ChartSettings()
        self._last_series      = []
        self._last_ctrl        = (None, None, None, None)
        self._last_auto_ylabel = ''

        self.fig = plt.figure(figsize=(18, 10.5), dpi=96)
        self.fig.patch.set_facecolor(BG)
        try:
            self.fig.canvas.manager.set_window_title('RT60 Dashboard')
        except Exception:
            pass

        self._build_layout()
        self._draw_header()
        self._build_controls()
        self._setup_plot_area()

        if initial_path:
            self._do_load(initial_path)

        plt.show()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        LX, LW = 0.012, 0.218

        self.ax_header   = self.fig.add_axes([LX,           0.930, LW,       0.055])
        self.ax_load_btn = self.fig.add_axes([LX,           0.877, LW * 0.57, 0.040])
        self.ax_path_box = self.fig.add_axes([LX + LW*0.59, 0.879, LW * 0.41, 0.036])

        self.ax_cmp_lbl  = self.fig.add_axes([LX, 0.840, LW, 0.028])
        self.ax_cmp      = self.fig.add_axes([LX, 0.762, LW, 0.075])

        self.ax_met_lbl  = self.fig.add_axes([LX, 0.724, LW, 0.028])
        self.ax_met      = self.fig.add_axes([LX, 0.580, LW, 0.140])

        self.ax_loc_lbl  = self.fig.add_axes([LX, 0.542, LW, 0.028])
        self.ax_loc      = self.fig.add_axes([LX, 0.438, LW, 0.100])

        self.ax_cond_lbl = self.fig.add_axes([LX, 0.400, LW, 0.028])
        self.ax_cond     = self.fig.add_axes([LX, 0.120, LW, 0.278])

        # Bottom button rows (sidebar)
        BH = 0.034
        self.ax_selall_btn = self.fig.add_axes([LX,              0.082, LW*0.48, BH])
        self.ax_clr_btn    = self.fig.add_axes([LX + LW*0.52,    0.082, LW*0.48, BH])

        self.ax_ctrl_btn   = self.fig.add_axes([LX,              0.044, LW,      BH])

        self.ax_full_btn   = self.fig.add_axes([LX,              0.006, LW*0.48, BH])
        self.ax_style_btn  = self.fig.add_axes([LX + LW*0.52,    0.006, LW*0.48, BH])

        # Save / export bar (right of sidebar, bottom)
        self.ax_save_png = self.fig.add_axes([0.268, 0.006, 0.100, BH])
        self.ax_save_svg = self.fig.add_axes([0.374, 0.006, 0.100, BH])
        self.ax_save_csv = self.fig.add_axes([0.480, 0.006, 0.100, BH])
        self.ax_status   = self.fig.add_axes([0.586, 0.006, 0.400, BH])

        # Main plot
        self.ax_plot = self.fig.add_axes([0.258, 0.062, 0.730, 0.920])

    # ── Header ────────────────────────────────────────────────────────────────

    def _draw_header(self):
        ax = self.ax_header
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, 0.85, 'RT60', transform=ax.transAxes,
                fontsize=20, fontweight='bold', color=ACCENT, va='top')
        ax.text(0.0, 0.28, 'A C O U S T I C   D A S H B O A R D', transform=ax.transAxes,
                fontsize=6.5, color=DIM, va='top')

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_controls(self):
        self._build_load_controls()
        self._build_compare_radio()
        self._build_metric_radio()
        self._build_location_radio()
        self._build_condition_checks()
        self._build_utility_buttons()
        self._build_save_buttons()
        self._build_status_bar()

    def _section_label(self, ax, text):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, 0.5, f'▸  {text}', transform=ax.transAxes,
                fontsize=7.5, color=DIM, fontweight='bold', va='center')

    def _style_btn(self, btn, color=ACCENT, hover='#1f2937'):
        btn.ax.set_facecolor(PANEL)
        try:
            btn.color      = PANEL
            btn.hovercolor = hover
        except Exception:
            pass
        btn.label.set_color(color)
        btn.label.set_fontsize(8)
        for sp in btn.ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(0.9)

    def _build_load_controls(self):
        self.btn_load = Button(self.ax_load_btn, '⊕  LOAD CSV')
        self._style_btn(self.btn_load, ACCENT)
        self.btn_load.on_clicked(self._on_load_click)

        self.ax_path_box.set_facecolor(PANEL)
        for sp in self.ax_path_box.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        self.ax_path_box.set_xticks([]); self.ax_path_box.set_yticks([])
        self.path_text = self.ax_path_box.text(
            0.05, 0.5, 'no file loaded',
            transform=self.ax_path_box.transAxes,
            fontsize=6.5, color=DIM, va='center', clip_on=True)

    def _build_compare_radio(self):
        self._section_label(self.ax_cmp_lbl, 'COMPARE BY')
        self.ax_cmp.set_facecolor(PANEL)
        for sp in self.ax_cmp.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        self.radio_cmp = RadioButtons(self.ax_cmp, COMPARE_MODES, activecolor=ACCENT)
        self._style_radio(self.radio_cmp)
        self.radio_cmp.on_clicked(self._on_compare_mode)

    def _build_metric_radio(self):
        self._section_label(self.ax_met_lbl, 'METRIC')
        self.ax_met.set_facecolor(PANEL)
        for sp in self.ax_met.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        self.radio_met = RadioButtons(self.ax_met, METRIC_LABELS, activecolor=ACCENT)
        self._style_radio(self.radio_met)
        self.radio_met.on_clicked(self._on_metric_change)

    def _build_location_radio(self):
        self._section_label(self.ax_loc_lbl, 'LOCATION')
        self.ax_loc.set_facecolor(PANEL)
        for sp in self.ax_loc.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        locs = available_locations() or LOC_LABELS
        self.radio_loc = RadioButtons(self.ax_loc, locs, activecolor=ACCENT)
        self._style_radio(self.radio_loc)
        self.radio_loc.on_clicked(self._on_location_change)
        self.current_location = locs[0]

    def _build_condition_checks(self):
        self._section_label(self.ax_cond_lbl, 'CONDITIONS')
        self.ax_cond.set_facecolor(PANEL)
        for sp in self.ax_cond.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        conds = available_conditions()
        if not conds:
            conds   = ['(load a CSV first)']
            actives = [False]
        else:
            actives = ([True]  * min(3, len(conds)) +
                       [False] * max(0, len(conds) - 3))
            self.active_conditions = {c for c, a in zip(conds, actives) if a}
        self._condition_labels = conds
        self.check_cond = CheckButtons(self.ax_cond, conds, actives)
        self._style_checks(self.check_cond)
        self.check_cond.on_clicked(self._on_condition_toggle)

    def _build_utility_buttons(self):
        self.btn_selall = Button(self.ax_selall_btn, '✓ SELECT ALL')
        self._style_btn(self.btn_selall, DIM)
        self.btn_selall.on_clicked(self._on_select_all)

        self.btn_clr = Button(self.ax_clr_btn, '✕ CLEAR ALL')
        self._style_btn(self.btn_clr, '#f78166')
        self.btn_clr.on_clicked(self._on_clear_all)

        self.btn_ctrl = Button(self.ax_ctrl_btn, '⊗  AVG CONTROL  ±σ')
        self._style_btn(self.btn_ctrl, CTRL_COLOR)
        self.btn_ctrl.on_clicked(self._on_toggle_avg_control)

        self.btn_full = Button(self.ax_full_btn, '◈ FULL BAND')
        self._style_btn(self.btn_full, DIM)
        self.btn_full.on_clicked(self._on_toggle_full)

        self.btn_style = Button(self.ax_style_btn, '⚙ CHART STYLE')
        self._style_btn(self.btn_style, '#d2a8ff')
        self.btn_style.on_clicked(self._on_chart_style)

    def _build_save_buttons(self):
        self.btn_png = Button(self.ax_save_png, '↓ EXPORT PNG')
        self._style_btn(self.btn_png, SUCCESS)
        self.btn_png.on_clicked(lambda e: self._on_save('png'))

        self.btn_svg = Button(self.ax_save_svg, '↓ EXPORT SVG')
        self._style_btn(self.btn_svg, SUCCESS)
        self.btn_svg.on_clicked(lambda e: self._on_save('svg'))

        self.btn_csv = Button(self.ax_save_csv, '↓ SAVE CSV')
        self._style_btn(self.btn_csv, SUCCESS)
        self.btn_csv.on_clicked(lambda e: self._on_save('csv'))

    def _build_status_bar(self):
        self.ax_status.set_facecolor(PANEL)
        for sp in self.ax_status.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.6)
        self.ax_status.set_xticks([]); self.ax_status.set_yticks([])
        self.status_text = self.ax_status.text(
            0.02, 0.5, 'Ready — load a CSV to begin.',
            transform=self.ax_status.transAxes,
            fontsize=7, color=DIM, va='center')

    # ── Widget styling ────────────────────────────────────────────────────────

    def _style_radio(self, radio):
        radio.ax.set_facecolor(PANEL)
        for lbl in radio.labels:
            lbl.set_color(TEXT)
            lbl.set_fontsize(8)
        for attr in ('circles', 'button_patch', 'buttons_patch'):
            patches = getattr(radio, attr, None)
            if patches is not None:
                items = patches if hasattr(patches, '__iter__') else [patches]
                for p in items:
                    try: p.set_edgecolor(BORDER)
                    except Exception: pass
                break

    def _style_checks(self, chk):
        chk.ax.set_facecolor(PANEL)
        conds = self._condition_labels
        for i, lbl in enumerate(chk.labels):
            cond  = conds[i] if i < len(conds) else None
            color = (CONDITION_STYLE_MAP[cond][0]
                     if cond and cond in CONDITION_STYLE_MAP
                     else PALETTE[i % len(PALETTE)])
            lbl.set_color(color)
            lbl.set_fontsize(7.8)
        patches = (getattr(chk, 'rectangles', None) or
                   getattr(chk, 'patches',    None) or [])
        for i, rect in enumerate(patches):
            cond  = conds[i] if i < len(conds) else None
            color = (CONDITION_STYLE_MAP[cond][0]
                     if cond and cond in CONDITION_STYLE_MAP
                     else PALETTE[i % len(PALETTE)])
            try:
                rect.set_edgecolor(color)
                rect.set_facecolor('none')
                rect.set_linewidth(1.2)
            except Exception:
                pass
        for item in getattr(chk, 'lines', []):
            targets = (item
                       if (hasattr(item, '__iter__') and
                           not hasattr(item, 'set_color'))
                       else [item])
            for ln in targets:
                try: ln.set_color(ACCENT)
                except Exception: pass

    # ── Plot area ─────────────────────────────────────────────────────────────

    def _setup_plot_area(self):
        ax = self.ax_plot
        ax.set_facecolor('#0a0f14')
        ax.set_xscale('log')
        ax.set_xlim(45, 12000)
        ax.set_xlabel('Frequency (Hz)', fontsize=9, color=DIM, labelpad=8)
        ax.set_ylabel('', fontsize=9, color=DIM, labelpad=8)
        ax.set_title('', fontsize=11, color=TEXT, pad=10)
        ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='both', which='both', labelsize=7.5,
                       colors=DIM, length=3)
        ax.grid(True, which='major', color='#1a2030', linewidth=0.7, alpha=0.9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
        self.plot_placeholder = ax.text(
            0.5, 0.5, 'Load a CSV file and select\nconditions to plot.',
            transform=ax.transAxes, fontsize=14, color=BORDER,
            ha='center', va='center', linespacing=1.8)

    def _set_status(self, msg, color=DIM):
        self.status_text.set_text(msg)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_load_click(self, event):
        path = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title='Select RT60 CSV',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
            root.destroy()
        except Exception:
            print('\nEnter path to CSV:')
            path = input('>>> ').strip().strip('"').strip("'")
        if path:
            self._do_load(path)

    def _do_load(self, path):
        ok, msg = load_csv(path)
        if ok:
            short = os.path.basename(path)
            self.path_text.set_text(short)
            self.csv_path = path
            self._rebuild_condition_checks()
            self._rebuild_location_radio()
            self._set_status(f'Loaded: {short}  ({len(DF)} rows)', SUCCESS)
            self._refresh_plot()
        else:
            self._set_status(f'Error: {msg}', '#f78166')

    def _rebuild_condition_checks(self):
        self.ax_cond.cla()
        self.ax_cond.set_facecolor(PANEL)
        for sp in self.ax_cond.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        conds   = available_conditions()
        actives = ([True]  * min(3, len(conds)) +
                   [False] * max(0, len(conds) - 3))
        self.active_conditions = {c for c, a in zip(conds, actives) if a}
        self._condition_labels = conds
        self.check_cond = CheckButtons(self.ax_cond, conds, actives)
        self._style_checks(self.check_cond)
        self.check_cond.on_clicked(self._on_condition_toggle)
        self.fig.canvas.draw_idle()

    def _rebuild_location_radio(self):
        self.ax_loc.cla()
        self.ax_loc.set_facecolor(PANEL)
        for sp in self.ax_loc.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
        locs = available_locations()
        self.radio_loc = RadioButtons(self.ax_loc, locs, activecolor=ACCENT)
        self._style_radio(self.radio_loc)
        self.radio_loc.on_clicked(self._on_location_change)
        if locs:
            self.current_location = locs[0]

    def _on_compare_mode(self, label):
        self.compare_mode = label
        self._refresh_plot()

    def _on_metric_change(self, label):
        self.current_metric = label
        self._refresh_plot()

    def _on_location_change(self, label):
        self.current_location = label
        self._refresh_plot()

    def _on_condition_toggle(self, label):
        if label in self.active_conditions:
            self.active_conditions.discard(label)
        else:
            self.active_conditions.add(label)
        self._refresh_plot()

    def _on_select_all(self, event):
        conds = available_conditions()
        if not conds:
            return
        self.active_conditions = set(conds)
        try:
            status = self.check_cond.get_status()
        except AttributeError:
            status = self.check_cond.get_checked()
        for i, active in enumerate(status):
            if not active:
                self.check_cond.set_active(i)
        self._refresh_plot()

    def _on_clear_all(self, event):
        self.active_conditions.clear()
        try:
            status = self.check_cond.get_status()
        except AttributeError:
            status = self.check_cond.get_checked()
        for i, active in enumerate(status):
            if active:
                self.check_cond.set_active(i)
        self._refresh_plot()

    def _on_toggle_full(self, event):
        self.show_full = not self.show_full
        self.btn_full.label.set_color(ACCENT if self.show_full else DIM)
        self._refresh_plot()

    def _on_toggle_avg_control(self, event):
        self.show_avg_control = not self.show_avg_control
        col = CTRL_COLOR if self.show_avg_control else DIM
        self.btn_ctrl.label.set_color(col)
        for sp in self.ax_ctrl_btn.spines.values():
            sp.set_edgecolor(col)
        self._refresh_plot()

    def _on_chart_style(self, event):
        new_settings = open_settings_dialog(self.settings)
        if new_settings is not self.settings:
            self.settings = new_settings
            self._refresh_plot()

    # ── Style lookup ──────────────────────────────────────────────────────────

    def _get_style(self, cond, sub_idx=0, mode='Conditions'):
        color, marker = CONDITION_STYLE_MAP.get(cond, (PALETTE[0], MARKERS[0]))
        if mode == 'Locations':
            ls = LOC_LINE_STYLES[sub_idx % len(LOC_LINE_STYLES)]
        elif mode == 'Metrics':
            ls = MET_LINE_STYLES[sub_idx % len(MET_LINE_STYLES)]
        else:
            ls = '-'
        return color, ls, marker

    # ── Core plot update ──────────────────────────────────────────────────────

    def _refresh_plot(self):
        global PLOT_LINES, LEGEND_OBJ
        ax  = self.ax_plot
        s   = self.settings

        for artist in PLOT_LINES:
            try: artist.remove()
            except Exception: pass
        PLOT_LINES = []
        if LEGEND_OBJ:
            try: LEGEND_OBJ.remove()
            except Exception: pass
            LEGEND_OBJ = None

        if DF is None:
            self.plot_placeholder.set_visible(True)
            self.fig.canvas.draw_idle()
            return

        self.plot_placeholder.set_visible(False)
        mode = self.compare_mode

        series_list = []

        if mode == 'Conditions':
            for cond in sorted(self.active_conditions):
                xs, ys = get_series(cond, self.current_location, self.current_metric)
                if xs is not None:
                    c, ls, mk = self._get_style(cond, 0, mode)
                    series_list.append((cond, c, ls, mk, xs, ys))
            subtitle = f'{self.current_metric}  ·  {self.current_location}'

        elif mode == 'Locations':
            locs = available_locations()
            for cond in sorted(self.active_conditions):
                for j, loc in enumerate(locs):
                    xs, ys = get_series(cond, loc, self.current_metric)
                    if xs is not None:
                        c, ls, mk = self._get_style(cond, j, mode)
                        series_list.append((f'{cond} / {loc}', c, ls, mk, xs, ys))
            subtitle = f'{self.current_metric}  ·  all locations'

        else:
            metrics = available_metrics()
            for cond in sorted(self.active_conditions):
                for j, met in enumerate(metrics):
                    xs, ys = get_series(cond, self.current_location, met)
                    if xs is not None:
                        c, ls, mk = self._get_style(cond, j, mode)
                        series_list.append((f'{cond} / {met}', c, ls, mk, xs, ys))
            subtitle = f'all metrics  ·  {self.current_location}'

        lw = s.line_width
        ms = s.marker_size if s.show_markers else 0

        for label, color, ls, mk, xs, ys in series_list:
            marker = mk if s.show_markers else ''
            line, = ax.plot(xs, ys,
                            color=color, linestyle=ls, marker=marker,
                            markersize=ms, markerfacecolor=color,
                            markeredgecolor=BG, markeredgewidth=0.7,
                            linewidth=lw, label=label, alpha=0.92)
            PLOT_LINES.append(line)

            if self.show_full and DF is not None and 'full' in DF.columns:
                cond_key = label.split('/')[0].strip() if '/' in label else label
                mask = ((DF['Condition'] == cond_key) &
                        (DF['Location']  == self.current_location) &
                        (DF['Metric']    == self.current_metric))
                row = DF[mask]
                if not row.empty:
                    hl = ax.axhline(row['full'].mean(), color=color,
                                    linewidth=0.8, linestyle=':', alpha=0.50)
                    PLOT_LINES.append(hl)

        ctrl_freqs = ctrl_mean = ctrl_lo = ctrl_hi = None
        if self.show_avg_control:
            ctrl_freqs, ctrl_mean, ctrl_lo, ctrl_hi = get_control_envelope(
                self.current_location, self.current_metric)
            if ctrl_freqs is not None:
                ctrl_line, = ax.plot(
                    ctrl_freqs, ctrl_mean,
                    color=CTRL_COLOR, linestyle='--', linewidth=lw * 0.9,
                    marker='o' if s.show_markers else '',
                    markersize=ms * 0.85,
                    markerfacecolor=CTRL_COLOR, markeredgecolor=BG,
                    markeredgewidth=0.7,
                    label=s.ctrl_label, zorder=5, alpha=0.95)
                PLOT_LINES.append(ctrl_line)
                fill = ax.fill_between(ctrl_freqs, ctrl_lo, ctrl_hi,
                                       color=CTRL_COLOR, alpha=0.13,
                                       linewidth=0, zorder=4)
                PLOT_LINES.append(fill)
                for edge_y in (ctrl_lo, ctrl_hi):
                    edge, = ax.plot(ctrl_freqs, edge_y,
                                    color=CTRL_COLOR, linewidth=0.7,
                                    linestyle=':', alpha=0.38, zorder=4)
                    PLOT_LINES.append(edge)

        # Grid
        ax.grid(s.show_grid, which='major', color='#1a2030',
                linewidth=0.7, alpha=0.9)

        # Labels
        met    = self.current_metric if mode != 'Metrics' else 'Multiple'
        unit   = METRIC_UNITS.get(met, '')
        auto_y = f'{met} ({unit})' if unit else met
        if mode == 'Metrics':
            auto_y = 'Value'

        ylabel = s.y_label if s.y_label else auto_y
        xlabel = s.x_label or 'Frequency (Hz)'
        title  = s.title if s.title else subtitle

        ax.set_xlabel(xlabel, fontsize=9, color=DIM, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=9, color=DIM, labelpad=8)
        ax.set_title(title,   fontsize=10, color=TEXT, pad=10)

        # Y limits
        all_ys = [ys for *_, ys in series_list if ys is not None]
        if ctrl_mean is not None:
            all_ys += [ctrl_mean, ctrl_lo, ctrl_hi]
        if all_ys and s.y_auto:
            combined = np.concatenate(all_ys)
            valid    = combined[np.isfinite(combined)]
            if len(valid):
                lo_lim, hi_lim = valid.min(), valid.max()
                pad = (hi_lim - lo_lim) * 0.12 if hi_lim != lo_lim else 0.5
                ax.set_ylim(lo_lim - pad, hi_lim + pad)
        elif not s.y_auto:
            try:
                lo_cur, hi_cur = ax.get_ylim()
                ymin = float(s.y_min) if s.y_min.strip() else lo_cur
                ymax = float(s.y_max) if s.y_max.strip() else hi_cur
                ax.set_ylim(ymin, ymax)
            except (ValueError, AttributeError):
                pass

        # Legend
        has_legend = series_list or ctrl_freqs is not None
        if has_legend:
            LEGEND_OBJ = ax.legend(
                fontsize=9.5,
                framealpha=0.92,
                loc=s.legend_pos,
                title=s.legend_title or None,
                fancybox=False,
                edgecolor=BORDER,
                labelspacing=0.6,
                handlelength=2.4,
                handleheight=1.3,
                borderpad=1.0,
                markerscale=1.4,
            )
            LEGEND_OBJ.get_frame().set_facecolor(PANEL)
            for t in LEGEND_OBJ.get_texts():
                t.set_color(TEXT)
            if LEGEND_OBJ.get_title():
                LEGEND_OBJ.get_title().set_color(DIM)

        # Cache for export
        self._last_series      = series_list
        self._last_ctrl        = (ctrl_freqs, ctrl_mean, ctrl_lo, ctrl_hi)
        self._last_auto_ylabel = auto_y

        n = len(series_list)
        ctrl_note = '  +  avg control' if self.show_avg_control else ''
        self._set_status(
            f'{n} series{ctrl_note}  ·  {self.compare_mode}  ·  {self.current_metric}',
            DIM)
        self.fig.canvas.draw_idle()

    # ── Save / export ─────────────────────────────────────────────────────────

    def _on_save(self, fmt):
        if fmt in ('png', 'svg'):
            self._save_figure(fmt)
        else:
            self._save_csv_export()

    def _save_figure(self, fmt):
        met  = self.current_metric
        loc  = self.current_location
        default_name = f'rt60_{met}_{loc}.{fmt}'
        path = self._ask_save_path(default_name, fmt)
        if not path:
            return
        try:
            export_report_figure(
                self._last_series,
                self._last_ctrl,
                self.settings,
                self._last_auto_ylabel,
                path, fmt)
            self._set_status(f'Exported: {os.path.basename(path)}  '
                             f'[{self.settings.export_theme}  '
                             f'{self.settings.export_dpi} dpi]', SUCCESS)
        except Exception as e:
            self._set_status(f'Export error: {e}', '#f78166')

    def _save_csv_export(self):
        if DF is None:
            self._set_status('No data loaded.', '#f78166')
            return
        default_name = f'rt60_export_{self.current_metric}.csv'
        path = self._ask_save_path(default_name, 'csv')
        if not path:
            return
        mode  = self.compare_mode
        conds = sorted(self.active_conditions)
        rows  = []
        for cond in conds:
            mask = DF['Condition'] == cond
            if mode == 'Conditions':
                mask &= (DF['Location'] == self.current_location)
                mask &= (DF['Metric']   == self.current_metric)
            elif mode == 'Locations':
                mask &= (DF['Metric'] == self.current_metric)
            rows.append(DF[mask])
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.to_csv(path, index=False)
            self._set_status(
                f'Exported {len(out)} rows → {os.path.basename(path)}',
                SUCCESS)
        else:
            self._set_status('Nothing selected to export.', '#f78166')

    def _ask_save_path(self, default_name, fmt):
        path = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.asksaveasfilename(
                title=f'Save as {fmt.upper()}',
                defaultextension=f'.{fmt}',
                initialfile=default_name,
                filetypes=[(f'{fmt.upper()} files', f'*.{fmt}'),
                           ('All files', '*.*')])
            root.destroy()
        except Exception:
            print(f'\nEnter save path (default: {default_name}):')
            p = input('>>> ').strip().strip('"').strip("'")
            path = p if p else default_name
        return path or None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    initial = sys.argv[1] if len(sys.argv) > 1 else None

    if initial is None:
        candidates = [
            'complete_cleaned_data.csv',
            'complete_data.csv',
            os.path.join(os.path.dirname(__file__), 'complete_cleaned_data.csv'),
            os.path.join(os.path.dirname(__file__), 'complete_data.csv'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                initial = c
                break

    RT60Dashboard(initial_path=initial)
