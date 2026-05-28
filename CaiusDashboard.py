"""
Acoustic Measurement Dashboard — Caius Chapel
Requires: matplotlib >= 3.7, pandas, numpy

Key fixes vs previous version
──────────────────────────────
1. OVERLAPPING-AXES BUG: ax_comb and ax_loc were at the same figure position.
   set_visible(False) does NOT prevent ax.contains(event) returning True, so
   BOTH CheckButtons received every click.  Fix: the inactive panel is moved
   off-screen with set_position() so only the active one is in bounds.

2. CALLBACK SYNC: _on_comb / _on_loc / _make_overlay_cb now read the widget's
   get_status() directly instead of toggling the internal set.  This is immune
   to any ordering assumptions about when the widget updates its visual state.

3. OVERLAY ALL/NONE: "All" selects every series in the group (including those
   not shown in the strip); "None" deselects all of them.  The strip only
   displays summaries/averages for readability.

4. SAVE BUTTON: saves the current axes (main plot only) as a timestamped PNG.
"""

import os
import sys
import datetime
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, CheckButtons, RadioButtons

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
ACCENT  = '#58a6ff'
TEXT    = '#e6edf3'
DIM     = '#8b949e'
WARN    = '#ffa657'
SUCCESS = '#3fb950'

PALETTE = [
    '#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657',
    '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#7ee787',
    '#ff8b94', '#e3b341', '#a5d6ff', '#ffb4a2', '#b7f4c0',
    '#f0b4f0', '#90e0ef', '#ffcb77', '#c9ada7', '#ffd6a5',
]

OVERLAY_COLORS = {'AJ': '#79c0ff', 'SL': '#56d364', 'RK': '#ffa657'}

METRIC_UNITS = {
    'C50': 'dB', 'C80': 'dB', 'D50': '%',
    'EDT': 's',  'T20': 's',  'T30': 's',
    'TS':  's',  'Topt': 's',
}
HIDDEN_METRICS = {'T60M'}

FREQ_TICKS       = [50, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 10000]
FREQ_TICK_LABELS = ['50', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '10k']

# Off-screen position used to park a panel so it never receives click events
_OFFSCREEN = [-2.0, 0.5, 0.27, 0.37]


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_files():
    base = os.path.dirname(os.path.abspath(__file__))
    comp_candidates = [
        "Comparison_dataset__cleaned_.csv",
        "Comparison dataset (cleaned).csv",
        "Comparison_dataset_cleaned.csv",
    ]
    rev_candidates = ["CAIUS_existing_Reverb_data_collated.csv"]

    comp = next((os.path.join(base, n) for n in comp_candidates
                 if os.path.exists(os.path.join(base, n))), None)
    rev  = next((os.path.join(base, n) for n in rev_candidates
                 if os.path.exists(os.path.join(base, n))), None)

    if not comp:
        sys.exit(f"ERROR: Cannot find Comparison CSV in {base}")
    if not rev:
        sys.exit(f"ERROR: Cannot find Reverb CSV in {base}")
    return comp, rev


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_comparison(path):
    df = pd.read_csv(path, encoding='latin1')
    df = df[~df['Metric'].isin(HIDDEN_METRICS)].copy()
    freq_cols = [c for c in df.columns if c.lstrip('-').replace('.', '').isdigit()]

    def map_loc(c):
        if not isinstance(c, str): return 'Unknown'
        if c.startswith('A'):      return 'Location A'
        if c.startswith('B'):      return 'Location B'
        if c.startswith('WE'):     return 'West End'
        return c

    df['LocationGroup'] = df['Combination'].apply(map_loc)
    for fc in freq_cols:
        df[fc] = pd.to_numeric(df[fc], errors='coerce')
    return df, freq_cols


def _parse_reverb_block(raw, row_start, row_end, col_end):
    blk = raw.iloc[row_start:row_end, 0:col_end].copy().reset_index(drop=True)
    blk.columns = blk.iloc[0].astype(str).str.strip()
    blk = blk.iloc[1:].reset_index(drop=True)
    blk.rename(columns={blk.columns[0]: 'Freq'}, inplace=True)
    for col in blk.columns:
        blk[col] = pd.to_numeric(blk[col], errors='coerce')
    return blk.dropna(subset=['Freq']).reset_index(drop=True)


def load_reverb(path):
    """
    Returns OrderedDict of groups:
      { 'AJ': {'label', 'color',
               'display_items': OrderedDict{name: (f,v)},   ← shown in strip
               'all_items':     OrderedDict{name: (f,v)}},  ← All button scope
        ... }
    display_items = summaries/averages only (keeps strip compact)
    all_items     = every series (used by All/None buttons)
    """
    raw = pd.read_csv(path, encoding='latin1', header=None)
    groups = OrderedDict()

    # ── Adrian James (AJ): rows 0-8, cols 0-13 ───────────────
    aj = _parse_reverb_block(raw, 0, 9, 14)
    aj_all = OrderedDict()

    for avg_name, cols_ in [
        ('Choir A (Avg)',  ['Choir A1 (s)', 'Choir A2 (s)', 'Choir A3 (s)']),
        ('Choir B (Avg)',  ['Choir B1 (s)', 'Choir B2 (s)', 'Choir B3 (s)']),
        ('Organ (Avg)',    ['Organ 1 (s)',  'Organ 2 (s)',  'Organ 3 (s)']),
        ('West End (Avg)',['West End 1 (s)', 'West End 2 (s)', 'West End 3 (s)']),
    ]:
        present = [c for c in cols_ if c in aj.columns]
        if present:
            aj_all[avg_name] = (aj['Freq'].values,
                                aj[present].mean(axis=1).values)
    if 'Ideal Time (s)' in aj.columns:
        aj_all['Ideal Time'] = (aj['Freq'].values, aj['Ideal Time (s)'].values)
    for c in aj.columns:
        if c not in ('Freq', 'Ideal Time (s)'):
            aj_all[c.replace(' (s)', '')] = (aj['Freq'].values, aj[c].values)

    groups['AJ'] = {
        'label':         'Adrian James (AJ)',
        'color':         OVERLAY_COLORS['AJ'],
        'display_items': OrderedDict(
            (k, v) for k, v in aj_all.items()
            if any(t in k for t in ('Avg', 'Ideal'))
        ),
        'all_items': aj_all,
    }

    # ── Shawn Li (SL): header row 13, data rows 14-19 ────────
    sl = _parse_reverb_block(raw, 13, 20, 13)
    sl_all = OrderedDict()
    avg_cands = [c for c in ['C3', 'C4', 'EXP4 C3', 'EXP1 C3'] if c in sl.columns]
    if avg_cands:
        sl_all['Average (C3/C4/EXP)'] = (sl['Freq'].values,
                                          sl[avg_cands].mean(axis=1).values)
    for c in sl.columns:
        if c != 'Freq' and not sl[c].isna().all():
            sl_all[c] = (sl['Freq'].values, sl[c].values)

    groups['SL'] = {
        'label':         'Shawn Li (SL)',
        'color':         OVERLAY_COLORS['SL'],
        'display_items': OrderedDict(
            (k, v) for k, v in sl_all.items()
            if 'Average' in k or 'Avg' in k
        ),
        'all_items': sl_all,
    }
    # If SL has no summary, show first 3 items
    if not groups['SL']['display_items']:
        groups['SL']['display_items'] = OrderedDict(
            list(sl_all.items())[:3])

    # ── Roger Kelly (RK): header row 27, data rows 28-34 ─────
    rk = _parse_reverb_block(raw, 27, 35, 7)
    rk_all = OrderedDict()
    for c in rk.columns:
        if c != 'Freq' and len(rk[c].dropna()) >= 2:
            rk_all[c] = (rk['Freq'].values, rk[c].values)

    groups['RK'] = {
        'label':         'Roger Kelly (RK)',
        'color':         OVERLAY_COLORS['RK'],
        'display_items': OrderedDict(
            (k, v) for k, v in rk_all.items()
            if 'Calc' in k
        ),
        'all_items': rk_all,
    }
    if not groups['RK']['display_items']:
        groups['RK']['display_items'] = OrderedDict(
            list(rk_all.items())[:4])

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# WIDGET HELPERS  — matplotlib 3.7+ API only
# ─────────────────────────────────────────────────────────────────────────────

def style_checks(cb, check_colors=None, fontsize=8):
    """
    Style CheckButtons with the mpl-3.7 set_*_props() API.
    Labels are set via direct Text iteration (avoids cycler scalar TypeError).
    """
    n    = len(cb.labels)
    cols = (check_colors or [ACCENT] * n)
    cols = (cols * n)[:n]
    for lbl in cb.labels:
        lbl.set_color(TEXT)
        lbl.set_fontsize(fontsize)
    cb.set_frame_props({'edgecolor': cols, 'facecolor': 'none', 'linewidth': 1.2})
    cb.set_check_props({'facecolor': cols, 'edgecolor': cols})


def style_radio(rb, fontsize=8):
    """Style RadioButtons with the mpl-3.7 set_*_props() API."""
    for lbl in rb.labels:
        lbl.set_color(TEXT)
        lbl.set_fontsize(fontsize)
    rb.set_radio_props({'facecolor': ACCENT, 'edgecolor': BORDER})


def _add_header(fig, x, y, w, h, text, color=ACCENT, fontsize=8):
    ax = fig.add_axes([x, y, w, h], facecolor=BG)
    ax.axis('off')
    ax.axhline(0.5, color=BORDER, linewidth=0.8)
    ax.text(0.5, 0.5, f'  {text}  ', transform=ax.transAxes,
            ha='center', va='center', color=color, fontsize=fontsize,
            fontweight='bold', bbox=dict(facecolor=BG, edgecolor='none', pad=2))
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

class AcousticDashboard:

    def __init__(self, comp_df, freq_cols, reverb_groups):
        self.comp_df       = comp_df
        self.freq_cols     = freq_cols
        self.reverb_groups = reverb_groups

        self.metrics        = sorted(comp_df['Metric'].unique())
        self.current_metric = 'T30' if 'T30' in self.metrics else self.metrics[0]
        self.avg_mode       = False

        self.selected_wedges       = {'Yes', 'No'}
        self.selected_combinations = set(comp_df['Combination'].unique())
        self.selected_locations    = set(comp_df['LocationGroup'].unique())
        self.selected_overlays: set = set()

        # Re-entrancy guard for bulk set_active() calls
        self._updating = False

        # Hard references — matplotlib holds only weak refs to Button callbacks
        self._btn_refs: list = []

        all_labels = (sorted(comp_df['Combination'].unique()) +
                      sorted(comp_df['LocationGroup'].unique()))
        self.color_map = {lbl: PALETTE[i % len(PALETTE)]
                          for i, lbl in enumerate(all_labels)}

        self._build_ui()
        self.refresh_plot()

    # ─── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        plt.rcParams.update({
            'text.color':      TEXT,
            'axes.labelcolor': TEXT,
            'xtick.color':     DIM,
            'ytick.color':     DIM,
            'axes.edgecolor':  BORDER,
        })

        self.fig = plt.figure(figsize=(18, 10), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                'Caius Chapel — Acoustic Analysis')
        except AttributeError:
            pass

        # Main plot — 21 % bottom reserved for overlay strip
        self.ax = self.fig.add_axes([0.19, 0.21, 0.51, 0.74], facecolor=PANEL)

        self._build_left_panel()
        self._build_right_panel()
        self._build_overlay_strip()

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        _add_header(self.fig, 0.01, 0.93, 0.16, 0.030, 'SELECT METRIC')
        ax_m = self.fig.add_axes([0.01, 0.63, 0.16, 0.29], facecolor=PANEL)
        self.radio_metric = RadioButtons(
            ax_m, self.metrics,
            active=self.metrics.index(self.current_metric),
            activecolor=ACCENT)
        style_radio(self.radio_metric)
        self.radio_metric.on_clicked(self._on_metric)

        _add_header(self.fig, 0.01, 0.57, 0.16, 0.030, 'WEDGE PANELS')
        ax_w = self.fig.add_axes([0.01, 0.49, 0.16, 0.08], facecolor=PANEL)
        self.check_wedge = CheckButtons(ax_w, ['Yes', 'No'], [True, True])
        style_checks(self.check_wedge, check_colors=[ACCENT, '#f78166'])
        self.check_wedge.on_clicked(self._on_wedge)

        _add_header(self.fig, 0.01, 0.43, 0.16, 0.030, 'QUICK SELECT')
        b1 = self._make_btn([0.01, 0.37, 0.07, 0.05], 'All',
                            self._select_all_filter)
        b2 = self._make_btn([0.09, 0.37, 0.07, 0.05], 'None',
                            self._select_none_filter)
        self._btn_refs += [b1, b2]

        # Save button
        _add_header(self.fig, 0.01, 0.31, 0.16, 0.030, 'EXPORT')
        b_save = self._make_btn([0.01, 0.25, 0.16, 0.05], 'Save Plot  [PNG]',
                                self._save_plot, fg=SUCCESS)
        self._btn_refs.append(b_save)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self):
        _add_header(self.fig, 0.72, 0.93, 0.27, 0.030, 'MEASUREMENT FILTER')

        ax_mode = self.fig.add_axes([0.72, 0.88, 0.27, 0.04], facecolor=PANEL)
        self.check_mode = CheckButtons(ax_mode, ['Group by Location'], [False])
        style_checks(self.check_mode, check_colors=[ACCENT])
        self.check_mode.on_clicked(self._on_avg_toggle)

        # Combination checkboxes — starts ON-SCREEN
        self.comb_labels = sorted(self.comp_df['Combination'].unique())
        self._comb_onscreen  = [0.72, 0.50, 0.27, 0.37]
        self.ax_comb = self.fig.add_axes(self._comb_onscreen, facecolor=PANEL)
        self.check_comb = CheckButtons(
            self.ax_comb, self.comb_labels, [True] * len(self.comb_labels))
        style_checks(self.check_comb, fontsize=7)
        self.check_comb.on_clicked(self._on_comb)

        # Location checkboxes — starts OFF-SCREEN to avoid event collision
        # (set_visible(False) alone does NOT prevent ax.contains() returning True)
        self.loc_labels = sorted(self.comp_df['LocationGroup'].unique())
        self._loc_onscreen   = [0.72, 0.50, 0.27, 0.37]
        self.ax_loc = self.fig.add_axes(_OFFSCREEN, facecolor=PANEL)
        self.check_loc = CheckButtons(
            self.ax_loc, self.loc_labels, [True] * len(self.loc_labels))
        style_checks(self.check_loc, fontsize=8)
        self.check_loc.on_clicked(self._on_loc)

        b3 = self._make_btn([0.72, 0.44, 0.13, 0.04], 'Select All',
                            self._select_all_filter)
        b4 = self._make_btn([0.86, 0.44, 0.13, 0.04], 'Select None',
                            self._select_none_filter)
        self._btn_refs += [b3, b4]

        ax_note = self.fig.add_axes([0.72, 0.21, 0.27, 0.21], facecolor=PANEL)
        ax_note.axis('off')
        ax_note.text(0.05, 0.97,
                     "Combination codes:\n"
                     "  A / B / WE — location prefix\n"
                     "  N / S / UP / 0 — seating config\n"
                     "  AJC — microphone array\n\n"
                     "Line style:\n"
                     "  ─── Wedges: Yes\n"
                     "  ╌╌╌ Wedges: No\n\n"
                     "Overlays: thick dotted lines\n"
                     "  \u25a0 = summary   \u25b2 = individual",
                     transform=ax_note.transAxes,
                     va='top', color=DIM, fontsize=7, fontfamily='monospace')

    # ── Bottom overlay strip ───────────────────────────────────────────────────

    def _build_overlay_strip(self):
        """
        Three-column strip.  Each column shows summary/average items only
        (keeps the strip compact).  All/None buttons cover the FULL set of
        series for that researcher — including hidden individual measurements.
        """
        _add_header(self.fig, 0.01, 0.185, 0.97, 0.025,
                    'RT60 REFERENCE OVERLAYS  '
                    '(dotted lines on plot  ·  strip shows summaries; '
                    'All/None include all series)',
                    color=WARN)

        group_keys = list(self.reverb_groups.keys())
        n_groups   = len(group_keys)
        col_w      = 0.97 / n_groups
        x0         = 0.01
        strip_y    = 0.02
        strip_h    = 0.155

        self._overlay_check_widgets: dict = {}
        self._overlay_check_labels:  dict = {}  # keys shown in strip

        for i, gk in enumerate(group_keys):
            gdata        = self.reverb_groups[gk]
            x_col        = x0 + i * col_w
            color        = gdata['color']
            display_keys = list(gdata['display_items'].keys())
            all_keys     = list(gdata['all_items'].keys())
            n_hidden     = len(all_keys) - len(display_keys)

            # Group header + item count
            label_text = (f"{gdata['label']}"
                          + (f"  (+{n_hidden} more)" if n_hidden else ''))
            ax_hdr = self.fig.add_axes(
                [x_col, strip_y + strip_h - 0.025, col_w - 0.01, 0.022],
                facecolor=BG)
            ax_hdr.axis('off')
            ax_hdr.text(0.5, 0.5, label_text,
                        transform=ax_hdr.transAxes,
                        ha='center', va='center',
                        color=color, fontsize=7.5, fontweight='bold')

            checks_h  = strip_h - 0.03 - 0.025
            ax_checks = self.fig.add_axes(
                [x_col, strip_y + 0.025, col_w - 0.01, checks_h],
                facecolor=PANEL)

            if display_keys:
                cb = CheckButtons(ax_checks, display_keys,
                                  [False] * len(display_keys))
                style_checks(cb,
                             check_colors=[color] * len(display_keys),
                             fontsize=7)
                cb.on_clicked(self._make_overlay_cb(gk))
                self._overlay_check_widgets[gk] = cb
                self._overlay_check_labels[gk]  = display_keys
            else:
                ax_checks.axis('off')
                ax_checks.text(0.5, 0.5, '(no data)',
                               transform=ax_checks.transAxes,
                               ha='center', color=DIM, fontsize=7)

            bw     = (col_w - 0.01) / 2 - 0.005
            b_all  = self._make_btn(
                [x_col, strip_y, bw, 0.022], 'All',
                self._make_overlay_all_cb(gk, display_keys, all_keys),
                fg=color)
            b_none = self._make_btn(
                [x_col + bw + 0.005, strip_y, bw, 0.022], 'None',
                self._make_overlay_none_cb(gk, display_keys, all_keys),
                fg=DIM)
            self._btn_refs += [b_all, b_none]

    # ─── BUTTON FACTORY ───────────────────────────────────────────────────────

    def _make_btn(self, rect, label, callback, fg=TEXT, fontsize=7):
        """Dark-styled Button.  Caller must store the returned object."""
        ax  = self.fig.add_axes(rect, facecolor=PANEL)
        btn = Button(ax, label, color=PANEL, hovercolor=BORDER)
        btn.label.set_color(fg)
        btn.label.set_fontsize(fontsize)
        btn.on_clicked(callback)
        return btn

    # ─── PANEL SWITCHING (off-screen move) ────────────────────────────────────

    def _show_comb_panel(self):
        """Move comb panel on-screen, loc panel off-screen."""
        self.ax_comb.set_position(self._comb_onscreen)
        self.ax_loc.set_position(_OFFSCREEN)

    def _show_loc_panel(self):
        """Move loc panel on-screen, comb panel off-screen."""
        self.ax_loc.set_position(self._loc_onscreen)
        self.ax_comb.set_position(_OFFSCREEN)

    # ─── CALLBACKS ────────────────────────────────────────────────────────────

    def _on_metric(self, label):
        self.current_metric = label
        self.refresh_plot()

    def _on_wedge(self, label):
        if self._updating:
            return
        # Read current state directly from widget — immune to ordering issues
        status = self.check_wedge.get_status()
        self.selected_wedges = {
            name for name, on in zip(['Yes', 'No'], status) if on
        }
        self.refresh_plot()

    def _on_avg_toggle(self, _label=None):
        self.avg_mode = self.check_mode.get_status()[0]
        if self.avg_mode:
            self._show_loc_panel()
        else:
            self._show_comb_panel()
        self.refresh_plot()

    def _on_comb(self, _label=None):
        """Sync selected_combinations from widget state."""
        if self._updating:
            return
        status = self.check_comb.get_status()
        self.selected_combinations = {
            name for name, on in zip(self.comb_labels, status) if on
        }
        self.refresh_plot()

    def _on_loc(self, _label=None):
        """Sync selected_locations from widget state."""
        if self._updating:
            return
        status = self.check_loc.get_status()
        self.selected_locations = {
            name for name, on in zip(self.loc_labels, status) if on
        }
        self.refresh_plot()

    # ── Bulk select/deselect ──────────────────────────────────────────────────

    def _bulk_set(self, widget, labels, target_set, desired_state):
        """
        Set all checkboxes to desired_state without re-entering callbacks.
        widget.eventson=False suppresses the widget's observer registry.
        self._updating=True suppresses our own guards.
        """
        self._updating    = True
        widget.eventson   = False
        try:
            status = widget.get_status()
            for idx, name in enumerate(labels):
                if status[idx] != desired_state:
                    widget.set_active(idx, state=desired_state)
            if desired_state:
                target_set.update(labels)
            else:
                target_set.difference_update(labels)
        finally:
            widget.eventson = True
            self._updating  = False

    def _select_all_filter(self, _=None):
        if self.avg_mode:
            self._bulk_set(self.check_loc,  self.loc_labels,
                           self.selected_locations,    True)
        else:
            self._bulk_set(self.check_comb, self.comb_labels,
                           self.selected_combinations, True)
        self.refresh_plot()

    def _select_none_filter(self, _=None):
        if self.avg_mode:
            self._bulk_set(self.check_loc,  self.loc_labels,
                           self.selected_locations,    False)
        else:
            self._bulk_set(self.check_comb, self.comb_labels,
                           self.selected_combinations, False)
        self.refresh_plot()

    # ── Overlay callbacks ─────────────────────────────────────────────────────

    def _make_overlay_cb(self, group_key):
        """Per-item overlay toggle — reads full widget state for robustness."""
        def _cb(_label=None):
            if self._updating:
                return
            widget = self._overlay_check_widgets.get(group_key)
            if widget is None:
                return
            display_keys = self._overlay_check_labels[group_key]
            status       = widget.get_status()
            for name, on in zip(display_keys, status):
                key = f"{group_key}|{name}"
                if on:
                    self.selected_overlays.add(key)
                else:
                    self.selected_overlays.discard(key)
            self.refresh_plot()
        return _cb

    def _make_overlay_all_cb(self, group_key, display_keys, all_keys):
        """
        Select ALL series in this group (including hidden individual measurements).
        Visually ticks every displayed checkbox.
        """
        def _cb(_=None):
            widget = self._overlay_check_widgets.get(group_key)
            self._updating = True
            if widget is not None:
                widget.eventson = False
            try:
                # Tick all displayed checkboxes
                if widget is not None:
                    status = widget.get_status()
                    for idx in range(len(display_keys)):
                        if not status[idx]:
                            widget.set_active(idx, state=True)
                # Add ALL keys to selected set (including hidden ones)
                for name in all_keys:
                    self.selected_overlays.add(f"{group_key}|{name}")
            finally:
                if widget is not None:
                    widget.eventson = True
                self._updating = False
            self.refresh_plot()
        return _cb

    def _make_overlay_none_cb(self, group_key, display_keys, all_keys):
        """Deselect ALL series in this group."""
        def _cb(_=None):
            widget = self._overlay_check_widgets.get(group_key)
            self._updating = True
            if widget is not None:
                widget.eventson = False
            try:
                if widget is not None:
                    status = widget.get_status()
                    for idx in range(len(display_keys)):
                        if status[idx]:
                            widget.set_active(idx, state=False)
                for name in all_keys:
                    self.selected_overlays.discard(f"{group_key}|{name}")
            finally:
                if widget is not None:
                    widget.eventson = True
                self._updating = False
            self.refresh_plot()
        return _cb

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_plot(self, _=None):
        """Save the main plot axes as a standalone PNG with a timestamp."""
        base      = os.path.dirname(os.path.abspath(__file__))
        ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f"acoustic_{self.current_metric}_{ts}.png"
        filepath  = os.path.join(base, filename)

        # Render only the main axes (tight crop around it)
        extent = self.ax.get_tightbbox(
            self.fig.canvas.get_renderer()).transformed(
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(filepath, bbox_inches=extent,
                         dpi=200, facecolor=PANEL)
        print(f"Saved: {filepath}")

        # Brief visual feedback on the button label
        for btn in self._btn_refs:
            if hasattr(btn, 'label') and 'Save' in btn.label.get_text():
                original = btn.label.get_text()
                btn.label.set_text('>> Saved!')
                btn.label.set_color(SUCCESS)
                self.fig.canvas.draw_idle()

                def _restore(orig=original, b=btn):
                    import threading
                    import time
                    time.sleep(1.5)
                    b.label.set_text(orig)
                    b.label.set_color(SUCCESS)
                    self.fig.canvas.draw_idle()

                import threading
                threading.Thread(target=_restore, daemon=True).start()
                break

    # ─── PLOT REFRESH ─────────────────────────────────────────────────────────

    def refresh_plot(self):
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        self.ax.set_xscale('log')
        self.ax.grid(True, which='major', color=BORDER,
                     alpha=0.40, linestyle='--', linewidth=0.6)
        self.ax.grid(True, which='minor', color=BORDER,
                     alpha=0.15, linestyle=':', linewidth=0.4)

        sub = self.comp_df[
            (self.comp_df['Metric'] == self.current_metric) &
            (self.comp_df['Wedges?'].isin(self.selected_wedges))
        ]
        x_freqs = [float(f) for f in self.freq_cols]

        # ── Comparison data ────────────────────────────────────
        if self.avg_mode:
            grouped = (sub
                       .groupby(['Wedges?', 'LocationGroup'])[self.freq_cols]
                       .mean()
                       .reset_index())
            for _, row in grouped.iterrows():
                if row['LocationGroup'] not in self.selected_locations:
                    continue
                lbl   = f"{row['Wedges?']} Wedges | {row['LocationGroup']}"
                color = self.color_map.get(row['LocationGroup'], '#ffffff')
                style = '-' if row['Wedges?'] == 'Yes' else '--'
                self.ax.plot(x_freqs, row[self.freq_cols].values,
                             label=lbl, color=color, linestyle=style,
                             marker='o', markersize=5, linewidth=1.8)
        else:
            for _, row in sub.iterrows():
                if row['Combination'] not in self.selected_combinations:
                    continue
                lbl   = f"{row['Wedges?']} | {row['Combination']}"
                color = self.color_map.get(row['Combination'], '#ffffff')
                style = '-' if row['Wedges?'] == 'Yes' else '--'
                self.ax.plot(x_freqs, row[self.freq_cols].values,
                             label=lbl, color=color, linestyle=style,
                             marker='o', markersize=3,
                             linewidth=1.2, alpha=0.75)

        # ── RT60 overlays ──────────────────────────────────────
        for key in sorted(self.selected_overlays):
            group_key, item_name = key.split('|', 1)
            gdata = self.reverb_groups.get(group_key)
            if gdata is None:
                continue
            entry = gdata['all_items'].get(item_name)
            if entry is None:
                continue
            freqs, vals = entry
            color       = gdata['color']
            is_summary  = any(t in item_name
                              for t in ('Avg', 'Average', 'Ideal', 'Calc'))
            self.ax.plot(
                freqs, vals,
                label=f'[{group_key}] {item_name}',
                color=color, linestyle=':',
                linewidth=2.5 if is_summary else 1.5,
                marker='s'   if is_summary else '^',
                markersize=6 if is_summary else 4,
                alpha=0.95)

        # ── Axes formatting ────────────────────────────────────
        unit     = METRIC_UNITS.get(self.current_metric, '')
        unit_str = f' ({unit})' if unit else ''

        self.ax.set_title(f'{self.current_metric}{unit_str}',
                          color=TEXT, fontsize=13, pad=14)
        self.ax.set_xlabel('Frequency (Hz)', color=DIM, fontsize=10)
        self.ax.set_ylabel(f'{self.current_metric}{unit_str}',
                           color=DIM, fontsize=10)

        self.ax.xaxis.set_major_locator(ticker.FixedLocator(FREQ_TICKS))
        self.ax.xaxis.set_minor_locator(ticker.NullLocator())
        self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(FREQ_TICK_LABELS))
        self.ax.set_xlim(45, 12000)
        self.ax.tick_params(axis='x', which='major',
                            labelsize=8, rotation=45, color=BORDER)
        self.ax.tick_params(axis='y', which='both',
                            labelsize=8, color=BORDER)

        # ── Legend ────────────────────────────────────────────
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(
                handles, labels,
                loc='best', fontsize=7,
                facecolor=BG, edgecolor=BORDER, labelcolor=TEXT,
                framealpha=0.85,
                ncol=max(1, len(handles) // 20))

        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    comp_path, rev_path = find_files()

    print(f"matplotlib {matplotlib.__version__}")
    print("Loading comparison data …")
    comp_df, freq_cols = load_comparison(comp_path)

    print("Loading reverb reference data …")
    reverb_groups = load_reverb(rev_path)

    for gk, gd in reverb_groups.items():
        n_disp = len(gd['display_items'])
        n_all  = len(gd['all_items'])
        print(f"  [{gk}] {gd['label']}: {n_disp} displayed / {n_all} total series")

    print(f"Comparison: {len(comp_df)} rows | "
          f"{len(comp_df['Metric'].unique())} metrics | "
          f"{len(comp_df['Combination'].unique())} combinations")

    AcousticDashboard(comp_df, freq_cols, reverb_groups)
    plt.show()
