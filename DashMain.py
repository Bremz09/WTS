#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State
from statistics import mean

# ─── App initialisation ───────────────────────────────────────────────────────
# CSS is served automatically from assets/base.css and assets/shell.css
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
)
server = app.server


# ─── Rider defaults ───────────────────────────────────────────────────────────
# [seat_RPM, seat_torque, seat_CdA, stand_RPM, stand_torque, stand_CdA,
#  mass, sprocket, chainring, seat_height]
PETCH   = [235, 207, 0.2050, 240, 223, 0.2563, 71.9, 15, 54, 0.96]
SHAANE  = [233, 253, 0.2340, 227, 289, 0.2925, 91.8, 15, 62, 1.04]
ELLESSE = [238, 202, 0.2180, 217, 270, 0.2725, 86.9, 15, 63, 1.01]

ORDER_OPTIONS = ["Petch, Shaane, Ellesse", "Shaane, Petch, Ellesse"]
CALC_OPTIONS  = ["Female Team Sprint", "Male Individual Pursuit"]

_RIDER_FIELDS = [
    'seat-rpm', 'seat-torque', 'seat-cda',
    'stand-rpm', 'stand-torque', 'stand-cda',
    'mass', 'sprocket', 'chainring', 'seat-height',
]


# ─── Layout helpers ───────────────────────────────────────────────────────────
def _num_input(id_, label, value, min_val, max_val, step=1.0):
    return html.Div([
        html.Label(label),
        dcc.Input(
            id=id_, type='number', value=value,
            min=min_val, max=max_val, step=step,
            debounce=False,
            style={'width': '100%'},
        ),
    ], className='input-field')


def _rider_section(prefix, label, defaults):
    d = defaults
    return html.Div([
        html.Div(label, className='section-divider'),
        html.Div([
            _num_input(f'{prefix}-seat-rpm',    'Seated Max RPM',     d[0], 0.01, 500,    1.0),
            _num_input(f'{prefix}-seat-torque', 'Seated Max Torque',  d[1], 0.01, 500,    1.0),
            _num_input(f'{prefix}-seat-cda',    'Seated CdA',         d[2], 0.0001, 2.0,  0.0001),
            _num_input(f'{prefix}-stand-rpm',   'Standing Max RPM',   d[3], 0.01, 500,    1.0),
            _num_input(f'{prefix}-stand-torque','Standing Max Torque',d[4], 0.01, 500,    1.0),
            _num_input(f'{prefix}-stand-cda',   'Standing CdA',       d[5], 0.0,  20.0,   0.0001),
            _num_input(f'{prefix}-mass',        'Total Mass (kg)',    d[6], 40.0, 150.0,  0.1),
            _num_input(f'{prefix}-sprocket',    'Sprocket',           d[7], 12,   22,     1),
            _num_input(f'{prefix}-chainring',   'Chain Ring',         d[8], 40,   100,    1),
            _num_input(f'{prefix}-seat-height', 'Seat Height (m)',    d[9], 0.5,  2.0,    0.01),
        ], className='input-group'),
    ])


def _dropdown_field(id_, label, options, value):
    return html.Div([
        html.Label(label),
        dcc.Dropdown(
            id=id_,
            options=[{'label': str(o), 'value': o} for o in options],
            value=value,
            clearable=False,
            className='dash-dropdown',
        ),
    ], className='input-field')


# ── Sidebar-specific input helpers (inline styles bypass CSS cascade issues) ──
_SB_LABEL = {
    'display': 'block', 'fontSize': '0.65rem', 'fontWeight': '700',
    'textTransform': 'uppercase', 'letterSpacing': '0.05em',
    'color': 'rgba(187,188,188,0.8)', 'marginBottom': '4px',
}
_SB_INPUT = {
    'width': '100%', 'padding': '6px 8px',
    'background': 'rgba(255,255,255,0.08)', 'color': '#ffffff',
    'border': '1px solid rgba(255,255,255,0.18)', 'borderRadius': '6px',
    'fontSize': '0.875rem', 'fontFamily': 'Arial, sans-serif',
    'boxSizing': 'border-box',
}
_SB_WRAP = {'marginBottom': '8px'}


def _sb_input(id_, label, value, min_val, max_val, step=1.0):
    return html.Div([
        html.Label(label, style=_SB_LABEL),
        dcc.Input(
            id=id_, type='number', value=value,
            min=min_val, max=max_val, step=step,
            debounce=False, style=_SB_INPUT,
        ),
    ], style=_SB_WRAP)


def _sb_dropdown(id_, label, options, value):
    return html.Div([
        html.Label(label, style=_SB_LABEL),
        dcc.Dropdown(
            id=id_,
            options=[{'label': str(o), 'value': o} for o in options],
            value=value, clearable=False,
            style={'fontSize': '0.875rem', 'color': '#000000'},
        ),
    ], style=_SB_WRAP)


def _main_layout():
    sidebar = html.Aside(className='sidebar', children=[
        # Logo
        html.Div(className='sidebar-logo', children=[
            html.Div([
                html.Div('CNZ', className='sidebar-logo-text'),
                html.Div('Performance Database', className='sidebar-logo-sub'),
            ]),
        ]),
        # Nav
        html.Nav(className='sidebar-nav', children=[
            html.Div('Modelling Tool', className='sidebar-nav-item active'),
        ]),
        # Global parameters
        html.Div(className='sidebar-params', children=[
            html.Div('Global Parameters', className='sidebar-params-title'),
            _sb_dropdown('g-track-circ', 'Track Circumference (m)', [250, 333, 500], 250),
            _sb_input('g-air-density',   'Air Density (kg/m³)',    1.168, 0.001,  3.2,   0.001),
            _sb_input('g-dist-sit',      'Distance at Sit (m)',    150.0, 0.01,   750.0, 0.1),
            _sb_input('g-stand-fat',     'Standing Fatigue (%)',     1.0, 0.01,   99.99, 0.01),
            _sb_input('g-seat-fat',      'Seated Fatigue (%)',       1.0, 0.01,   99.99, 0.01),
            _sb_input('g-fat-onset',     'Fatigue Onset (s)',        1.0,  0.1,    2.0,  0.1),
            _sb_input('g-straight-bank', 'Straight Bank Angle (°)', 13.00, 0.0, 90.0, 0.01),
            _sb_input('g-bend-bank',     'Bend Bank Angle (°)',     46.13, 0.0, 90.0, 0.01),
            _sb_input('g-pl-trans',      'PL to Transition (m)',    31.25, 0.0, 90.0, 0.01),
            _sb_input('g-trans-len',     'Transition Length (m)',   10.00, 0.0, 90.0, 0.01),
            html.Button('Run Simulation', id='submit-btn', n_clicks=0,
                        style={'marginTop': '12px', 'width': '100%'}),
        ]),
        # Bottom
        html.Div(className='sidebar-bottom', children=[
            html.P('CNZ Performance DB'),
            html.P('Dash App'),
        ]),
    ])

    main = html.Div(className='main-area', children=[
        # Top bar
        html.Header(className='top-bar', children=[
            html.Span('Modelling Tool', className='top-bar-title'),
            html.Span('Female Team Sprint Simulator',
                      style={'fontSize': '0.8rem', 'color': 'var(--color-text-muted)'}),
        ]),

        # Page content
        html.Div(className='page-content', children=[

            # ── Model selection card ──
            html.Div(className='card', children=[
                html.Div('Simulation Settings', className='card-title'),
                html.Div(className='input-group', children=[
                    _dropdown_field('calc-select',  'Model',        CALC_OPTIONS,  CALC_OPTIONS[0]),
                    _dropdown_field('order-select', 'Rider Order',  ORDER_OPTIONS, ORDER_OPTIONS[0]),
                ]),
            ]),

            # ── Rider specs card ──
            html.Div(className='card', children=[
                html.Div('Rider Specifications', className='card-title'),
                html.Div(id='rider-sections'),
            ]),

            # ── Results ──
            html.Div(id='results-container'),
        ]),
    ])

    return html.Div(className='shell-wrapper', children=[sidebar, main])


# ─── Top-level app layout ─────────────────────────────────────────────────────
app.layout = _main_layout()


# ─── Populate rider sections when order changes ───────────────────────────────
@app.callback(
    Output('rider-sections', 'children'),
    Input('order-select', 'value'),
    prevent_initial_call=False,
)
def update_rider_sections(order):
    if order == ORDER_OPTIONS[0]:
        names    = ["Rider 1 — Petch",  "Rider 2 — Shaane", "Rider 3 — Ellesse"]
        defaults = [PETCH, SHAANE, ELLESSE]
    else:
        names    = ["Rider 1 — Shaane", "Rider 2 — Petch",  "Rider 3 — Ellesse"]
        defaults = [SHAANE, PETCH, ELLESSE]
    return html.Div([
        _rider_section('r1', names[0], defaults[0]),
        _rider_section('r2', names[1], defaults[1]),
        _rider_section('r3', names[2], defaults[2]),
    ])


# ─── Main simulation callback ─────────────────────────────────────────────────
_R_STATES = [State(f'r{n}-{f}', 'value') for n in (1, 2, 3) for f in _RIDER_FIELDS]
_G_STATES = [
    State('g-air-density',   'value'),
    State('g-dist-sit',      'value'),
    State('g-stand-fat',     'value'),
    State('g-seat-fat',      'value'),
    State('g-fat-onset',     'value'),
    State('g-track-circ',    'value'),
    State('g-straight-bank', 'value'),
    State('g-bend-bank',     'value'),
    State('g-pl-trans',      'value'),
    State('g-trans-len',     'value'),
]


@app.callback(
    Output('results-container', 'children'),
    Input('submit-btn', 'n_clicks'),
    _R_STATES + _G_STATES,
    prevent_initial_call=True,
)
def run_simulation(_n, *args):
    # ── Unpack inputs ──────────────────────────────────────────────────────────
    r_vals = list(args[:30])
    g_vals = list(args[30:])

    (seat_max_RPM_1, seat_max_torque_1, seat_CdA_1,
     stand_max_RPM_1, stand_max_torque_1, stand_CdA_1,
     total_mass_1, sprocket_1, chainring_1, seat_height_1) = r_vals[0:10]

    (seat_max_RPM_2, seat_max_torque_2, seat_CdA_2,
     stand_max_RPM_2, stand_max_torque_2, stand_CdA_2,
     total_mass_2, sprocket_2, chainring_2, seat_height_2) = r_vals[10:20]

    (seat_max_RPM_3, seat_max_torque_3, seat_CdA_3,
     stand_max_RPM_3, stand_max_torque_3, stand_CdA_3,
     total_mass_3, sprocket_3, chainring_3, seat_height_3) = r_vals[20:30]

    (air_density, dist_at_sit, standing_fatigue_rate, seated_fatigue_rate,
     fatigue_onset, _track_circ, straight_bank_angle, bend_bank_angle,
     pl_to_trans, transition_length) = g_vals

    # ── Constants ─────────────────────────────────────────────────────────────
    wheel_circ, bike_length = 2.096, 1.7122
    ks, mu_rr, lean_smoothing, increment, efficiency = 0.0072, 0.0016, 1, 0.1, 0.97
    rad_of_curve = (250 - 4 * pl_to_trans) / (2 * math.pi)
    deg_to_rad, rad_to_deg = math.pi / 180, 180 / math.pi

    # ── Athlete class ─────────────────────────────────────────────────────────
    class Athlete:
        def __init__(self, seat_max_RPM, seat_max_torque, stand_max_RPM, stand_max_torque,
                     stand_CdA, seat_CdA, total_mass, gear, seat_height, max_power,
                     stand_TC_slope, seat_TC_slope):
            self.seat_max_RPM = seat_max_RPM; self.seat_max_torque = seat_max_torque
            self.stand_max_RPM = stand_max_RPM; self.stand_max_torque = stand_max_torque
            self.stand_CdA = stand_CdA; self.seat_CdA = seat_CdA
            self.total_mass = total_mass; self.gear = gear
            self.seat_height = seat_height; self.max_power = max_power
            self.stand_TC_slope = stand_TC_slope; self.seat_TC_slope = seat_TC_slope

        def initialize_state(self, initial_speed, bank_angle, rad_of_curve,
                             air_density, mu_rr, ks, efficiency, bike_length):
            self.time = 0; self.COM_speed = initial_speed; self.COM_dist = 0
            self.CdA = self.stand_CdA; self.cadence = 0; self.torque = self.stand_max_torque
            self.power_input = self.cadence * self.torque * (math.pi / 30)
            self.power_usable = self.power_input * efficiency; self.acc_fatigue = 0
            self.bank = bank_angle; self.lean = 0
            self.camber = abs(self.bank - self.lean)
            self.r_wh = 2 * rad_of_curve; self.r_cm = 2 * rad_of_curve
            self.prop_force = 2 * math.pi * self.torque / (2.096 * (self.gear / 27))
            self.aero_drag = 0.5 * air_density * self.CdA * self.COM_speed ** 2
            self.weight_force = 9.81 * self.total_mass; self.centripetal_force = 0
            self.reaction_force = math.sqrt(self.weight_force ** 2 + self.centripetal_force ** 2)
            self.normal_force = self.reaction_force * math.cos(math.radians(self.camber))
            self.rr = self.normal_force * mu_rr * (1 + (self.camber * ks))
            self.wheel_speed = 0; self.wheel_dist = 0
            self.segment = self.wheel_dist % 125
            self.accel = (self.prop_force - (self.rr + self.aero_drag)) / self.total_mass
            self.air_speed = 0; self.gap = -bike_length

    def make_athlete(rpm_s, tor_s, rpm_st, tor_st, cda_st, cda_s, mass, chain, spr, sh):
        return Athlete(rpm_s, tor_s, rpm_st, tor_st, cda_st, cda_s, mass,
                       27 * chain / spr, sh, rpm_s * tor_s * math.pi / 120,
                       -tor_st / rpm_st, -tor_s / rpm_s)

    p1 = make_athlete(seat_max_RPM_1, seat_max_torque_1, stand_max_RPM_1, stand_max_torque_1,
                      stand_CdA_1, seat_CdA_1, total_mass_1, chainring_1, sprocket_1, seat_height_1)
    p2 = make_athlete(seat_max_RPM_2, seat_max_torque_2, stand_max_RPM_2, stand_max_torque_2,
                      stand_CdA_2, seat_CdA_2, total_mass_2, chainring_2, sprocket_2, seat_height_2)
    p3 = make_athlete(seat_max_RPM_3, seat_max_torque_3, stand_max_RPM_3, stand_max_torque_3,
                      stand_CdA_3, seat_CdA_3, total_mass_3, chainring_3, sprocket_3, seat_height_3)

    for rider, spd in ((p1, 2), (p2, 2), (p3, 1.6)):
        rider.initialize_state(spd, straight_bank_angle, rad_of_curve,
                               air_density, mu_rr, ks, efficiency, bike_length)

    # ── Track geometry ────────────────────────────────────────────────────────
    def get_bank_lean_camber(segment, lean_initial, v_com, seat_height):
        bend_length = 125 - 2 * (pl_to_trans + transition_length)
        r_wh = rad_of_curve
        if (segment < pl_to_trans) or (segment > 125 - pl_to_trans):
            bank = straight_bank_angle
            r_wh = r_cm = 100000
        elif segment <= pl_to_trans + transition_length:
            pct = (segment - pl_to_trans) / transition_length
            bank = straight_bank_angle + pct * (bend_bank_angle - straight_bank_angle)
            r_wh = 2 * rad_of_curve - pct * rad_of_curve
        elif segment <= pl_to_trans + transition_length + bend_length:
            bank = bend_bank_angle
        else:
            pct = (segment - (pl_to_trans + transition_length + bend_length)) / transition_length
            bank = bend_bank_angle + pct * (straight_bank_angle - bend_bank_angle)
            r_wh = rad_of_curve + pct * rad_of_curve
        lean = rad_to_deg * math.atan(
            v_com ** 2 / (9.81 * (r_wh - seat_height * math.sin(deg_to_rad * lean_initial)))
        )
        while lean - lean_initial > 0.1:
            lean_initial = lean
            lean = rad_to_deg * math.atan(
                v_com ** 2 / (9.81 * (r_wh - seat_height * math.sin(deg_to_rad * lean)))
            )
        r_cm = r_wh - seat_height * math.sin(deg_to_rad * lean) if r_wh < 2 * rad_of_curve else r_wh
        return bank, r_wh, r_cm, lean, bank - lean

    # ── Simulation helpers ────────────────────────────────────────────────────
    _BASE_FIELDS = (
        'COM_speed', 'COM_dist', 'bank', 'r_wh', 'r_cm', 'lean', 'camber',
        'wheel_speed', 'wheel_dist', 'cadence', 'torque', 'power_input',
        'power_usable', 'prop_force', 'aero_drag', 'weight_force', 'segment',
        'centripetal_force', 'reaction_force', 'normal_force', 'rr', 'accel',
    )

    def new_data(rider, with_demand=False):
        d = {'time': [rider.time]}
        d.update({k: [getattr(rider, k)] for k in _BASE_FIELDS})
        d['gap'] = [getattr(rider, 'gap', 0)]
        d['air_speed'] = [getattr(rider, 'air_speed', 0)]
        if with_demand:
            d.update({k: [getattr(rider, k, 1 if k == 'dem_sup' else 0)]
                      for k in ('accel_demand', 'rr_demand', 'aero_demand', 'power_demand', 'dem_sup')})
        return d

    def append_state(rider, d, with_demand=False):
        d['time'].append(rider.time)
        for k in _BASE_FIELDS:
            d[k].append(getattr(rider, k))
        d['gap'].append(getattr(rider, 'gap', 0))
        d['air_speed'].append(getattr(rider, 'air_speed', 0))
        if with_demand:
            for k in ('accel_demand', 'rr_demand', 'aero_demand', 'power_demand', 'dem_sup'):
                d[k].append(getattr(rider, k))

    def update_lean(rider, leans):
        rider.bank, rider.r_wh, rider.r_cm, rider.lean, rider.camber = \
            get_bank_lean_camber(rider.segment, rider.lean, rider.COM_speed, rider.seat_height)
        leans.append(rider.lean)
        if len(leans) > lean_smoothing:
            leans[:] = leans[1:]
        rider.lean = mean(leans)

    def update_kinematics(rider):
        rider.wheel_speed = rider.COM_speed * (rider.r_wh / rider.r_cm)
        rider.wheel_dist += rider.wheel_speed * increment
        rider.cadence = 60 * rider.wheel_speed / ((rider.gear / 27) * wheel_circ)

    def update_forces(rider, cda):
        rider.power_input = rider.cadence * rider.torque * (math.pi / 30)
        rider.power_usable = rider.power_input * efficiency
        rider.prop_force = 2 * math.pi * efficiency * rider.torque / (2.096 * (rider.gear / 27))
        rider.aero_drag = 0.5 * air_density * cda * rider.COM_speed ** 2
        rider.weight_force = 9.81 * rider.total_mass
        rider.segment = rider.wheel_dist % 125
        rider.centripetal_force = (
            0 if (rider.segment < pl_to_trans or rider.segment > 125 - pl_to_trans)
            else (rider.total_mass * rider.COM_speed ** 2) / rider.r_cm
        )
        rider.reaction_force = math.sqrt(rider.weight_force ** 2 + rider.centripetal_force ** 2)
        rider.normal_force = rider.reaction_force * math.cos(deg_to_rad * rider.camber)
        rider.rr = rider.normal_force * mu_rr * (1 + abs(rider.camber) * ks)
        rider.accel = (rider.prop_force - (rider.rr + rider.aero_drag)) / rider.total_mass

    def make_df(d, extra_cols=()):
        cols = ['time'] + list(_BASE_FIELDS) + ['gap', 'air_speed'] + list(extra_cols)
        return pd.DataFrame({c: d[c] for c in cols}).rename(columns={'time': 'Time'})

    def intp(xval, df, xcol, ycol):
        return np.interp([xval], df[xcol], df[ycol])

    # ── P1 simulation ─────────────────────────────────────────────────────────
    p1d = new_data(p1)
    p1_leans = []
    for is_standing, stop_dist in [(True, dist_at_sit), (False, 250)]:
        if not is_standing:
            p1.CdA = p1.seat_CdA
        while p1.wheel_dist < stop_dist:
            p1.time += increment
            p1.COM_speed += increment * p1.accel
            p1.COM_dist += p1.COM_speed * increment
            update_lean(p1, p1_leans)
            update_kinematics(p1)
            if is_standing:
                if p1.time < fatigue_onset:
                    p1.torque = p1.stand_max_torque + p1.stand_TC_slope * p1.cadence
                else:
                    p1.acc_fatigue += increment * standing_fatigue_rate / 100
                    p1.torque = p1.stand_max_torque * (1 - p1.acc_fatigue) + p1.stand_TC_slope * p1.cadence
                update_forces(p1, p1.stand_CdA)
            else:
                p1.acc_fatigue += increment * seated_fatigue_rate / 100
                p1.torque = p1.seat_max_torque * (1 - p1.acc_fatigue) + p1.seat_TC_slope * p1.cadence
                update_forces(p1, p1.seat_CdA)
            append_state(p1, p1d)
    df_p1 = make_df(p1d)

    # ── P2 simulation ─────────────────────────────────────────────────────────
    p2d = new_data(p2)
    p2_leans = []
    count = 0
    for is_standing, stop_dist in [(True, dist_at_sit), (False, 500)]:
        if not is_standing:
            p2.CdA = p2.seat_CdA
        while p2.wheel_dist < stop_dist:
            p2.time += increment
            p2.COM_speed += increment * p2.accel
            p2.COM_dist += p2.COM_speed * increment
            update_lean(p2, p2_leans)
            update_kinematics(p2)
            p2.gap = df_p1["wheel_dist"][count] - p2.wheel_dist - bike_length if count < len(df_p1) else 0
            p2.air_speed = p2.COM_speed
            cda = p2.stand_CdA if is_standing else p2.seat_CdA
            if is_standing:
                if p2.time < fatigue_onset:
                    p2.torque = p2.stand_max_torque + p2.stand_TC_slope * p2.cadence
                else:
                    p2.acc_fatigue += increment * standing_fatigue_rate / 100
                    p2.torque = p2.stand_max_torque * (1 - p2.acc_fatigue) + p2.stand_TC_slope * p2.cadence
            else:
                p2.acc_fatigue += increment * seated_fatigue_rate / 100
                p2.torque = p2.seat_max_torque * (1 - p2.acc_fatigue) + p2.seat_TC_slope * p2.cadence
            update_forces(p2, cda)
            if p2.gap > 0.2:
                p2.aero_drag *= (100 - (-8.1136 * p2.gap + 50.051)) / 100
                p2.accel = (p2.prop_force - (p2.rr + p2.aero_drag)) / p2.total_mass
            count += 1
            append_state(p2, p2d)
    df_p2 = make_df(p2d)

    # ── P3 simulation ─────────────────────────────────────────────────────────
    p3.dem_sup = 1; p3.accel_demand = 0; p3.rr_demand = 0; p3.aero_demand = 0; p3.power_demand = 0
    p3d = new_data(p3, with_demand=True)
    p3_leans = []
    count = 1
    for is_standing, stop_dist in [(True, dist_at_sit), (False, 750)]:
        if not is_standing:
            p3.CdA = p3.seat_CdA
        while p3.wheel_dist < stop_dist:
            if count < len(df_p2):
                p3.accel_demand = p3.total_mass * df_p2["accel"][count] * df_p2["wheel_speed"][count]
                p3.rr_demand    = p3.rr * df_p2["wheel_speed"][count]
                p3.aero_demand  = 0.5 * air_density * p3.CdA * p3.COM_speed * p3.air_speed ** 2
                p3.power_demand = p3.accel_demand + p3.rr_demand + p3.aero_demand
            else:
                p3.accel_demand = p3.rr_demand = p3.aero_demand = p3.power_demand = 0
            p3.time += increment
            p3.COM_speed += increment * p3.accel
            p3.COM_dist += p3.COM_speed * increment
            update_lean(p3, p3_leans)
            update_kinematics(p3)
            p3.gap = df_p2["wheel_dist"][count] - p3.wheel_dist - bike_length if count < len(df_p2) else 0
            p3.air_speed = p3.COM_speed
            cda = p3.stand_CdA if is_standing else p3.seat_CdA
            if is_standing:
                if p3.time < fatigue_onset:
                    p3.torque = p3.stand_max_torque + p3.stand_TC_slope * p3.cadence
                else:
                    p3.acc_fatigue += increment * p3.dem_sup * standing_fatigue_rate / 100
                    p3.torque = p3.stand_max_torque * (1 - p3.acc_fatigue) + p3.stand_TC_slope * p3.cadence
            else:
                p3.acc_fatigue += increment * p3.dem_sup * seated_fatigue_rate / 100
                p3.torque = p3.seat_max_torque * (1 - p3.acc_fatigue) + p3.seat_TC_slope * p3.cadence
            update_forces(p3, cda)
            p3.dem_sup = p3.power_demand / p3.power_usable if p3.power_usable > p3.power_demand else 1
            if p3.gap > 0.2:
                p3.aero_drag *= (100 - (-8.1136 * p3.gap + 50.051)) / 100
                p3.accel = (p3.prop_force - (p3.rr + p3.aero_drag)) / p3.total_mass
            count += 1
            append_state(p3, p3d, with_demand=True)
    df_p3 = make_df(p3d, extra_cols=('accel_demand', 'rr_demand', 'aero_demand', 'power_demand', 'dem_sup'))

    # ── Summary tables ────────────────────────────────────────────────────────
    dists_p1 = [62.5, 125, 187.5, 250]
    dists_p2 = dists_p1 + [312.5, 375, 437.5, 500]
    dists_p3 = dists_p2 + [562.5, 625, 687.5, 750]

    def qt(d, df_px):
        return round(intp(d, df_px, 'wheel_dist', 'Time')[0], 3)

    p1_qt = {d: qt(d, df_p1) for d in dists_p1}
    p2_qt = {d: qt(d, df_p2) for d in dists_p2}
    p3_qt = {d: qt(d, df_p3) for d in dists_p3}

    def _table(df, table_id=None):
        kwargs = {'id': table_id} if table_id else {}
        return dash_table.DataTable(
            **kwargs,
            data=df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '7px 10px',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '0.8rem',
                'color': '#333333',
                'border': 'none',
                'borderBottom': '1px solid #e0e0e0',
            },
            style_header={
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '0.7rem',
                'fontWeight': '700',
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em',
                'color': '#666666',
                'backgroundColor': '#f8fafc',
                'borderBottom': '2px solid #e0e0e0',
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
            ],
        )

    df_time = pd.DataFrame([1, 2, 3], columns=["Time"])
    for d in dists_p1:
        df_time[str(d)] = [p1_qt[d], p2_qt[d], p3_qt[d]]
    for d in [312.5, 375, 437.5, 500]:
        df_time[str(d)] = [0, p2_qt[d], p3_qt[d]]
    for d in [562.5, 625, 687.5, 750]:
        df_time[str(d)] = [0, 0, p3_qt[d]]

    df_gap = pd.DataFrame([2, 3], columns=["Time_gap"])
    for d in dists_p1:
        df_gap[str(d)] = [p2_qt[d] - p1_qt[d], p3_qt[d] - p2_qt[d]]
    for d in [312.5, 375, 437.5, 500]:
        df_gap[str(d)] = [0, p3_qt[d] - p2_qt[d]]

    df_dist_gap = pd.DataFrame([2, 3], columns=["Dist_gap"])
    for d in dists_p1:
        df_dist_gap[str(d)] = [round(intp(p1_qt[d], df_p2, 'Time', 'gap')[0], 2),
                               round(intp(p2_qt[d], df_p3, 'Time', 'gap')[0], 2)]
    for d in [312.5, 375, 437.5, 500]:
        df_dist_gap[str(d)] = [0, round(intp(p2_qt[d], df_p3, 'Time', 'gap')[0], 2)]

    def make_summary_df(field, label):
        df_s = pd.DataFrame([1, 2, 3], columns=[label])
        for d in dists_p1:
            df_s[str(d)] = [round(intp(p1_qt[d], df_p1, 'Time', field)[0], 2),
                            round(intp(p2_qt[d], df_p2, 'Time', field)[0], 2),
                            round(intp(p3_qt[d], df_p3, 'Time', field)[0], 2)]
        for d in [312.5, 375, 437.5, 500]:
            df_s[str(d)] = [0,
                            round(intp(p2_qt[d], df_p2, 'Time', field)[0], 2),
                            round(intp(p3_qt[d], df_p3, 'Time', field)[0], 2)]
        for d in [562.5, 625, 687.5, 750]:
            df_s[str(d)] = [0, 0, round(intp(p3_qt[d], df_p3, 'Time', field)[0], 2)]
        return df_s

    df_cadence    = make_summary_df('cadence',     'Cadence')
    df_wheel_speed = make_summary_df('wheel_speed', 'wheel_speed')
    df_wheel_speed = df_wheel_speed.apply(lambda x: x * 3.6)
    df_wheel_speed["wheel_speed"] = [1, 2, 3]

    # ── Per-rider figures ─────────────────────────────────────────────────────
    def power_speed_fig(df_px, label, color='#DB6B30'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            mode='lines', x=df_px["Time"], y=df_px["power_usable"],
            name=f"{label} Power", yaxis='y',
            line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            mode='lines', x=df_px["Time"], y=df_px["wheel_speed"],
            name=f"{label} Wheel speed", yaxis="y2",
            line=dict(color='#0E4174', width=2, dash='dot'),
        ))
        fig.update_layout(
            xaxis=dict(domain=[0.0, 0.94], title='Time (s)',
                       gridcolor='#e0e0e0', linecolor='#e0e0e0'),
            yaxis=dict(title=dict(text="Power (W)", font=dict(color=color)),
                       tickfont=dict(color=color), gridcolor='#e0e0e0'),
            yaxis2=dict(title=dict(text="Wheel Speed (m/s)", font=dict(color='#0E4174')),
                        tickfont=dict(color='#0E4174'),
                        overlaying="y", side="right", position=0.94),
            title_text=f"{label} — Power & Wheel Speed",
            title_font=dict(size=13, color='#000000'),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(t=40, b=40, l=50, r=60),
            font=dict(family='Arial, sans-serif', size=11),
        )
        return fig

    def time_to(dist, df_px):
        row = df_px.iloc[(df_px['wheel_dist'] - dist).abs().argsort()[:2]].reset_index(drop=True)
        return row["Time"][1] + (dist - row["wheel_dist"][1]) / row["wheel_speed"][1]

    p1_250_time = (
        df_p1["Time"][len(df_p1) - 2]
        + (250 - df_p1["wheel_dist"][len(df_p1) - 2]) / df_p1["wheel_speed"][len(df_p1) - 2]
    )

    fig_all = go.Figure()
    _colors = ['#DB6B30', '#0E4174', '#00AEC7']
    for (df_px, label), col in zip([(df_p1, 'P1'), (df_p2, 'P2'), (df_p3, 'P3')], _colors):
        fig_all.add_trace(go.Scatter(
            mode='lines', x=df_px["Time"], y=df_px["power_usable"],
            name=f"{label} Power", yaxis='y',
            line=dict(color=col, width=2),
        ))
        fig_all.add_trace(go.Scatter(
            mode='lines', x=df_px["Time"], y=df_px["wheel_speed"],
            name=f"{label} Speed", yaxis='y2',
            line=dict(color=col, width=1.5, dash='dot'),
        ))
    fig_all.update_layout(
        xaxis=dict(domain=[0.0, 0.94], title='Time (s)', gridcolor='#e0e0e0'),
        yaxis=dict(title='Power (W)', gridcolor='#e0e0e0'),
        yaxis2=dict(title='Wheel Speed (m/s)', overlaying='y', side='right', position=0.94),
        title_text='All Riders — Power & Wheel Speed',
        title_font=dict(size=13),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        legend=dict(orientation='h', y=-0.18),
        margin=dict(t=40, b=50, l=50, r=60),
        font=dict(family='Arial, sans-serif', size=11),
    )

    # ── Assemble results layout ───────────────────────────────────────────────
    def _stat_chip(label, value):
        return html.Span([html.Span(f'{label}: '), value], className='stat-chip')

    return html.Div([
        html.Div(className='card', children=[
            html.Div('Split Times (s)', className='card-title'),
            _table(df_time),
        ]),
        html.Div(className='card', children=[
            html.Div('Time Gaps (s)', className='card-title'),
            _table(df_gap),
        ]),
        html.Div(className='card', children=[
            html.Div('Distance Gaps (m)', className='card-title'),
            _table(df_dist_gap),
        ]),
        html.Div(className='card', children=[
            html.Div('Cadence at Splits (RPM)', className='card-title'),
            _table(df_cadence),
        ]),
        html.Div(className='card', children=[
            html.Div('Wheel Speed at Splits (km/h)', className='card-title'),
            _table(df_wheel_speed),
        ]),

        html.Div(className='card', children=[
            html.Div('Rider 1', className='card-title'),
            _stat_chip('250m', f'{round(p1_250_time, 3)} s'),
            dcc.Graph(figure=power_speed_fig(df_p1, 'P1', '#DB6B30'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(className='card', children=[
            html.Div('Rider 2', className='card-title'),
            _stat_chip('250m', f'{round(time_to(250, df_p2), 3)} s'),
            _stat_chip('500m', f'{round(time_to(500, df_p2), 3)} s'),
            dcc.Graph(figure=power_speed_fig(df_p2, 'P2', '#0E4174'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(className='card', children=[
            html.Div('Rider 3', className='card-title'),
            _stat_chip('250m', f'{round(time_to(250, df_p3), 3)} s'),
            _stat_chip('500m', f'{round(time_to(500, df_p3), 3)} s'),
            _stat_chip('750m', f'{round(time_to(750, df_p3), 3)} s'),
            dcc.Graph(figure=power_speed_fig(df_p3, 'P3', '#00AEC7'),
                      config={'displayModeBar': False}),
        ]),
        html.Div(className='card', children=[
            html.Div('All Riders — Power & Speed', className='card-title'),
            dcc.Graph(figure=fig_all, config={'displayModeBar': False}),
        ]),
    ])


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050)
