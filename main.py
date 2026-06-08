#!/usr/bin/env python
# coding: utf-8

import math, datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean
import matplotlib

st.set_page_config(page_title='TS Modelling Tool', page_icon=":bike:", layout="wide")


st.header("Modelling Tool")
update = datetime.date.today() + pd.DateOffset(hour=12)

def intp(xval, df, xcol, ycol):
    return np.interp([xval], df[xcol], df[ycol])

# Default specs: [seat_RPM, seat_torque, seat_CdA, stand_RPM, stand_torque, stand_CdA, mass, sprocket, chainring, seat_height, stand_fatigue_rate(Nm/s), seat_fatigue_rate(Nm/s)]
RIDER_SPECS = {
    "Petch":   [235, 207, 0.2050, 240, 223, 0.2563, 71.9, 15, 54, 0.96, 2.2, 2.1],
    "Shaane":  [233, 253, 0.2340, 227, 289, 0.2925, 91.8, 15, 62, 1.04, 2.9, 2.5],
    "Ellesse": [238, 202, 0.2180, 217, 270, 0.2725, 86.9, 15, 63, 1.01, 2.7, 2.0],
}
rider_options = list(RIDER_SPECS.keys())

c1, c2, c3 = st.columns(3)
p1_name = c1.selectbox("P1:", rider_options, index=0, key="P1_selector")
p2_name = c2.selectbox("P2:", rider_options, index=1, key="P2_selector")
p3_name = c3.selectbox("P3:", rider_options, index=2, key="P3_selector")

rider_names = [p1_name, p2_name, p3_name]
if len(set(rider_names)) < 3:
    st.error("Each position must have a different rider.")
    st.stop()
rider_defaults = [RIDER_SPECS[n] for n in rider_names]

# Apply any pending gear overrides queued by "Use these gears" before widgets are created.
_pending = st.session_state.pop("pending_gears", None)
if _pending:
    for kn, name, cr, sp in _pending:
        if name == rider_names[int(kn) - 1]:
            st.session_state[f"{kn}_{name}_8"] = int(sp)
            st.session_state[f"{kn}_{name}_9"] = int(cr)

def rider_inputs(name, kn, d):
    st.subheader(f"{name} specs")
    pfx = f"{kn}_{name}"
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    v = [
        c1.number_input("Seated Max RPM:",    min_value=0.01, max_value=500.0,  value=float(d[0]), key=f"{pfx}_1"),
        c2.number_input("Seated Max Torque:", min_value=0.01, max_value=500.0,  value=float(d[1]), key=f"{pfx}_2"),
        c3.number_input("Seated CdA:",        min_value=0.0001, max_value=2.0,  value=float(d[2]), step=1e-4, format="%.4f", key=f"{pfx}_3"),
        c4.number_input("Standing Max RPM:",  min_value=0.01, max_value=500.0,  value=float(d[3]), key=f"{pfx}_4"),
        c5.number_input("Standing Max Torque:", min_value=0.01, max_value=500.0, value=float(d[4]), key=f"{pfx}_5"),
        c6.number_input("Standing CdA:",      min_value=0.0,  max_value=20.0,   value=float(d[5]), step=1e-4, format="%.4f", key=f"{pfx}_6"),
    ]
    c1, c2, c3, c4 = st.columns(4)
    v += [
        c1.number_input("Total Mass:",  min_value=40.0, max_value=150.0, value=float(d[6]), step=0.1, format="%.1f", key=f"{pfx}_7"),
        c2.number_input("Sprocket:",    min_value=12,   max_value=22,    value=int(d[7]),   step=1,   key=f"{pfx}_8"),
        c3.number_input("Chain Ring:",  min_value=40,   max_value=100,   value=int(d[8]),   step=1,   key=f"{pfx}_9"),
        c4.number_input("Seat Height:", min_value=0.50, max_value=2.00,  value=float(d[9]),            key=f"{pfx}_10"),
    ]
    c1, c2 = st.columns(2)
    v += [
        c1.number_input("Standing Fatigue Rate (Nm/s):", min_value=0.0, max_value=100.0, value=float(d[10]), step=0.1, format="%.1f", key=f"{pfx}_11"),
        c2.number_input("Seated Fatigue Rate (Nm/s):",   min_value=0.0, max_value=100.0, value=float(d[11]), step=0.1, format="%.1f", key=f"{pfx}_12"),
    ]
    return v

with st.sidebar:
    st.subheader("Global specs")
    air_density           = st.number_input("Air Density:",             min_value=0.001, max_value=3.2,   value=1.168, step=1e-3, format="%.3f", key="4_1")
    dist_at_sit           = st.number_input("Distance at sit:",         min_value=0.01,  max_value=750.0, value=150.0, step=0.1,  format="%.1f", key="4_2")
    fatigue_onset         = st.number_input("Onset of Fatigue (s):",    min_value=0.1,   max_value=2.0,   value=1.0,   step=0.1,  format="%.1f", key="4_5")
    track_circumference   = st.selectbox("Track Circumference:",        [250, 333, 500],                              key="Track_circumference")
    straight_bank_angle   = st.number_input("Straight Bank Angle:",     min_value=0.0, max_value=90.0, value=13.00)
    bend_bank_angle       = st.number_input("Bend Bank Angle:",         min_value=0.0, max_value=90.0, value=46.13)
    pl_to_trans           = st.number_input("Distance from Pursuit Line to Transition:", min_value=0.0, max_value=90.0, value=31.25)
    transition_length     = st.number_input("Transition length:",       min_value=0.0, max_value=90.0, value=10.00)

with st.form("my_form"):
    r1 = rider_inputs(rider_names[0], "1", rider_defaults[0])
    r2 = rider_inputs(rider_names[1], "2", rider_defaults[1])
    r3 = rider_inputs(rider_names[2], "3", rider_defaults[2])
    bc1, bc2 = st.columns(2)
    submitted = bc1.form_submit_button("Update Specs")
    optimize = bc2.form_submit_button("Calculate Optimal Gears")

opt_c1, opt_c2 = st.columns(2)
opt_span = opt_c1.number_input("Span:", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="opt_span")
_MAX_COMBOS = 5000
_max_n = int(_MAX_COMBOS ** (1 / 3))  # max values per rider so n^3 <= cap
_min_inc = math.ceil((2 * opt_span / (_max_n - 1)) * 100) / 100
opt_inc = opt_c2.number_input("Increment:", min_value=_min_inc, max_value=float(2 * opt_span),
                              value=max(_min_inc, 0.5), step=0.05, format="%.2f", key="opt_inc")

(seat_max_RPM_1, seat_max_torque_1, seat_CdA_1, stand_max_RPM_1, stand_max_torque_1, stand_CdA_1, total_mass_1, sprocket_1, chainring_1, seat_height_1, stand_fatigue_rate_1, seat_fatigue_rate_1) = r1
(seat_max_RPM_2, seat_max_torque_2, seat_CdA_2, stand_max_RPM_2, stand_max_torque_2, stand_CdA_2, total_mass_2, sprocket_2, chainring_2, seat_height_2, stand_fatigue_rate_2, seat_fatigue_rate_2) = r2
(seat_max_RPM_3, seat_max_torque_3, seat_CdA_3, stand_max_RPM_3, stand_max_torque_3, stand_CdA_3, total_mass_3, sprocket_3, chainring_3, seat_height_3, stand_fatigue_rate_3, seat_fatigue_rate_3) = r3

@st.cache_data
def _run_sim(r1t, r2t, r3t,
                 air_density, dist_at_sit, fatigue_onset,
                 straight_bank_angle, bend_bank_angle, pl_to_trans, transition_length):
        """Run the full simulation. All args are primitives so st.cache_data can hash them."""
        wheel_circ, bike_length = 2.096, 1.7122
        ks, mu_rr, lean_smoothing, increment, efficiency = 0.0072, 0.0016, 1, 0.1, 0.97
        rad_of_curve = (250 - 4 * pl_to_trans) / (2 * math.pi)
        deg_to_rad, rad_to_deg = math.pi / 180, 180 / math.pi

        class Athlete:
            def __init__(self, seat_max_RPM, seat_max_torque, stand_max_RPM, stand_max_torque,
                         stand_CdA, seat_CdA, total_mass, gear, seat_height, max_power,
                         stand_TC_slope, seat_TC_slope, stand_fatigue_rate, seat_fatigue_rate):
                self.seat_max_RPM = seat_max_RPM; self.seat_max_torque = seat_max_torque
                self.stand_max_RPM = stand_max_RPM; self.stand_max_torque = stand_max_torque
                self.stand_CdA = stand_CdA; self.seat_CdA = seat_CdA
                self.total_mass = total_mass; self.gear = gear
                self.seat_height = seat_height; self.max_power = max_power
                self.stand_TC_slope = stand_TC_slope; self.seat_TC_slope = seat_TC_slope
                self.stand_fatigue_rate = stand_fatigue_rate; self.seat_fatigue_rate = seat_fatigue_rate

            def initialize_state(self, initial_speed, bank_angle, rad_of_curve, air_density, mu_rr, ks, efficiency, bike_length):
                self.time = 0; self.COM_speed = initial_speed; self.COM_dist = 0
                self.CdA = self.stand_CdA; self.cadence = 0; self.torque = self.stand_max_torque
                self.power_input = self.cadence * self.torque * (math.pi / 30)
                self.power_usable = self.power_input * efficiency
                self.T_max_stand = self.stand_max_torque; self.T_max_seat = self.seat_max_torque
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

        # r order: seat_RPM, seat_torque, seat_CdA, stand_RPM, stand_torque, stand_CdA,
        #          mass, sprocket, chainring, seat_height, stand_fatigue_rate, seat_fatigue_rate
        def _make_athlete(seat_RPM, seat_torque, seat_CdA, stand_RPM, stand_torque, stand_CdA,
                          mass, sprocket, chainring, seat_height, sfr, sefr):
            return Athlete(seat_RPM, seat_torque, stand_RPM, stand_torque, stand_CdA, seat_CdA, mass,
                           27 * chainring / sprocket, seat_height, seat_RPM * seat_torque * math.pi / 120,
                           -stand_torque / stand_RPM, -seat_torque / seat_RPM, sfr, sefr)

        p1 = _make_athlete(*r1t)
        p2 = _make_athlete(*r2t)
        p3 = _make_athlete(*r3t)

        for rider, spd in ((p1, 2), (p2, 2), (p3, 1.6)):
            rider.initialize_state(spd, straight_bank_angle, rad_of_curve, air_density, mu_rr, ks, efficiency, bike_length)

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
            lean = rad_to_deg * math.atan(v_com ** 2 / (9.81 * (r_wh - seat_height * math.sin(deg_to_rad * lean_initial))))
            while lean - lean_initial > 0.1:
                lean_initial = lean
                lean = rad_to_deg * math.atan(v_com ** 2 / (9.81 * (r_wh - seat_height * math.sin(deg_to_rad * lean))))
            r_cm = r_wh - seat_height * math.sin(deg_to_rad * lean) if r_wh < 2 * rad_of_curve else r_wh
            return bank, r_wh, r_cm, lean, bank - lean

        _BASE_FIELDS = ('COM_speed', 'COM_dist', 'bank', 'r_wh', 'r_cm', 'lean', 'camber',
                        'wheel_speed', 'wheel_dist', 'cadence', 'torque', 'power_input',
                        'power_usable', 'prop_force', 'aero_drag', 'weight_force', 'segment',
                        'centripetal_force', 'reaction_force', 'normal_force', 'rr', 'accel')

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
            rider.segment = rider.wheel_dist % 125
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

        # --- P1 simulation ---
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
                        p1.torque = max(0.0, p1.stand_max_torque + p1.stand_TC_slope * p1.cadence)
                    else:
                        p1.T_max_stand -= p1.stand_fatigue_rate * increment
                        p1.torque = max(0.0, p1.T_max_stand + p1.stand_TC_slope * p1.cadence)
                    update_forces(p1, p1.stand_CdA)
                else:
                    p1.T_max_seat -= p1.seat_fatigue_rate * increment
                    p1.torque = max(0.0, p1.T_max_seat + p1.seat_TC_slope * p1.cadence)
                    update_forces(p1, p1.seat_CdA)
                append_state(p1, p1d)
        df_p1 = make_df(p1d)

        # --- P2 simulation ---
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
                        p2.torque = max(0.0, p2.stand_max_torque + p2.stand_TC_slope * p2.cadence)
                    else:
                        p2.T_max_stand -= p2.stand_fatigue_rate * increment
                        p2.torque = max(0.0, p2.T_max_stand + p2.stand_TC_slope * p2.cadence)
                else:
                    p2.T_max_seat -= p2.seat_fatigue_rate * increment
                    p2.torque = max(0.0, p2.T_max_seat + p2.seat_TC_slope * p2.cadence)
                update_forces(p2, cda)
                if p2.gap > 0.0:
                    reduction_pct = max(0.0, -8.1136 * p2.gap + 50.051)
                    p2.aero_drag *= (100 - reduction_pct) / 100
                    p2.accel = (p2.prop_force - (p2.rr + p2.aero_drag)) / p2.total_mass
                count += 1
                append_state(p2, p2d)
        df_p2 = make_df(p2d)

        # --- P3 simulation ---
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
                    p3.rr_demand = p3.rr * df_p2["wheel_speed"][count]
                    p3.aero_demand = 0.5 * air_density * p3.CdA * p3.COM_speed * p3.air_speed ** 2
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
                        p3.torque = max(0.0, p3.stand_max_torque + p3.stand_TC_slope * p3.cadence)
                    else:
                        p3.T_max_stand -= p3.stand_fatigue_rate * increment * p3.dem_sup
                        p3.torque = max(0.0, p3.T_max_stand + p3.stand_TC_slope * p3.cadence)
                else:
                    p3.T_max_seat -= p3.seat_fatigue_rate * increment * p3.dem_sup
                    p3.torque = max(0.0, p3.T_max_seat + p3.seat_TC_slope * p3.cadence)
                update_forces(p3, cda)
                p3.dem_sup = p3.power_demand / p3.power_usable if p3.power_usable > p3.power_demand else 1
                if p3.gap > 0.0:
                    reduction_pct = max(0.0, -8.1136 * p3.gap + 50.051)
                    p3.aero_drag *= (100 - reduction_pct) / 100
                    p3.accel = (p3.prop_force - (p3.rr + p3.aero_drag)) / p3.total_mass
                count += 1
                append_state(p3, p3d, with_demand=True)
        df_p3 = make_df(p3d, extra_cols=('accel_demand', 'rr_demand', 'aero_demand', 'power_demand', 'dem_sup'))
        return df_p1, df_p2, df_p3

df_p1, df_p2, df_p3 = _run_sim(
    tuple(r1), tuple(r2), tuple(r3),
    air_density, dist_at_sit, fatigue_onset,
    straight_bank_angle, bend_bank_angle, pl_to_trans, transition_length
)

if optimize:
    def _gear_range(g):
        return np.round(np.arange(g - opt_span, g + opt_span + opt_inc * 0.5, opt_inc), 4)

    g1_base = 27 * r1[8] / r1[7]
    g2_base = 27 * r2[8] / r2[7]
    g3_base = 27 * r3[8] / r3[7]
    g1s, g2s, g3s = _gear_range(g1_base), _gear_range(g2_base), _gear_range(g3_base)
    total = len(g1s) * len(g2s) * len(g3s)

    if total > _MAX_COMBOS:
        st.error(f"{total} combinations exceeds cap of {_MAX_COMBOS}. Increase Increment or reduce Span.")
    else:
        # Override gear by setting sprocket=27, chainring=gear → _make_athlete computes gear = 27*chainring/sprocket = gear
        def _spec_with_gear(spec, gear):
            return (*spec[:7], 27, float(gear), *spec[9:])

        progress = st.progress(0.0, text=f"Running {total} simulations...")
        results = []
        done = 0
        for g1 in g1s:
            r1_mod = _spec_with_gear(r1, g1)
            for g2 in g2s:
                r2_mod = _spec_with_gear(r2, g2)
                for g3 in g3s:
                    r3_mod = _spec_with_gear(r3, g3)
                    df1, df2, df3 = _run_sim(
                        r1_mod, r2_mod, r3_mod,
                        air_density, dist_at_sit, fatigue_onset,
                        straight_bank_angle, bend_bank_angle, pl_to_trans, transition_length
                    )
                    t1 = float(np.interp(250, df1['wheel_dist'].values, df1['Time'].values))
                    t2 = float(np.interp(500, df2['wheel_dist'].values, df2['Time'].values))
                    t3 = float(np.interp(750, df3['wheel_dist'].values, df3['Time'].values))
                    results.append((float(g1), float(g2), float(g3), t1, t2, t3))
                    done += 1
                    if done % 25 == 0 or done == total:
                        progress.progress(done / total, text=f"{done}/{total}")
        progress.empty()

        df_opt = pd.DataFrame(results, columns=["P1 gear", "P2 gear", "P3 gear",
                                                 "P1 250m (s)", "P2 500m (s)", "P3 750m (s)"])
        df_opt = df_opt.sort_values("P3 750m (s)").head(10).reset_index(drop=True)
        st.session_state["opt_results"] = {
            "df": df_opt,
            "total": int(total),
            "rider_names": list(rider_names),
        }

if "opt_results" in st.session_state:
    _opt = st.session_state["opt_results"]
    df_opt = _opt["df"]
    _opt_names = _opt["rider_names"]

    # Pre-compute lookup of all physical (chainring, sprocket) -> gear
    _PHYS_COMBOS = [(cr, sp, 27 * cr / sp) for cr in range(40, 101) for sp in range(12, 23)]

    def _closest_combo(target):
        cr, sp, g = min(_PHYS_COMBOS, key=lambda x: abs(x[2] - target))
        return cr, sp, g

    with st.expander(f"Optimal gear results — top 10 of {_opt['total']} combos", expanded=True):
        st.dataframe(
            df_opt.style.format({
                "P1 gear": "{:.2f}", "P2 gear": "{:.2f}", "P3 gear": "{:.2f}",
                "P1 250m (s)": "{:.3f}", "P2 500m (s)": "{:.3f}", "P3 750m (s)": "{:.3f}",
            }),
            use_container_width=False,
        )

        st.subheader("Closest possible gear (best result)")
        best = df_opt.iloc[0]
        closest = [_closest_combo(best[f"{pos} gear"]) for pos in ("P1", "P2", "P3")]
        df_closest = pd.DataFrame(
            [
                {
                    "Position": pos,
                    "Rider": _opt_names[i],
                    "Target gear": round(float(best[f"{pos} gear"]), 3),
                    "Chainring": cr,
                    "Sprocket": sp,
                    "Actual gear": round(g, 3),
                    "Delta": round(g - float(best[f"{pos} gear"]), 3),
                }
                for i, (pos, (cr, sp, g)) in enumerate(zip(("P1", "P2", "P3"), closest))
            ]
        ).set_index("Position")
        st.dataframe(df_closest, use_container_width=False)

        if st.button("Use these gears", key="apply_opt_gears"):
            st.session_state["pending_gears"] = [
                (kn, name, int(cr), int(sp))
                for kn, name, (cr, sp, _g) in zip(("1", "2", "3"), _opt_names, closest)
            ]
            del st.session_state["opt_results"]
            st.rerun()

fig_dem_v_supp = px.line(df_p3, x="Time", y=[df_p3["dem_sup"], df_p3["COM_speed"], df_p3["gap"]])
# st.plotly_chart(fig_dem_v_supp, use_container_width=True)

# --- Summary ---
st.header("Summary")
table_inc = st.select_slider("Table increment (m):", options=[5, 25, 62.5, 125], value=25)

def _dists(start, stop, inc):
    return [round(float(v), 4) for v in np.arange(start, stop + inc * 0.5, inc)]

dists_p1      = _dists(table_inc, 250, table_inc)
dists_p2_only = _dists(250 + table_inc, 500, table_inc)
dists_p3_only = _dists(500 + table_inc, 750, table_inc)
dists_p2 = dists_p1 + dists_p2_only
dists_p3 = dists_p2 + dists_p3_only

def qt(d, df_px):
    return round(intp(d, df_px, 'wheel_dist', 'Time')[0], 3)

p1_qt = {d: qt(d, df_p1) for d in dists_p1}
p2_qt = {d: qt(d, df_p2) for d in dists_p2}
p3_qt = {d: qt(d, df_p3) for d in dists_p3}

col_labels = {d: f"{d:g}m" for d in dists_p3}

df_time = pd.DataFrame({"Rider": rider_names}).set_index("Rider")
for d in dists_p1:
    df_time[col_labels[d]] = [p1_qt[d], p2_qt[d], p3_qt[d]]
for d in dists_p2_only:
    df_time[col_labels[d]] = [0, p2_qt[d], p3_qt[d]]
for d in dists_p3_only:
    df_time[col_labels[d]] = [0, 0, p3_qt[d]]
st.subheader("Time")
st.dataframe(df_time, use_container_width=False)

df_gap = pd.DataFrame({"Rider": [rider_names[1], rider_names[2]]}).set_index("Rider")
for d in dists_p1:
    df_gap[col_labels[d]] = [p2_qt[d] - p1_qt[d], p3_qt[d] - p2_qt[d]]
for d in dists_p2_only:
    df_gap[col_labels[d]] = [0, p3_qt[d] - p2_qt[d]]
st.subheader("Gap (s)")
st.dataframe(df_gap, use_container_width=False)

df_dist_gap = pd.DataFrame({"Rider": [rider_names[1], rider_names[2]]}).set_index("Rider")
for d in dists_p1:
    df_dist_gap[col_labels[d]] = [round(d - intp(p1_qt[d], df_p2, 'Time', 'wheel_dist')[0], 2),
                                   round(d - intp(p2_qt[d], df_p3, 'Time', 'wheel_dist')[0], 2)]
for d in dists_p2_only:
    df_dist_gap[col_labels[d]] = [0, round(d - intp(p2_qt[d], df_p3, 'Time', 'wheel_dist')[0], 2)]
st.subheader("Gap (m)")
st.dataframe(df_dist_gap, use_container_width=False)

def make_summary_df(field, label, dp=2):
    df_s = pd.DataFrame({"Rider": rider_names}).set_index("Rider")
    for d in dists_p1:
        df_s[col_labels[d]] = [round(intp(p1_qt[d], df_p1, 'Time', field)[0], dp),
                                round(intp(p2_qt[d], df_p2, 'Time', field)[0], dp),
                                round(intp(p3_qt[d], df_p3, 'Time', field)[0], dp)]
    for d in dists_p2_only:
        df_s[col_labels[d]] = [0,
                                round(intp(p2_qt[d], df_p2, 'Time', field)[0], dp),
                                round(intp(p3_qt[d], df_p3, 'Time', field)[0], dp)]
    for d in dists_p3_only:
        df_s[col_labels[d]] = [0, 0, round(intp(p3_qt[d], df_p3, 'Time', field)[0], dp)]
    return df_s

df_cadence = make_summary_df('cadence', 'Cadence', dp=0)
styled_cadence = df_cadence.replace(0, np.nan).style.background_gradient(axis=1, cmap='RdYlGn').format("{:.0f}", na_rep="")
st.subheader("Cadence")
st.dataframe(styled_cadence, use_container_width=False)

df_wheel_speed = make_summary_df('wheel_speed', 'wheel_speed')
df_wheel_speed = df_wheel_speed.apply(lambda x: round(x * 3.6, 2))
styled_speed = df_wheel_speed.replace(0, np.nan).style.background_gradient(axis=1, cmap='RdYlGn').format("{:.2f}", na_rep="")
st.subheader("Speed (kph)")
st.dataframe(styled_speed, use_container_width=False)

df_power = make_summary_df('power_usable', 'Power', dp=0)
styled_power = df_power.replace(0, np.nan).style.background_gradient(axis=1, cmap='RdYlGn').format("{:.0f}", na_rep="")
st.subheader("Power (W)")
st.dataframe(styled_power, use_container_width=False)

for _df in (df_p1, df_p2, df_p3):
    _df['apparent_cda'] = _df['aero_drag'] / (0.5 * air_density * _df['COM_speed'] ** 2)

df_cda = make_summary_df('apparent_cda', 'Apparent CdA', dp=4)
styled_cda = df_cda.replace(0, np.nan).style.background_gradient(axis=1, cmap='RdYlGn_r').format("{:.4f}", na_rep="")
st.subheader("Apparent CdA")
st.dataframe(styled_cda, use_container_width=False)

df_aero = make_summary_df('aero_drag', 'Aero Drag', dp=1)
styled_aero = df_aero.replace(0, np.nan).style.background_gradient(axis=1, cmap='RdYlGn_r').format("{:.1f}", na_rep="")
st.subheader("Aero Drag (N)")
st.dataframe(styled_aero, use_container_width=False)

# --- CSV export ---
combined_csv = pd.concat([
    df_p1.assign(rider=rider_names[0]),
    df_p2.assign(rider=rider_names[1]),
    df_p3.assign(rider=rider_names[2])
])
st.download_button("Download CSV", combined_csv.to_csv(index=False),
                   "team_sprint_results.csv", "text/csv")

# --- Per-rider tabs ---
from plotly.subplots import make_subplots as _make_subplots

def time_to(dist, df_px):
    row = df_px.iloc[(df_px['wheel_dist'] - dist).abs().argsort()[:2]].reset_index(drop=True)
    return row["Time"][1] + (dist - row["wheel_dist"][1]) / row["wheel_speed"][1]

# Combined 3-row power+speed subplot (shown above tabs)
fig_all = _make_subplots(rows=3, cols=1, shared_xaxes=True,
                         specs=[[{"secondary_y": True}]] * 3,
                         subplot_titles=rider_names)
for i, (df_px, label) in enumerate([(df_p1, rider_names[0]), (df_p2, rider_names[1]), (df_p3, rider_names[2])], 1):
    fig_all.add_trace(go.Scatter(x=df_px["Time"], y=df_px["power_usable"],
                                 name=f"{label} Power", line=dict(color='royalblue')), row=i, col=1)
    fig_all.add_trace(go.Scatter(x=df_px["Time"], y=df_px["wheel_speed"] * 3.6,
                                 name=f"{label} Speed (km/h)", line=dict(color='crimson')),
                      row=i, col=1, secondary_y=True)
fig_all.update_yaxes(title_text="Power (W)", secondary_y=False)
fig_all.update_yaxes(title_text="Speed (km/h)", secondary_y=True)
fig_all.update_layout(title_text="Power & Speed — all riders", height=700)
st.plotly_chart(fig_all, use_container_width=True)

# Fatigue (T_max) chart — reconstruct T_max using per-rider TC slope
fig_fatigue = go.Figure()
for df_px, label, r in [(df_p1, rider_names[0], r1), (df_p2, rider_names[1], r2), (df_p3, rider_names[2], r3)]:
    # r = [seat_RPM, seat_torque, seat_CdA, stand_RPM, stand_torque, ...]
    stand_slope = -float(r[4]) / float(r[3])   # -stand_max_torque / stand_max_RPM
    t_max_series = df_px["torque"] - df_px["cadence"] * stand_slope
    fig_fatigue.add_trace(go.Scatter(x=df_px["Time"], y=t_max_series, mode='lines', name=f"{label} T_max"))
fig_fatigue.update_layout(title_text="T_max over time (reconstructed)", xaxis_title="Time (s)", yaxis_title="T_max (Nm)")
st.plotly_chart(fig_fatigue, use_container_width=True)

# Per-rider tabs
tab1, tab2, tab3, tab4 = st.tabs([rider_names[0], rider_names[1], rider_names[2], "Animation"])

with tab1:
    p1_250_time = df_p1["Time"].iloc[-2] + ((250 - df_p1["wheel_dist"].iloc[-2]) / df_p1["wheel_speed"].iloc[-2])
    st.write(f"Time to 250m: **{round(p1_250_time, 3)} s**")

with tab2:
    st.write(f"Time to 250m: **{round(time_to(250, df_p2), 3)} s**")
    st.write(f"Time to 500m: **{round(time_to(500, df_p2), 3)} s**")

with tab3:
    st.write(f"Time to 250m: **{round(time_to(250, df_p3), 3)} s**")
    st.write(f"Time to 500m: **{round(time_to(500, df_p3), 3)} s**")
    st.write(f"Time to 750m: **{round(time_to(750, df_p3), 3)} s**")

with tab4:
    t_max_anim = float(max(df_p1["Time"].iloc[-1], df_p2["Time"].iloc[-1], df_p3["Time"].iloc[-1]))
    anim_times = np.arange(0.0, t_max_anim + 0.2, 0.2)

    def get_dist_at(df_px, t):
        t_arr = df_px["Time"].values
        wd_arr = df_px["wheel_dist"].values
        return float(np.interp(min(float(t), float(t_arr[-1])), t_arr, wd_arr))

    colors_anim = ['crimson', 'royalblue', 'seagreen']
    y_riders = [0.18, 0.0, -0.18]
    max_dist_anim = 760

    marker_xs, marker_ys = [], []
    label_xs, label_ys, label_texts = [], [], []
    for d in range(0, max_dist_anim + 1, 10):
        marker_xs += [d, d, None]
        marker_ys += [-0.55, 0.55, None]
        label_xs.append(d)
        label_ys.append(-0.82)
        label_texts.append(f"{d}m")

    static_traces = [
        go.Scatter(x=[0, max_dist_anim], y=[0, 0],
                   mode='lines', line=dict(color='#555', width=3),
                   showlegend=False, hoverinfo='skip'),
        go.Scatter(x=marker_xs, y=marker_ys,
                   mode='lines', line=dict(color='rgba(180,180,180,0.5)', width=1),
                   showlegend=False, hoverinfo='skip'),
        go.Scatter(x=label_xs, y=label_ys, mode='text',
                   text=label_texts, textfont=dict(size=9, color='#aaaaaa'),
                   showlegend=False, hoverinfo='skip'),
    ]

    init_rider_traces = []
    for df_px, name, color, y in zip([df_p1, df_p2, df_p3], rider_names, colors_anim, y_riders):
        d0 = get_dist_at(df_px, 0.0)
        init_rider_traces.append(go.Scatter(
            x=[d0], y=[y], mode='markers+text',
            marker=dict(size=20, color=color),
            text=[name], textposition='top center',
            textfont=dict(size=12, color='white'),
            name=name
        ))

    anim_frames = []
    for i, t in enumerate(anim_times):
        d1 = get_dist_at(df_p1, t)
        d2 = get_dist_at(df_p2, t)
        d3 = get_dist_at(df_p3, t)
        frame_data = []
        for dist, name, color, y in zip([d1, d2, d3], rider_names, colors_anim, y_riders):
            frame_data.append(go.Scatter(
                x=[dist], y=[y], mode='markers+text',
                marker=dict(size=20, color=color),
                text=[name], textposition='top center',
                textfont=dict(size=12, color='white'),
                name=name, showlegend=False
            ))
        anim_frames.append(go.Frame(
            data=frame_data,
            traces=[3, 4, 5],
            name=f"f{i}"
        ))

    slider_steps_anim = [
        dict(method='animate',
             args=[[f"f{i}"], dict(mode='immediate', frame=dict(duration=0, redraw=False))],
             label=f"{t:.0f}s" if i % 5 == 0 else "")
        for i, t in enumerate(anim_times)
    ]

    fig_anim = go.Figure(data=static_traces + init_rider_traces, frames=anim_frames)
    fig_anim.update_layout(
        title="Team Sprint — Side View",
        height=380,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='white',
        xaxis=dict(range=[0, max_dist_anim], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1.3, 1.3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        margin=dict(l=20, r=20, t=60, b=120),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.08),
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=-0.18, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=200, redraw=False),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')])
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Time: ', suffix=' s', visible=True,
                              font=dict(size=14)),
            pad=dict(t=10, b=10),
            x=0.0, y=-0.05, len=1.0,
            steps=slider_steps_anim
        )]
    )
    st.plotly_chart(fig_anim, use_container_width=True)
