#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from datetime import date, timedelta

import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import streamlit as st
import streamlit.components.v1 as components
import st_debug as d
import altair as alt
import pystarboard.data as filecoin_data

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u
from PIL import Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("debug.css")

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr

@st.cache_data
def get_max_historical_data(current_date):
    start_date = date(2021, 3, 15)
    _, max_historical_rbp = u.get_historical_daily_onboarded_power(start_date, current_date)
    max_historical_rbp = float(np.nanmax(max_historical_rbp))
    _, max_historical_rr = u.get_historical_renewal_rate(start_date, current_date)
    max_historical_rr = float(np.nanmax(max_historical_rr))
    return max_historical_rbp, max_historical_rr

@st.cache_data
def rbp_match_minting(start_date, end_date):
    start_day = du.datetime64_delta_to_days(np.datetime64(start_date) - C.NETWORK_START)
    end_day = du.datetime64_delta_to_days(np.datetime64(end_date) - C.NETWORK_START)

    # run a hypothetical scenario where RBP matched baseline
    init_baseline_eib = filecoin_data.get_storage_baseline_value(start_date) / C.EXBI
    zero_cum_capped_power_eib = filecoin_data.get_cum_capped_rb_power(start_date) / C.EXBI
    baseline_power_arr = minting.compute_baseline_power_array(
        np.datetime64(start_date),
        np.datetime64(end_date),
        init_baseline_eib
    )
    cum_capped_power_EIB = baseline_power_arr.cumsum() + zero_cum_capped_power_eib
    network_time = minting.network_time(cum_capped_power_EIB)
    cum_baseline_reward = minting.cum_baseline_reward(network_time)
    days_vec = jnp.arange(start_day, end_day)
    cum_simple_reward = minting.cum_simple_minting(days_vec)
    cum_network_reward = cum_baseline_reward + cum_simple_reward
    cum_network_reward_zero = cum_network_reward[0]

    day_network_reward = jnp.zeros(len(cum_network_reward)+1)
    day_network_reward = day_network_reward.at[1:].set(cum_network_reward)
    day_network_reward = day_network_reward.at[0].set(cum_network_reward_zero) # equiv. to prepend in NP
    day_network_reward = jnp.diff(day_network_reward)
    day_network_reward = day_network_reward.at[0].set(day_network_reward[1]) # to match mechaFIL
    baseline_dates = du.get_t(start_date, end_date=end_date)
    return cum_network_reward, baseline_dates


def plot_panel(scenario_results, baseline, start_date, current_date, end_date):
    # convert results dictionary into a dataframe so that we can use altair to make nice plots
    status_quo_results = scenario_results['status-quo']
    max_hist_results = scenario_results['max-hist']
    # print(status_quo_results.keys())
    # print(status_quo_results['circ_supply'].shape)

    power_dff = pd.DataFrame()
    power_dff['Configured'] = status_quo_results['rb_total_power_eib']
    power_dff['Max-Historical'] = max_hist_results['rb_total_power_eib']
    power_dff['QAP'] = status_quo_results['qa_total_power_eib']
    power_dff['Baseline'] = baseline
    power_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    minting_dff = pd.DataFrame()
    ixx = (current_date - start_date).days
    simple_before_curdate = status_quo_results['cum_network_reward'][:ixx]/1e6
    simple_after_curdate = status_quo_results['cum_simple_reward'][ixx:]/1e6  + (simple_before_curdate[-1]-status_quo_results['cum_simple_reward'][ixx]/1e6)
    simple_minting = np.concatenate([simple_before_curdate, simple_after_curdate])
    minting_dff['Configured'] = status_quo_results['cum_network_reward']/1e6
    minting_dff['Max-Historical'] = max_hist_results['cum_network_reward']/1e6
    minting_dff['Simple Minting Only'] = simple_minting
    minting_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))
    # get hypothetical minting if RBP matched baseline
    cum_network_reward, baseline_dates = rbp_match_minting(start_date, end_date)
    ## circ-supply DF
    dates_of_interest = [current_date]
    doi = current_date + timedelta(days=365*5)
    while doi < end_date:
        dates_of_interest.append(doi)
        doi += timedelta(days=365*5)
    final_date = end_date - timedelta(days=1)
    if final_date not in dates_of_interest:
        dates_of_interest.append(final_date)
    date2ts = {}
    for d in dates_of_interest:
        ix = (d - start_date).days
        # ix_cur = (current_date - start_date).days
        ts_configured_ix = status_quo_results['cum_network_reward'][ix]/1e6 - status_quo_results['network_gas_burn'][ix]/1e6  + status_quo_results['total_vest'][ix]/1e6
        ts_simple_ix = minting_dff['Simple Minting Only'][ix] - status_quo_results['network_gas_burn'][ix]/1e6  + status_quo_results['total_vest'][ix]/1e6
        ts_maxhist_ix = max_hist_results['cum_network_reward'][ix]/1e6 - max_hist_results['network_gas_burn'][ix]/1e6  + max_hist_results['total_vest'][ix]/1e6

        date2ts[d] = {
            'Simple Minting Only': float(ts_simple_ix),
            'Configured': float(ts_configured_ix),
            'Max-Historical': float(ts_maxhist_ix)
        }
    ts_df = pd.DataFrame(date2ts).T
    ts_df['date'] = pd.to_datetime(ts_df.index)
    ts_df = ts_df.reset_index(drop=True)
    ts_dff = ts_df.copy()
    ts_dff.set_index('date', inplace=True)

    hypothetical_minting_df = pd.DataFrame()
    ixx = (current_date - start_date).days
    rbp_baseline_before_curdate = status_quo_results['cum_network_reward'][:ixx]/1e6
    rbp_baseline_after_curdate = cum_network_reward[ixx:]/1e6 + (rbp_baseline_before_curdate[-1]-cum_network_reward[ixx]/1e6)
    rbp_baseline_minting = np.concatenate([rbp_baseline_before_curdate, rbp_baseline_after_curdate])
    hypothetical_minting_df['RBP=Baseline'] = rbp_baseline_minting
    # hypothetical_minting_df['RBP=Baseline'] = cum_network_reward/1e6
    hypothetical_minting_df['date'] = pd.to_datetime(baseline_dates)
    minting_dff = minting_dff.merge(hypothetical_minting_df, on='date', how='left')
    current_date_vline = alt.Chart(
        pd.DataFrame({'date': [current_date]})).mark_rule(strokeDash=[3,5], color='black').encode(x='date:T')

    power_df = pd.melt(power_dff, id_vars=["date"], 
                        value_vars=["Baseline", "Configured", "Max-Historical"],
                        var_name='Power', 
                        value_name='EiB')
    power_df['EiB'] = power_df['EiB']
    power = (
        alt.Chart(power_df)
        .mark_line()
        .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                y=alt.Y("EiB").scale(type='log'), 
                color=alt.Color('Power', 
                                sort=['Baseline', "Max-Historical", 'Configured'],
                                legend=alt.Legend(title="Onboarding Configuration")),
                strokeDash=alt.StrokeDash('Power', 
                                        sort=['Baseline', "Max-Historical", 'Configured'],
                                        legend=None,
                                        scale=alt.Scale(domain=['Baseline', "Max-Historical", 'Configured'],
                                                        range=[[0], [0], [2, 2]])))
        .properties(title="Network Power (RBP)", width=800, height=200)
        # .configure_title(fontSize=20, anchor='middle')
    )
    power += current_date_vline
    # st.altair_chart(power.interactive(), use_container_width=True) 

    minting_df = pd.melt(minting_dff, id_vars=["date"],
                            value_vars=["RBP=Baseline", "Max-Historical", "Configured", "Simple Minting Only"],
                            var_name='Scenario', value_name='Mined FIL')
    minting = (
        alt.Chart(minting_df)
        .mark_line()
        .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                y=alt.Y("Mined FIL", title='M-FIL'), 
                color=alt.Color('Scenario', 
                            sort=['RBP=Baseline', "Max-Historical", 'Configured', 'Simple Minting Only'],
                            legend=alt.Legend(title="Onboarding Configuration")),
                strokeDash=alt.StrokeDash('Scenario', 
                                        sort=['RBP=Baseline', "Max-Historical", "Configured", 'Simple Minting Only'],
                                        legend=None,
                                        scale=alt.Scale(domain=['RBP=Baseline', "Max-Historical", "Configured", 'Simple Minting Only'],
                                                        range=[[0], [0], [2, 2], [0]])))
        .properties(title="Mined FIL", width=800, height=200)
        # .configure_title(fontSize=20, anchor='middle')
    )
    
    minting += current_date_vline
    # st.altair_chart(minting.interactive(), use_container_width=True)
    autosize = alt.AutoSizeParams(contains="content", resize=True, type='fit-x')
    final_chart = alt.vconcat(minting, power, autosize=autosize).resolve_scale(x='shared', color='independent')
    st.altair_chart(final_chart, use_container_width=True)
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.altair_chart(minting.interactive(), use_container_width=True)
    # with col2:
    #     st.altair_chart(power.interactive(), use_container_width=True)
    st.markdown("##### Total Supply")
    ts_dff = ts_dff[['Simple Minting Only', 'Configured', 'Max-Historical']]
    ts_dff.rename(columns={'Simple Minting Only': 'Simple Minting Only (B-FIL)', 'Configured': 'Configured (B-FIL)', 'Max-Historical': 'Max-Historical (B-FIL)'}, inplace=True)
    ts_dff['Simple Minting Only (B-FIL)'] = ts_dff['Simple Minting Only (B-FIL)'].apply(lambda x: f'{x/1000.:.2f}')
    ts_dff['Configured (B-FIL)'] = ts_dff['Configured (B-FIL)'].apply(lambda x: f'{x/1000.:.2f}')
    ts_dff['Max-Historical (B-FIL)'] = ts_dff['Max-Historical (B-FIL)'].apply(lambda x: f'{x/1000.:.2f}')
    ts_dff.index = pd.to_datetime(ts_dff.index).date
    st.dataframe(ts_dff)  # convert to percentage


def run_sim(rbp, rr, fpr, lock_target, start_date, current_date, forecast_length_days, sector_duration_days, offline_data):
    simulation_results = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data
    )
    
    return simulation_results 

def forecast_economy(start_date=None, current_date=None, end_date=None, forecast_length_days=365*10, fpr_pct=50):
    t1 = time.time()
    
    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = fpr_pct

    lock_target = 0.3
    sector_duration_days = 360
    
    # get offline data
    t2 = time.time()
    offline_data, _, _, _ = get_offline_data(start_date, current_date, end_date)
    t3 = time.time()

    max_hist_rbp, max_hist_rr = get_max_historical_data(current_date)
    # run simulation for the configured scenario, and for a pessimsitc and optimistic version of it
    scenario_strings = ['status-quo', 'max-hist']
    scenario_results = {}
    for ii, scenario_str in enumerate(scenario_strings):
        if scenario_str == 'status-quo':
            rbp_val = rb_onboard_power_pib_day
            rr_val = max(0.0, min(1.0, renewal_rate_pct / 100.))
            fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100.))
        else:
            rbp_val = max_hist_rbp
            rr_val = max_hist_rr
            fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100.))

        rbp = jnp.ones(forecast_length_days) * rbp_val
        rr = jnp.ones(forecast_length_days) * rr_val
        fpr = jnp.ones(forecast_length_days) * fpr_val
        
        simulation_results = run_sim(rbp, rr, fpr, lock_target, start_date, current_date, forecast_length_days, sector_duration_days, offline_data) 
        scenario_results[scenario_strings[ii]] = simulation_results

    baseline = minting.compute_baseline_power_array(
        np.datetime64(start_date), np.datetime64(end_date), offline_data['init_baseline_eib'],
    )

    # plot
    plot_panel(scenario_results, baseline, start_date, current_date, end_date)
    t4 = time.time()
    # d.debug(f"Time to forecast: {t4-t3}")
    # d.debug(f"Total Time: {t4-t1}")

def main():
    im = Image.open("./fil_minting_explorer/static/fil_logo.png")
    st.set_page_config(
        page_title="Filecoin Economics Explorer",
        page_icon=im,
        layout="wide",
    )
    current_date = date.today() - timedelta(days=3)
    # mo_start = max(current_date.month - 1 % 12, 1)
    # start_date = date(current_date.year, mo_start, 1)
    start_date = date(2021, 3, 15)
    # forecast_length_days=365*16
    # end_date = current_date + timedelta(days=forecast_length_days)
    end_date = date(2040, 1, 1)
    forecast_length_days = (end_date - current_date).days
    forecast_kwargs = {
        'start_date': start_date,
        'current_date': current_date,
        'end_date': end_date,
        'forecast_length_days': forecast_length_days,
    }

    _, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr = get_offline_data(start_date, current_date, end_date)
    smoothed_last_historical_renewal_pct = int(smoothed_last_historical_rr * 100)
    smoothed_last_historical_fil_plus_pct = int(smoothed_last_historical_fpr * 100)
    forecast_kwargs['fpr_pct'] = smoothed_last_historical_fil_plus_pct
    
    with st.sidebar:
        st.title('Filecoin Economics Explorer')

        st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=100., value=smoothed_last_historical_rbp, step=.1, format='%0.02f', key="rbp_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        st.slider("SP Renewal Rate (%)", min_value=10, max_value=99, value=smoothed_last_historical_renewal_pct, step=1, format='%d', key="rr_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        # st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_fil_plus_pct, step=1, format='%d', key="fpr_slider",
        #         on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
    
        st.button("Forecast", on_click=forecast_economy, kwargs=forecast_kwargs, key="forecast_button")

    
    if "debug_string" in st.session_state:
        st.markdown(
            f'<div class="debug">{ st.session_state["debug_string"]}</div>',
            unsafe_allow_html=True,
        )
    components.html(
        d.js_code(),
        height=0,
        width=0,
    )

if __name__ == '__main__':
    main()