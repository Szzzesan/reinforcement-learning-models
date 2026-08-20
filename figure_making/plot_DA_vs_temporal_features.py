import glob
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import t
from scipy import stats
import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation

from src.data_loader import load_and_concat_population_data
import src.config as config


def run_td_error_lmem_analysis(td_df):
    """
    Preprocesses trajectory replay data and runs a Linear Mixed Effects Model (LMEM)
    on the simulated TD errors, mirroring the in vivo dopamine analysis.

    Returns the fitted results and the processed dataframe used for the model.
    """
    # --- Step 1: Data Preparation ---
    # Filter (mirroring IRI > 1 from the neural data) and copy to avoid warnings
    # df = td_df[(td_df['event_timer'] > 1)].copy()
    df = td_df[(td_df['event_timer'] > 0)].copy()
    # df = td_df.copy()

    # Log transform the temporal features (NRI -> time_in_port, IRI -> event_timer)
    df['log_time_in_port'] = np.log(df['time_in_port'])
    df['log_event_timer'] = np.log(df['event_timer'])

    # Standardize regressors for comparable coefficient scales
    for col in ['log_time_in_port', 'log_event_timer']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_std'] = (df[col] - mean) / std

    # Ensure context is treated as an integer for the formula
    df['context'] = df['context'].astype(int)

    # --- Step 2: Define and Fit the Model ---
    # We model the interaction of time metrics and contextual block.
    # Anatomical predictors (hemisphere/side) are removed.
    model_formula = (
        "td_error ~ (log_time_in_port_std + log_event_timer_std "
        "+ C(context, Treatment(1)))**2"
    )

    random_slopes = (
        "1 + "  # Random Intercept
        "log_time_in_port_std + "  # Random Slope for Time in Port
        "log_event_timer_std + "  # Random Slope for Event Timer
        "C(context, Treatment(1))"  # Random Slope for Context
    )

    print("Fitting TD-error LMEM... this may take a moment.")

    # Define the model. We group by 'animal' and use 'session' as a variance component.
    # The 'site' variance component from the DA model is dropped.
    model = smf.mixedlm(
        model_formula,
        data=df,
        groups=df["animal"],
        re_formula=random_slopes,
        vc_formula={
            "session": "0 + C(session)"
        }
    )

    # Fit the model using Powell's method (matches your original DA script)
    model_results = model.fit(method="powell", reml=False)

    print("Model converged.")
    print(model_results.summary())

    return model_results, df


def _prepare_binned_data(master_df, group_by_cols):
    data = master_df[
        (master_df['event_timer'] > 1) |
        (master_df['event_timer'] == master_df['time_in_port'])
        ].copy()
    bins = [0, 0.8, 1.8, 2.9, 4.1, 5.5, 7.3, 9.6, np.inf]
    # bins = [0, 0.6, 1.3, 2.1, 2.9, 3.9, 5.0, 6.2, 7.6, 9.4, np.inf]
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    bin_labels[-1] = f'>{bins[-2]}'
    data['cat_code'] = pd.cut(data['time_in_port'], bins=bins, labels=bin_labels)
    summary_data = data.groupby(group_by_cols, observed=True).agg(
        time_in_port=('time_in_port', 'median'),
        td_error=('td_error', 'mean')
    ).reset_index()
    return summary_data


def _prepare_collapsed_block_split_data(master_df):
    data = master_df[master_df['event_timer'] > 1].copy()
    summary_data = data.groupby(['animal', 'session', 'context'], observed=True).agg(
        td_error=('td_error', 'mean')).reset_index()
    return summary_data


def _paired_t_for_NRI_bins(summary_data):
    cat_code = summary_data['cat_code'].unique()
    grouped = summary_data.groupby(['animal', 'session'])
    compared = []
    paired_t_stats = []
    paired_p_values = []
    for i in range(len(cat_code) - 1):
        cat1 = cat_code[i]
        cat2 = cat_code[i + 1]
        DA_early = []
        DA_late = []
        for (animal, session), group in grouped:
            DA1 = group.loc[group['cat_code'] == cat1, 'td_error'].values
            DA2 = group.loc[group['cat_code'] == cat2, 'td_error'].values
            if (len(DA1) == 1) & (len(DA2) == 1):
                DA_early.append(DA1[0])
                DA_late.append(DA2[0])
        t_stat, p_value = stats.ttest_rel(DA_late, DA_early, alternative='greater')
        compared.append((cat1, cat2))
        paired_t_stats.append(t_stat)
        paired_p_values.append(p_value)
        print(f't_stat = {t_stat:.3f}, p_value = {p_value:.4f}')
    return compared, paired_t_stats, paired_p_values


def _paired_t_for_blocks(summary_data):
    grouped = summary_data.groupby(['session'])
    cat_code = summary_data['cat_code'].unique().tolist()
    compared = []
    paired_t_stats = []
    paired_p_values = []

    for cat in cat_code:
        DA_low = []
        DA_high = []
        for session, group in grouped:
            DA1 = group.loc[(group['context'] == 0) & (group['cat_code'] == cat), 'td_error'].values
            DA2 = group.loc[(group['context'] == 1) & (group['cat_code'] == cat), 'td_error'].values
            if (len(DA1) == 1) & (len(DA2) == 1):
                DA_low.append(DA1[0])
                DA_high.append(DA2[0])
        t_stat, p_value = stats.ttest_rel(DA_low, DA_high)
        compared.append(cat)
        paired_t_stats.append(t_stat)
        paired_p_values.append(p_value)
        print(f't_stat = {t_stat:.3f}, p_value = {p_value:.4f}')
    return compared, paired_t_stats, paired_p_values


def _paired_t_for_collapsed_blocks(grouped_data):
    pivot_df = grouped_data.pivot_table(index=['session'], columns='context', values='td_error')
    DA_low = pivot_df[0.0]  # todo: not sure if this would work we will see
    DA_high = pivot_df[1.0]
    t_stat, p_value = stats.ttest_rel(DA_low, DA_high)
    return t_stat, p_value


def _set_axes_for_box_and_swarm(axes):
    axes.set_xticklabels(axes.get_xticklabels(), ha='center')
    # axes.set_title('DA vs. NRI', pad=-5)
    axes.set_xlabel('Reward Time from Entry (s)', labelpad=2)
    axes.set_ylabel('TD Error')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)


def fige_DA_vs_NRI_v2(master_df, dodge=True, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    session_summary_data = _prepare_binned_data(master_df, ['animal', 'session', 'cat_code'])

    sns.boxplot(data=session_summary_data, x='cat_code', y='td_error', linewidth=1, showfliers=False,
                notch=True, width=0.9,
                boxprops=dict(facecolor='lightgrey', alpha=0.4),
                medianprops={'linewidth': 2, 'color': 'black'},
                ax=axes)
    # add the stats annotation over the boxplots
    group_compared, t_stats, p_values = _paired_t_for_NRI_bins(session_summary_data)
    # y_bars = [4.2, 4.6, 4.8, 5.0, 5.5, 5.5, 6.2]
    y_bar_start = 1.06
    y_bar_current = y_bar_start
    for center in range(len(p_values)):
        if p_values[center] < 0.0001:
            annot = "****"
            y_stagger = 0.008
        elif p_values[center] < 0.001:
            annot = "***"
            y_stagger = 0.006
        elif p_values[center] < 0.01:
            annot = "**"
            y_stagger = 0.004
        elif p_values[center] < 0.05:
            annot = "*"
            y_stagger = 0.002
        else:
            annot = 'ns'
            y_stagger = 0
        x_center1, x_center2 = center, center + 1
        inset = 0.05
        bracket_x1 = x_center1 + inset
        bracket_x2 = x_center2 - inset
        y_bar_current = y_bar_start
        y, h = y_bar_current, 0.001
        col = 'k'
        axes.plot([bracket_x1, bracket_x1, bracket_x2, bracket_x2],
                  [y, y + h, y + h, y], lw=1, c=col)
        axes.text((bracket_x1 + bracket_x2) * 0.5, y + 0.001, annot,
                  ha='center', va='bottom', color=col, fontsize=10)

    for patch in axes.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.6))

    if dodge:  # then use the default color palette for categorical variable
        animal_order = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK007', 'RK008']
        sns.swarmplot(data=session_summary_data, x='cat_code', y='td_error',
                      hue='animal', hue_order=animal_order, size=2,
                      dodge=dodge, legend=False, ax=axes,
                      linewidth=0.5,
                      edgecolor='face')
    if not dodge:  # use the diverging color palette of our preference
        animal_order = session_summary_data.groupby('animal')['td_error'].mean().sort_values(ascending=False).index
        custom_palette = sns.color_palette("RdYlBu", n_colors=len(animal_order))
        sns.swarmplot(data=session_summary_data, x='cat_code', y='td_error',
                      hue='animal', size=2,
                      dodge=dodge, hue_order=animal_order, palette='cmr.guppy',
                      legend=False, ax=axes,
                      linewidth=0.75,
                      edgecolor='face')
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)
    _set_axes_for_box_and_swarm(axes)
    # axes.set_ylim(0.879, 1.001)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def figf_DA_vs_NRI_block_split_v2(master_df, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    session_summary_data = _prepare_binned_data(master_df, ['session', 'cat_code', 'context'])
    block_palette = sns.color_palette('Set2', 2)
    hue_order = [0, 1]
    custom_palette = {0: block_palette[0], 1: block_palette[1]}
    sns.boxplot(data=session_summary_data, x='cat_code', y='td_error', hue='context', notch=True, gap=0.1,
                hue_order=hue_order, palette=custom_palette,
                boxprops=dict(alpha=0.4),
                medianprops={'linewidth': 1, 'color': 'black'},
                legend=False,
                showfliers=False, ax=axes)
    # add stats annotation
    group_compared, t_stats, p_values = _paired_t_for_blocks(session_summary_data)
    y_bars = [4.2, 4.8, 4.8, 5.0, 5.3, 5.5, 5.8, 6.25]
    y_bars = [1.05] * 8
    for center in range(len(p_values)):
        if p_values[center] < 0.0001:
            annot = "****"
        elif p_values[center] < 0.001:
            annot = "***"
        elif p_values[center] < 0.01:
            annot = "**"
        elif p_values[center] < 0.05:
            annot = "*"
        else:
            annot = 'ns'
        x_center1, x_center2 = center - 0.2, center + 0.2
        inset = 0.05
        bracket_x1 = x_center1 + inset
        bracket_x2 = x_center2 - inset
        y, h = y_bars[center], 0.001
        col = 'k'
        axes.plot([bracket_x1, bracket_x1, bracket_x2, bracket_x2],
                  [y, y + h, y + h, y], lw=1, c=col)
        axes.text((bracket_x1 + bracket_x2) * 0.5, y + 0.001, annot,
                  ha='center', va='bottom', color=col, fontsize=10)

    std = session_summary_data.groupby(['cat_code', 'context'])['td_error'].transform('std')
    mean = session_summary_data.groupby(['cat_code'])['td_error'].transform('mean')
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    swarm_data = session_summary_data[
        (session_summary_data['td_error'] > lower_bound) & (session_summary_data['td_error'] < upper_bound)]
    swarm_data = session_summary_data
    # custom_palette = {'0.4': sns.color_palette('colorblind')[0], '0.8': sns.color_palette('colorblind')[3]}
    sns.swarmplot(data=swarm_data, x='cat_code', y='td_error', hue='context',
                  size=2, palette=custom_palette,
                  dodge=True, legend=False, ax=axes,
                  linewidth=0.5,
                  edgecolor='face')
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)
    _set_axes_for_box_and_swarm(axes)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def figf_summary_block_split(master_df, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(2, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    master_df = master_df[master_df['event_timer'] > 1].copy()
    block_palette = sns.color_palette('Set2', 2)
    hue_order = [0, 1]
    # custom_palette = {'0': block_palette[0], '1': block_palette[1]}
    custom_palette = {'0.0': block_palette[0], '1.0': block_palette[1]}

    grouped_data = _prepare_collapsed_block_split_data(master_df)

    sns.boxplot(data=grouped_data,
                x='context',
                y='td_error',
                order=['0.0', '1.0'],  # Use the exact float-strings here too
                palette=custom_palette,
                notch=True,
                boxprops=dict(alpha=0.4),
                width=0.4,
                medianprops={'linewidth': 1, 'color': 'black'},
                legend=False,
                showfliers=False,
                ax=axes)

    # Add stats annotation
    t_stat, p_value = _paired_t_for_collapsed_blocks(grouped_data)
    if p_value < 0.0001:
        annot = "****"
    elif p_value < 0.001:
        annot = "***"
    elif p_value < 0.01:
        annot = "**"
    elif p_value < 0.05:
        annot = "*"
    else:
        annot = 'ns'
    y_bar = 1.05
    center_x1, center_x2 = 0, 1
    inset = 0.05
    bracket_x1 = center_x1 + inset
    bracket_x2 = center_x2 - inset
    y, h = y_bar, 0.001
    col = 'k'
    axes.plot([bracket_x1, bracket_x1, bracket_x2, bracket_x2],
              [y, y + h, y + h, y], lw=1, c=col)
    axes.text((bracket_x1 + bracket_x2) * 0.5, y + 0.001, annot,
              ha='center', va='bottom', color=col, fontsize=10)

    sns.swarmplot(data=grouped_data,
                  x='context',
                  y='td_error',
                  order=['0.0', '1.0'],
                  palette=custom_palette,
                  size=2,
                  legend=False,
                  linewidth=0.5,
                  edgecolor='face',
                  ax=axes)
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)

    axes.set_xticks([0, 1], ['Low', 'High'], fontsize='small')
    axes.set_xlabel('Context\nReward Rate')
    axes.set_ylabel("")
    # axes.set_yticklabels([])
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def fige_DA_vs_IRI_binned(master_df, axes=None):
    """
    Population box plot of DA peak amplitudes binned by IRI.
    Modified from fig_dopamine.fige_DA_vs_NRI_v2.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False
    data = master_df[(master_df['event_timer'] > 1)].copy()
    # Define bins specifically for IRI
    bins = [1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.7, 3.6, np.inf]
    bin_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    bin_labels[-1] = f'>{bins[-2]}'
    data['cat_code'] = pd.cut(data['event_timer'], bins=bins, labels=bin_labels)

    summary_data = data.groupby(['animal', 'session', 'cat_code'], observed=True).agg(
        td_error=('td_error', 'mean')).reset_index()

    sns.boxplot(data=summary_data, x='cat_code', y='td_error', linewidth=1, showfliers=False,
                notch=True, width=0.9, boxprops=dict(facecolor='lightgrey', alpha=0.4),
                medianprops={'linewidth': 2, 'color': 'black'}, ax=axes)

    animal_order = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043', 'RK007', 'RK008']
    sns.swarmplot(data=summary_data, x='cat_code', y='td_error', hue='animal', hue_order=animal_order, size=2,
                  dodge=True, legend=False, ax=axes, linewidth=0.5, edgecolor='face')
    fill_alpha = 0.5
    for collection in axes.collections:
        face_colors = collection.get_facecolors()
        face_colors[:, 3] = fill_alpha
        collection.set_facecolors(face_colors)

    pivot_df = summary_data.pivot_table(index=['animal', 'session'],
                                        columns='cat_code', values='td_error')

    y_max = summary_data['td_error'].max()
    h = 0.001  # Height of the bracket
    y_bars = [1.05] * 7
    y_bar_start = 4.2
    y_bar_current = y_bar_start
    inset = 0.05

    # Compare adjacent bins
    for i in range(len(bin_labels) - 1):
        bin1, bin2 = bin_labels[i], bin_labels[i + 1]

        # Get paired values
        pair_data = pivot_df[[bin1, bin2]].dropna()
        if len(pair_data) > 3:  # Ensure enough samples for t-test
            t_stat, p_val = stats.ttest_rel(pair_data[bin2], pair_data[bin1], alternative='greater')

            # Significance markers
            if p_val < 0.001:
                sig = '***'
                y_stagger = 0.3
            elif p_val < 0.01:
                sig = '**'
                y_stagger = 0.2
            elif p_val < 0.05:
                sig = '*'
                y_stagger = 0.1
            else:
                sig = 'n.s.'
                y_stagger = 0

            # Draw brackets and text
            x1, x2 = i + inset, i + 1 - inset
            y_bar_current = y_bar_current + y_stagger
            y = 1.06
            axes.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c='black')
            axes.text((x1 + x2) * 0.5, y + 0.001, sig, ha='center', va='bottom', color='black', fontsize=10)

    # axes.set_yticks([1, 2, 3, 4, 5, 6])
    axes.set_xlabel('Inter-Reward Interval (s)', labelpad=2)
    axes.set_ylabel('TD Error')
    axes.spines[['top', 'right']].set_visible(False)
    if return_handle:
        fig.tight_layout()
        fig.show()
        return fig, axes


def get_td_marginal_estimates(model_results, regressor_name):
    """
    Calculates the Marginal Mean estimate and SE for a main effect,
    averaging over its interaction with Context.

    Formula: Beta_Marginal = Beta_Main + 0.5 * Beta_Int_Context
    """
    params = model_results.params
    cov_params = model_results.cov_params()

    # 1. Base weights: Main Effect = 1.0
    weights = {regressor_name: 1.0}

    # Helper to find interaction name in params
    def find_interaction(main, inter_term_fragment):
        # Statsmodels names can be "A:B" or "B:A"
        for p in params.index:
            if main in p and inter_term_fragment in p and ":" in p:
                return p
        return None

    # 2. Identify Interactions to marginalize over
    # In the TD model, we only marginalize over 'context'
    context_frag = "context"
    term_context = find_interaction(regressor_name, context_frag)

    if term_context:
        weights[term_context] = 0.5

    # 3. Calculate Estimate (Linear Combination)
    estimate = sum(params[name] * w for name, w in weights.items() if name in params)

    # 4. Calculate Variance (w' * Cov * w)
    variance = 0.0
    for name_i, w_i in weights.items():
        if name_i not in cov_params.index: continue
        for name_j, w_j in weights.items():
            if name_j not in cov_params.index: continue
            variance += w_i * w_j * cov_params.loc[name_i, name_j]

    se = np.sqrt(variance)

    return estimate, se, weights


def plot_td_lmem_coefficients(model_results, axes=None):
    """
    Plots the LMEM coefficients for the TD-Error model, mirroring the DA plot.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        return_handle = True
    else:
        fig = None
        return_handle = False

    # Define mapping to simplify statsmodels variable names
    # Assuming context is 0 and 1, Treatment(1) makes 0 the non-reference
    name_map = {
        "C(context, Treatment(1))[T.0]": "context",
        "log_time_in_port_std": "time",
        "log_event_timer_std": "IRI"
    }

    plot_data = []

    # Process known main effects to get MARGINAL means
    for tech_name, simple_name in name_map.items():
        if tech_name in model_results.params:
            est, se, _ = get_td_marginal_estimates(model_results, tech_name)
            plot_data.append({
                'name': simple_name,
                'estimate': est,
                'lower': est - 1.96 * se,
                'upper': est + 1.96 * se,
                'is_main': True
            })

    # Process interactions (keep as raw coefficients)
    for p in model_results.params.index:
        if ':' in p and p not in name_map:
            # Generate a simplified name
            simple = p
            for k, v in name_map.items():
                simple = simple.replace(k, v)
            # Cleanup naming
            simple = simple.replace(":", " * ")

            # Only keep interactions relevant to our mapped terms
            if any(x in simple for x in ['time', 'context', 'IRI']):
                est = model_results.params[p]
                ci = model_results.conf_int().loc[p]

                plot_data.append({
                    'name': simple,
                    'estimate': est,
                    'lower': ci[0],
                    'upper': ci[1],
                    'is_main': False
                })

    # Convert to DataFrame
    df_plot = pd.DataFrame(plot_data)

    # Sort to match the style of the DA plot as closely as possible
    desired_order = [
        'time', 'IRI', 'context',
        'time * IRI', 'time * context', 'IRI * context'
    ]
    df_plot['sort_cat'] = pd.Categorical(df_plot['name'], categories=desired_order, ordered=True)
    df_plot = df_plot.sort_values('sort_cat').dropna()

    # Calculate errors for errorbar
    yerr = [df_plot['estimate'] - df_plot['lower'], df_plot['upper'] - df_plot['estimate']]

    # Plot
    axes.errorbar(
        x=df_plot['name'],
        y=df_plot['estimate'],
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=2,
        linewidth=1.5,
        markersize=4,
        ecolor='red'
    )

    axes.axhline(y=0, color='grey', linestyle='--')
    axes.set_ylabel('Coefficient (95% CI)')

    # Using fixedLocator to prevent warning before set_xticklabels
    axes.set_xticks(range(len(df_plot['name'])))
    axes.set_xticklabels(df_plot['name'], rotation=10, ha='right')

    axes.set_xlabel('Regressor')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    if return_handle:
        plt.tight_layout()
        plt.show()
        return fig, axes


def make_figure(master_df):
    """
        Sets up a 3-row composite figure.
        Row 1: Full width
        Row 2: 5:1 width ratio split
        Row 3: Full width
        """
    # 1. Initialize figure (12 inches wide, 12 inches tall fits 3 rows nicely)
    fig = plt.figure(figsize=(12, 15))

    # 2. Setup GridSpec: 3 rows, 6 columns
    # (Using 6 columns allows us to do a 5:1 ratio for the second row easily)
    gs = gridspec.GridSpec(nrows=4, ncols=6, figure=fig, hspace=0.4, wspace=0.4)

    # 3. Assign axes to the grid
    ax1 = fig.add_subplot(gs[0, :])  # Row 1: Spans all 6 columns
    ax2 = fig.add_subplot(gs[1, :5])  # Row 2 Left: Spans first 5 columns
    ax3 = fig.add_subplot(gs[1, 5])  # Row 2 Right: Spans the 6th column
    ax4 = fig.add_subplot(gs[2, :])  # Row 3: Spans all 6 columns
    ax5 = fig.add_subplot(gs[3, :])

    # --- PLOT ROW 1 ---
    fige_DA_vs_NRI_v2(master_df, dodge=True, axes=ax1)
    ax1.set_title("TD Error vs. Time in Port", pad=6)

    # --- PLOT ROW 2 ---
    figf_DA_vs_NRI_block_split_v2(master_df, axes=ax2)
    ax2.set_title("TD Error vs. Time (Split by Block)", pad=6)

    figf_summary_block_split(master_df, axes=ax3)
    ax3.set_title("Summary", pad=6)

    # --- PLOT ROW 3 ---
    fige_DA_vs_IRI_binned(master_df, axes=ax4)
    ax4.set_title("TD Error vs. IRI", pad=6)

    # --- PLOT ROW 4 ---
    model_results, df = run_td_error_lmem_analysis(master_df)
    plot_td_lmem_coefficients(model_results, axes=ax5)
    ax5.set_title("Linear Mixed-Effects Model Results", pad=6)

    # --- SYNC Y-LIMITS ---
    # Fetch the y-limits that matplotlib auto-calculated for each individual plot
    # (This ensures we don't accidentally cut off your hardcoded p-value brackets)
    axes = [ax1, ax2, ax3, ax4]
    min_y = min([ax.get_ylim()[0] for ax in axes])
    max_y = max([ax.get_ylim()[1] for ax in axes])

    # Apply the global min/max to all subplots
    for ax in axes:
        ax.set_ylim(min_y, 1.08)

    # Clean up the summary plot (ax3) so it looks like an extension of ax2
    ax3.set_ylabel("")
    # ax3.set_yticklabels([])

    axis_letter_pairs = zip([ax1, ax2, ax3, ax4, ax5], ['a', 'b', 'c', 'd', 'e'])

    for ax, letter in axis_letter_pairs:
        ax.text(0.0, 1.0, letter,
                transform=(ax.transAxes + ScaledTranslation(-20 / 72, +7 / 72, fig.dpi_scale_trans)),
                fontsize=16, va='bottom', fontfamily='sans-serif', weight='bold')

    # Final layout adjustment
    plt.tight_layout()
    plt.show()

    return fig, axes


def save_publication_figure(fig, filename="composite_summary_figure.png", save_folder="figures", dpi=300):
    """
    Saves a Matplotlib figure at high resolution.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The name of the file (supports .png, .pdf, .svg, etc.).
        save_folder (str): The directory where the figure should be saved.
        dpi (int): Dots per inch. 300 is standard for publication/print.
    """
    # 1. Ensure the target directory exists
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. Construct the full file path
    full_path = save_path / filename

    # 3. Save the figure
    print(f"💾 Saving figure to {full_path} at {dpi} DPI...")
    fig.savefig(
        full_path,
        dpi=dpi,
        bbox_inches='tight',  # Prevents labels/letters from being cropped off
        facecolor='white',  # Ensures the background isn't transparent
        edgecolor='none'
    )
    print("✅ Figure saved successfully!")


def main():
    SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animal_list = SZ_animals + RK_animals
    master_df = load_and_concat_population_data(animal_ids=animal_list, file_name="tde_reward_features")
    print('hello')


if __name__ == "__main__":
    # main()

    SZ_animals = ['SZ036', 'SZ037', 'SZ038', 'SZ039', 'SZ042', 'SZ043']
    RK_animals = ['RK007', 'RK008']
    animal_list = SZ_animals + RK_animals
    # animal_list = ["SZ036"]
    master_df = load_and_concat_population_data(animal_ids=animal_list, file_name="tde_reward_features")

    if not master_df.empty:
        fig, axes = make_figure(master_df)
        project_root = Path(config.MODELING_PROJECT_ROOT)
        save_folder = project_root / Path(config.STEP4_MODELING_PLOTS_SUBDIR)

        save_publication_figure(fig, filename="modeling_tde_vs_reward_features.png", save_folder=save_folder, dpi=300)

    # fige_DA_vs_NRI_v2(master_df, dodge=True, axes=None)
    # figf_DA_vs_NRI_block_split_v2(master_df, axes=None)
    # figf_summary_block_split(master_df, axes=None)
    # fige_DA_vs_IRI_binned(master_df, axes=None)
    # plot_td_lmem_coefficients(model_results, axes=None)
    print('hello')
