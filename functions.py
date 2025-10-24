#############
# ATE - AVERAGE SENSITIVITY TO CHANNEL BY SEGMENT
#############
def compute_segment_channel_impact(final_results, categorical_mapping_df, segment_order, group_channel_func):
    """
    Compute average treatment effects (ATE) by segment and test_month,
    reshape to long format, and assign channel groups.

    Parameters
    ----------
    final_results : pd.DataFrame
        DataFrame containing segment, test_month, and ITE_* columns.
    categorical_mapping_df : pd.DataFrame
        DataFrame with columns ['original_value', 'encoded_value', 'column']
        used to map segment numeric values to segment names.
    segment_order : list
        Desired order of segments (used for reindexing).
    group_channel_func : callable
        Function that takes a channel name and returns a grouped channel label.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ['segment', 'test_month', 'channel', 'impact', 'channel_group']
    """
    # --- Map segment values to names using categorical_mapping_df ---
    segment_mapping = categorical_mapping_df[categorical_mapping_df['column'] == 'segment']
    segment_map_dict = dict(zip(segment_mapping['encoded_value'], segment_mapping['original_value']))

    # Create a copy to avoid modifying the original dataframe
    final_results = final_results.copy()
    final_results['segment'] = final_results['segment'].map(segment_map_dict)

    # --- Identify ITE columns ---
    ce_cols = final_results.filter(like='ITE_').columns

    # --- Calculate group means ---
    group_means = final_results.groupby(['segment', 'test_month'])[ce_cols].mean()

    # --- Calculate overall mean ---
    overall_mean = (
        final_results.groupby('test_month')[ce_cols]
        .mean()
        .assign(segment='Overall')
        .set_index('segment', append=True)
        .reorder_levels(['segment', 'test_month'])
    )

    # --- Combine ---
    final = pd.concat([group_means, overall_mean]).sort_index(level=1)

    # --- Reorder segments ---
    final = final.reindex(segment_order, level=0)

    # --- Convert to long format ---
    final_long = (
        final.stack()
        .to_frame(name='impact')
        .rename_axis(['segment', 'test_month', 'channel'])
        .reset_index()
    )

    # --- Add channel group ---
    final_long['channel_group'] = final_long['channel'].apply(group_channel_func)

    return final_long

import pandas as pd

import pandas as pd

def compute_lift_summary(final_results, outcomes, hcp_cohort, group_channel_func, 
                         segment_order=None):
    """
    Compute lift metrics (TTT, ATT, Contribution) for each treatment channel,
    aggregated by segment and test month.

    Parameters
    ----------
    final_results : pd.DataFrame
        DataFrame containing columns for ITE_* (causal effects) and treatment indicators.
    outcomes : pd.DataFrame
        DataFrame containing outcome values with columns ['npi', 'ds', 'projected_value'].
    hcp_cohort : pd.DataFrame
        DataFrame mapping NPIs to 'segment'.
    group_channel_func : callable
        Function that maps channel names to a higher-level group (e.g., BAE, DTC, Digital).
    segment_order : list, optional
        Desired order of segments for final output sorting.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['segment', 'test_month', 'channel', 'channel_group', 
         'TTT', 'ATT', 'Contribution', 'projected_value']
    """

    # --- Identify treatment channels ---
    treatcols = final_results.columns[final_results.columns.str.startswith('ITE_')]
    treatcols = treatcols.str.replace('ITE_', '', regex=False)

    # --- Calculate Lift for each channel ---
    for ch in treatcols:
        treatment_col = ch
        ce_col = 'ITE_' + ch
        new_col = 'Lift ' + ch
        final_results[new_col] = final_results[treatment_col] * final_results[ce_col]

    lift_cols = final_results.filter(like='Lift ').columns

    # --- Group by segment and month ---
    group_stats = (
        final_results.groupby(['segment', 'test_month'])[lift_cols]
        .agg(['sum', 'mean'])
    )
    group_stats.columns = [f'{col[0]}_{col[1]}' for col in group_stats.columns]

    # --- Overall (all segments combined) ---
    overall_stats = (
        final_results.groupby('test_month')[lift_cols]
        .agg(['sum', 'mean'])
    )
    overall_stats.columns = [f'{col[0]}_{col[1]}' for col in overall_stats.columns]
    overall_stats = (
        overall_stats
        .assign(segment='Overall')
        .set_index('segment', append=True)
        .reorder_levels(['segment', 'test_month'])
    )

    # --- Combine segment-level and overall stats ---
    final_stats = pd.concat([group_stats, overall_stats]).sort_index(level=1)

    # --- Order segments ---
    if segment_order is not None:
        final_stats = final_stats.reindex(segment_order, level=0)

    # --- Reshape to long format ---
    final_stats_long = (
        final_stats.stack()
        .to_frame(name='impact')
        .rename_axis(['segment', 'test_month', 'metric'])
        .reset_index()
    )

    # --- Split metric into channel and stat (sum or mean) ---
    final_stats_long[['channel', 'stat']] = final_stats_long['metric'].str.extract(r'Lift (.*)_(sum|mean)')
    final_stats_long = final_stats_long.drop(columns='metric')

    # --- Pivot for sum and mean ---
    final_sum2 = final_stats_long.pivot_table(
        index=['segment', 'test_month', 'channel'],
        columns='stat',
        values='impact'
    ).reset_index()

    # --- Create channel group ---
    final_sum2['channel_group'] = final_sum2['channel'].apply(group_channel_func)

    # --- Compute contribution ---
    monthly_contribution_a = outcomes.merge(
        hcp_cohort[['npi', 'segment']], how='left', on='npi'
    )
    monthly_contribution_a = (
        monthly_contribution_a.groupby(['segment', 'ds'])['projected_value']
        .sum()
        .reset_index()
    )

    monthly_contribution_b = (
        outcomes.groupby(['ds'])['projected_value']
        .sum()
        .reset_index()
        .assign(segment='Overall')
    )

    monthly_contribution = pd.concat([monthly_contribution_a, monthly_contribution_b])

    final_sum2 = final_sum2.merge(
        monthly_contribution,
        how='left',
        left_on=['segment', 'test_month'],
        right_on=['segment', 'ds']
    )

    final_sum2['contribution'] = final_sum2['sum'] / final_sum2['projected_value']

    # --- Rename and reorder columns ---
    final_sum2 = final_sum2[
        ['segment', 'test_month', 'channel', 'channel_group',
         'sum', 'mean', 'contribution', 'projected_value']
    ]
    final_sum2.columns = [
        'segment', 'test_month', 'channel', 'channel_group',
        'TTT', 'ATT', 'Contribution', 'projected_value'
    ]

    # --- Optional segment & channel sorting ---
    if segment_order is not None:
        final_sum2['segment_order'] = final_sum2['segment'].map({
            seg: i for i, seg in enumerate(segment_order)
        })
    else:
        final_sum2['segment_order'] = 0

    final_sum2['channel_order'] = final_sum2['channel'].apply(
        lambda x: '0' + x if x == 'BAE Detailing F2F' else '1' + x
    )

    final_sum2 = (
        final_sum2
        .sort_values(by=['test_month', 'segment_order', 'channel_order'])
        .reset_index(drop=True)
        .drop(columns=['segment_order', 'channel_order'])
    )

    return final_sum2, monthly_contribution

