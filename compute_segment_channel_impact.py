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
