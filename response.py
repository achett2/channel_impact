def plot_causal_response_from_df(response_df, channel_name, show_empirical=True, show_mroi=False):
    """
    Plot causal and empirical response curves or mROI curve using precomputed response DataFrame.
    """

    df = response_df[response_df['channel'] == channel_name]

    if show_mroi:
        plt.figure(figsize=(7, 5))
        plt.plot(df['dose'], df['mROI'], label='mROI', linewidth=2, alpha=0.7)
        plt.xlabel('dose')
        plt.ylabel('Marginal ROI')
        plt.title(f'Marginal ROI - {channel_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        plt.figure(figsize=(7, 5))
        plt.plot(df['dose'], df['causal_response'], label='Causal Response Curve', linewidth=2)
        if show_empirical and 'empirical_avg_ite' in df.columns:
            plt.scatter(df['dose'], df['empirical_avg_ite'], color='black', label='Empirical Avg ITEs', zorder=3)
        plt.xlabel(f'{channel_name} Exposure (e.g., # of Rep Visits)')
        plt.ylabel('Expected Change in Scripts')
        plt.title(f'Causal vs Empirical Response — {channel_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

def generate_channel_response_curves(model_data, feature_colsm, treatments, train_months, test_month, npi_sample_size=10000):
    """
    Fits causal models (CausalForestDML) per channel and returns causal + empirical response data.
    This version handles both 1D and 2D treatment/effect outputs safely.

    Parameters:
    -----------
    npi_sample_size : int or None, default=10000
        Number of NPIs to sample for computational efficiency.
        Set to None to use all NPIs (no sampling).
    """

    all_responses = []

    # --- 1. Sample NPIs for computational efficiency ---
    if npi_sample_size is None:
        # Use all NPIs
        print("Using all NPIs (no sampling)")
        model_data2 = model_data.copy()
    else:
        # Sample specified number of NPIs
        print(f"Sampling {npi_sample_size} NPIs")
        npi_sample = model_data['npi'].drop_duplicates().sample(n=npi_sample_size, random_state=42)
        model_dataq = model_data[model_data['npi'].isin(npi_sample)]
        model_data2 = model_dataq.copy()

    # --- 2. Split train/test periods ---
    train = model_data2[model_data2['ds'].isin(train_months)]
    test = model_data2[model_data2['ds'] == test_month]

    Y_train = train['Y'].values
    Y_test = test['Y'].values

    # --- 3. Loop through all treatment channels ---
    for treatment in treatments:
        print(f'\n=== Estimating causal response for: {treatment} ===')

        channel_name = treatment
        channel_index = 0  # single treatment modeling

        # --- 3a. Drop current treatment from feature set ---
        feature_cols = [f for f in feature_colsm if f != treatment]

        # --- 3b. Prepare data ---
        X_train = train[feature_cols].values
        X_test = test[feature_cols].values
        T_train = train[[treatment]].values
        T_test = test[[treatment]].values

        # --- 4. Fit causal forest model ---
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1),
            n_estimators=200,
            min_samples_leaf=10,
            cv=2,
            random_state=42,
            n_jobs=-1
        )

        cf.fit(Y=Y_train, T=T_train, X=X_train)

        # --- 5. Predict ITEs on test set ---
        ites = cf.effect(X_test)

        # ✅ Ensure 2D shape for safe indexing
        if ites.ndim == 1:
            ites = ites.reshape(-1, 1)
        if T_test.ndim == 1:
            T_test = T_test.reshape(-1, 1)

        # --- 6. Compute causal response (dose-response curve) ---
        baseline_T = T_test.mean(axis=0, keepdims=True)
        n = X_test.shape[0]
        dose_grid = np.linspace(0, np.percentile(T_test, 99.9), 25)
        avg_pred = []

        for t in dose_grid:
            T_cf = np.tile(baseline_T, (n, 1))
            T_cf[:, channel_index] = t
            eff = cf.effect(X_test, T0=np.tile(baseline_T, (n, 1)), T1=T_cf)
            avg_pred.append(eff.mean())

        causal_response = np.array(avg_pred)
        causal_response -= causal_response[0]  # normalize baseline

        # --- 7. Compute empirical averages safely ---
        obs_df = pd.DataFrame({
            'treatment': T_test[:, channel_index],
            'ite': ites[:, channel_index]
        })
        obs_df['treatment_round'] = obs_df['treatment'].round(0)
        empirical_avg = obs_df.groupby('treatment_round')['ite'].mean().reset_index()
        empirical_avg = empirical_avg.rename(columns={'treatment_round': 'dose', 'ite': 'empirical_avg_ite'})

        # --- 8. Combine causal + empirical data ---
        response_df = pd.DataFrame({'dose': dose_grid, 'causal_response': causal_response})
        response_df = response_df.merge(empirical_avg, on='dose', how='outer')
        response_df['channel'] = channel_name
        response_df['mROI'] = response_df['causal_response'].diff() / response_df['dose'].diff()

        all_responses.append(response_df)

    # --- 9. Combine all channel results ---
    all_responses_df = pd.concat(all_responses, ignore_index=True)
    return all_responses_df
