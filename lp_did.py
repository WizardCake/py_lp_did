import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def detect_treatment_horizon(df, treatment_var, unit_var, time_var):
    """
    Detect the time when treatment starts for each unit.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        treatment_var (str): Column name for treatment indicator (0 or 1).
        unit_var (str): Column name for unit identifiers.
        time_var (str): Column name for time periods.

    Returns:
        pd.DataFrame: DataFrame with an additional 'treatment_horizon' column.
    """
    # Find the first time period when treatment occurs for each unit
    treatment_start = df[df[treatment_var] == 1].groupby(unit_var)[time_var].min().reset_index()
    treatment_start.columns = [unit_var, 'treatment_horizon']
    # Merge with the original DataFrame
    df = df.merge(treatment_start, on=unit_var, how='left')
    return df

def apply_dynamic_clean_control(df, treatment_var, time_var, treatment_horizon_var):
    """
    Exclude units from the control group after they receive treatment.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        treatment_var (str): Column name for treatment indicator (0 or 1).
        time_var (str): Column name for time periods.
        treatment_horizon_var (str): Column name for treatment start times.

    Returns:
        pd.DataFrame: Filtered DataFrame with dynamic control units.
    """
    # Keep units that are untreated or have not yet received treatment at time t
    condition = df[treatment_horizon_var].isna() | (df[time_var] < df[treatment_horizon_var])
    df_clean = df[condition].copy()
    return df_clean

def local_projection(df, outcome_var, treatment_var, horizons, unit_var, time_var, covariates=None):
    """
    Perform local projection regressions over specified time horizons.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        outcome_var (str): Column name for the outcome variable.
        treatment_var (str): Column name for treatment indicator (0 or 1).
        horizons (list or range): Time horizons to project.
        unit_var (str): Column name for unit identifiers.
        time_var (str): Column name for time periods.
        covariates (list, optional): List of additional covariate column names.

    Returns:
        dict: Dictionary containing regression results for each horizon.
    """
    results = {}
    for h in horizons:
        # Calculate the change in outcome variable at horizon h
        df[f'outcome_h{h}'] = df.groupby(unit_var)[outcome_var].shift(-h) - df[outcome_var]
        # Drop missing values
        df_h = df.dropna(subset=[f'outcome_h{h}', treatment_var])
        # Apply dynamic clean control
        df_h = apply_dynamic_clean_control(df_h, treatment_var, time_var, 'treatment_horizon')
        # Construct regression formula
        formula = f'outcome_h{h} ~ {treatment_var}'
        if covariates:
            formula += ' + ' + ' + '.join(covariates)
        # Fit the regression model with clustered standard errors
        model = ols(formula, data=df_h).fit(cov_type='cluster', cov_kwds={'groups': df_h[unit_var]})
        results[h] = model
        # Display summary for each horizon
        print(f"\nHorizon {h} regression results:")
        print(model.summary())
    return results

def extract_effects(results, treatment_var):
    """
    Extract estimated treatment effects and statistics from regression results.

    Parameters:
        results (dict): Dictionary of regression results from local projections.
        treatment_var (str): Column name for treatment indicator.

    Returns:
        pd.DataFrame: DataFrame containing treatment effects and statistics.
    """
    data = []
    for h, result in results.items():
        coef = result.params.get(treatment_var, np.nan)
        stderr = result.bse.get(treatment_var, np.nan)
        tvalue = result.tvalues.get(treatment_var, np.nan)
        pvalue = result.pvalues.get(treatment_var, np.nan)
        conf_int = result.conf_int().loc[treatment_var].values
        data.append({
            'horizon': h,
            'estimate': coef,
            'std_error': stderr,
            't_value': tvalue,
            'p_value': pvalue,
            'conf_lower': conf_int[0],
            'conf_upper': conf_int[1]
        })
    effects_df = pd.DataFrame(data)
    return effects_df

def calculate_average_effect(effects_df):
    """
    Calculate the average treatment effect from the estimated effects.

    Parameters:
        effects_df (pd.DataFrame): DataFrame containing treatment effects.

    Returns:
        float: Average treatment effect across all horizons.
    """
    average_effect = effects_df['estimate'].mean()
    return average_effect

def plot_results(effects_df):
    """
    Plot the estimated treatment effects over time horizons.

    Parameters:
        effects_df (pd.DataFrame): DataFrame containing treatment effects.
    """
    horizons = effects_df['horizon']
    estimates = effects_df['estimate']
    conf_lower = effects_df['conf_lower']
    conf_upper = effects_df['conf_upper']

    plt.figure(figsize=(10, 6))
    plt.plot(horizons, estimates, marker='o', label='Estimated Treatment Effect')
    plt.fill_between(horizons, conf_lower, conf_upper, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Horizon')
    plt.ylabel('Treatment Effect')
    plt.title('Local Projections: Treatment Effects Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_synthetic_data_heterogeneous():
    """
    Generate a synthetic dataset with heterogeneous treatment effects and staggered adoption.

    Returns:
        pd.DataFrame: Synthetic dataset for analysis.
    """
    # Define parameters for the dataset
    n_units = 9                  # Number of units
    n_periods = 36               # Number of time periods
    proportion_treated = 1 / 3   # Proportion of units receiving treatment
    random_walk_effect = 0.42    # Intensity of random walk effect

    # Define staggered adoption parameters
    min_treatment_start = int(0.5 * n_periods)
    max_treatment_start = int(0.8 * n_periods)

    # Generate IDs and time periods
    ids = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Randomly select units to receive treatment
    treated_units = np.random.choice(np.arange(n_units), size=int(n_units * proportion_treated), replace=False)

    # Randomly assign treatment start times for each unit
    treatment_start = np.random.choice(np.arange(min_treatment_start, max_treatment_start), size=n_units)
    treatment = np.zeros(n_units * n_periods)

    # Apply staggered treatment to selected units
    for i in treated_units:
        start_period = treatment_start[i]
        treatment_indices = np.where((ids == i) & (time >= start_period))
        treatment[treatment_indices] = 1

    # Function to generate time series with seasonality using Holt-Winters method
    def generate_holt_winters_series(n_periods, seasonal_periods=12):
        time_series = np.arange(n_periods)
        seasonal_effect = np.sin(2 * np.pi * time_series / seasonal_periods)
        trend = 0.05 * time_series
        noise = np.random.normal(0, 0.1, n_periods)
        data = trend + seasonal_effect + noise
        model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        fit = model.fit(optimized=True)
        return fit.fittedvalues

    # Function to generate a random walk series
    def generate_random_walk(n_periods, step_size=0.02):
        steps = np.random.normal(0, step_size, n_periods)
        return np.cumsum(steps)

    # Initialize list to store outcome values
    y_values = []

    # Generate heterogeneous treatment effects
    treatment_effects = {i: np.random.uniform(0.03, 0.08) for i in treated_units}

    # Generate time series data for each unit
    for i in range(n_units):
        # Generate base time series with trend and seasonality
        y_series = generate_holt_winters_series(n_periods)
        # Add random walk effect
        random_walk = generate_random_walk(n_periods, step_size=random_walk_effect)
        y_series += random_walk
        # Apply treatment effect if the unit is treated
        if i in treated_units:
            start_period = treatment_start[i]
            treatment_effect = treatment_effects[i]
            for t in range(start_period, n_periods):
                dynamic_effect = treatment_effect * (t - start_period) / (n_periods - start_period)
                y_series[t] += dynamic_effect
        # Append to the list of outcome values
        y_values.extend(y_series)

    # Generate random covariates
    covariate1 = np.random.normal(size=n_units * n_periods)
    covariate2 = np.random.normal(size=n_units * n_periods)

    # Create the final DataFrame
    df = pd.DataFrame({
        'id': ids,
        'time': time,
        'treatment': treatment,
        'y': y_values,
        'covariate1': covariate1,
        'covariate2': covariate2
    })
    return df

def main():
    """
    Main function to execute the local projection analysis.
    """
    # Generate synthetic data
    hetero = generate_synthetic_data_heterogeneous()

    # Detect treatment horizons
    hetero = detect_treatment_horizon(hetero, 'treatment', 'id', 'time')

    # Define time horizons for local projections
    horizons = range(0, 20)

    # Run local projection regressions
    results = local_projection(
        hetero,
        outcome_var='y',
        treatment_var='treatment',
        horizons=horizons,
        unit_var='id',
        time_var='time',
        covariates=['covariate1', 'covariate2']
    )

    # Extract treatment effects and statistics
    effects_df = extract_effects(results, 'treatment')

    # Calculate average treatment effect
    average_effect = calculate_average_effect(effects_df)

    # Display the estimated effects
    print("\nEstimated Treatment Effects over Horizons:")
    print(effects_df)
    print(f"\nAverage Treatment Effect: {average_effect}")

    # Plot the results
    plot_results(effects_df)

if __name__ == '__main__':
    main()
