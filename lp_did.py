import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def detect_treatment_horizon(df, treatment_var, unit_var, time_var):
    """
    Detects the horizon (time of treatment start) for each unit.
    """
    treatment_start = df[df[treatment_var]==1].groupby(unit_var)[time_var].min().reset_index()
    treatment_start.columns = [unit_var, 'treatment_horizon']
    df = df.merge(treatment_start, on=unit_var, how='left')
    return df

def apply_dynamic_clean_control(df, treatment_var, time_var, treatment_horizon_var):
    """
    Applies clean control dynamically by excluding units that are treated at or before each time horizon.
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean[treatment_horizon_var].isna() | (df_clean[time_var] < df_clean[treatment_horizon_var])]
    return df_clean

def local_projection(df, outcome_var, treatment_var, horizons, unit_var, time_var, covariates=None):
    """
    Applies Local Projections method across specified horizons.
    """
    results = {}
    for h in horizons:
        df[f'outcome_h{h}'] = df.groupby(unit_var)[outcome_var].shift(-h) - df[outcome_var]
        df_h = df.dropna(subset=[f'outcome_h{h}', treatment_var])
        # Apply dynamic clean control
        df_h = apply_dynamic_clean_control(df_h, treatment_var, time_var, 'treatment_horizon')
        # Create formula
        formula = f'outcome_h{h} ~ {treatment_var}'
        if covariates:
            formula += ' + ' + ' + '.join(covariates)
        # Run regression
        model = ols(formula, data=df_h).fit(cov_type='cluster', cov_kwds={'groups': df_h[unit_var]})
        results[h] = model
        print(f"Horizon {h}:\n", model.summary())
    return results

def extract_effects(results):
    """
    Extracts the estimated treatment effects and standard errors from regression results.
    """
    data = []
    for h, result in results.items():
        coef = result.params.get('treatment', np.nan)
        stderr = result.bse.get('treatment', np.nan)
        tvalue = result.tvalues.get('treatment', np.nan)
        pvalue = result.pvalues.get('treatment', np.nan)
        conf_int = result.conf_int().loc['treatment'].values
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
    Calculates the average treatment effect from the effects DataFrame.
    """
    average_effect = effects_df['estimate'].mean()
    return average_effect

def plot_results(effects_df):
    """
    Plots the treatment effects across time horizons.
    """
    horizons = effects_df['horizon']
    estimates = effects_df['estimate']
    conf_lower = effects_df['conf_lower']
    conf_upper = effects_df['conf_upper']
    plt.figure(figsize=(10,6))
    plt.plot(horizons, estimates, marker='o', label='Estimated Treatment Effect')
    plt.fill_between(horizons, conf_lower, conf_upper, color='gray', alpha=0.2, label='95% CI')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Horizon')
    plt.ylabel('Treatment Effect')
    plt.title('Local Projections: Treatment Effects Over Time')
    plt.legend()
    plt.show()


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Definir parâmetros para o dataset
n_units = 9  # número de unidades (indivíduos)
n_periods = 36  # número de períodos de tempo
proportion_treated = 1/3  # proporção de indivíduos que recebem o tratamento
treatment_effect = 0.09  # efeito homogêneo do tratamento
random_walk_effect = 0.252  # intensidade do efeito do random walk

# Definir o ponto de início do tratamento (com adoção escalonada, ou seja, "staggered adoption")
min_treatment_start = int(0.5 * n_periods)  # Tratamento pode começar após 20% do tempo
max_treatment_start = int(0.8 * n_periods)  # Tratamento não pode começar após 80% do tempo

# Gerar IDs para as unidades
ids = np.repeat(np.arange(n_units), n_periods)

# Gerar a variável de tempo
time = np.tile(np.arange(n_periods), n_units)

# Selecionar 1/3 dos indivíduos para o tratamento
treated_units = np.random.choice(np.arange(n_units), size=int(n_units * proportion_treated), replace=False)

# Gerar o tratamento (tratamento começa em momentos escalonados, entre 20% e 80% do tempo)
treatment_start = np.random.choice(np.arange(min_treatment_start, max_treatment_start), size=n_units)
treatment = np.zeros(n_units * n_periods)

# Aplicar tratamento escalonado para os indivíduos selecionados
for i in treated_units:
    treatment[i * n_periods + treatment_start[i]: (i + 1) * n_periods] = 1

# Função para gerar séries temporais utilizando Holt-Winters (com sazonalidade)
def generate_holt_winters_series(n_periods, seasonal_periods=12):
    # Gera dados com sazonalidade
    time = np.arange(n_periods)
    seasonal_effect = np.sin(2 * np.pi * time / seasonal_periods)
    trend = 0.05 * time
    noise = np.random.normal(0, 0.1, n_periods)
    
    # Dados base (com tendência, sazonalidade e ruído)
    data = trend + seasonal_effect + noise
    
    # Aplicar Holt-Winters para suavizar a série
    model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    return fit.fittedvalues

# Função para gerar um pequeno random walk
def generate_random_walk(n_periods, step_size=0.02):
    steps = np.random.normal(0, step_size, n_periods)
    return np.cumsum(steps)  # Caminhada aleatória com soma cumulativa

# Inicializar a lista para armazenar as séries temporais simuladas
y_values = []

# Gerar a série temporal para cada indivíduo
for i in range(n_units):
    # Gerar a série temporal usando Holt-Winters com sazonalidade
    y_series = generate_holt_winters_series(n_periods)
    
    # Adicionar um pequeno random walk à série temporal
    random_walk = generate_random_walk(n_periods, step_size=random_walk_effect)
    y_series += random_walk
    
    # Se o indivíduo for tratado, aplicar o efeito dinâmico e homogêneo do tratamento
    if i in treated_units:
        treatment_period_start = treatment_start[i]
        
        # Efeito dinâmico: O efeito do tratamento aumenta ao longo do tempo, mas é homogêneo entre unidades
        for t in range(treatment_period_start, n_periods):
            dynamic_effect = treatment_effect * (t - treatment_period_start) / (n_periods - treatment_period_start)  # Efeito dinâmico ajustado
            y_series[t] += dynamic_effect  # Efeito cumulativo, mas homogêneo entre unidades
    
    # Armazenar a série temporal gerada
    y_values.extend(y_series)

# Gerar covariáveis aleatórias
covariate1 = np.random.normal(size=n_units * n_periods)
covariate2 = np.random.normal(size=n_units * n_periods)

# Criar o DataFrame final
homogeneo = pd.DataFrame({
    'id': ids,
    'time': time,
    'treatment': treatment,
    'y': y_values,
    'covariate1': covariate1,
    'covariate2': covariate2
})

# Exibir as primeiras linhas do dataset gerado
print(homogeneo.head())

# Salvar em um arquivo CSV, se necessário
# homogeneo.to_csv('synthetic_data_homogeneous.csv', index=False)

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Definir parâmetros para o dataset
n_units = 9  # número de unidades (indivíduos)
n_periods = 36  # número de períodos de tempo
proportion_treated = 1/3  # proporção de indivíduos que recebem o tratamento
random_walk_effect = 0.42  # intensidade do efeito do random walk

# Definir o ponto de início do tratamento (com adoção escalonada, ou seja, "staggered adoption")
min_treatment_start = int(0.5 * n_periods)  # Tratamento pode começar após 20% do tempo
max_treatment_start = int(0.8 * n_periods)  # Tratamento não pode começar após 80% do tempo

# Gerar IDs para as unidades
ids = np.repeat(np.arange(n_units), n_periods)

# Gerar a variável de tempo
time = np.tile(np.arange(n_periods), n_units)

# Selecionar 1/3 dos indivíduos para o tratamento
treated_units = np.random.choice(np.arange(n_units), size=int(n_units * proportion_treated), replace=False)

# Gerar o tratamento (tratamento começa em momentos escalonados, entre 20% e 80% do tempo)
treatment_start = np.random.choice(np.arange(min_treatment_start, max_treatment_start), size=n_units)
treatment = np.zeros(n_units * n_periods)

# Aplicar tratamento escalonado para os indivíduos selecionados
for i in treated_units:
    treatment[i * n_periods + treatment_start[i]: (i + 1) * n_periods] = 1

# Função para gerar séries temporais utilizando Holt-Winters (com sazonalidade)
def generate_holt_winters_series(n_periods, seasonal_periods=12):
    # Gera dados com sazonalidade
    time = np.arange(n_periods)
    seasonal_effect = np.sin(2 * np.pi * time / seasonal_periods)
    trend = 0.05 * time
    noise = np.random.normal(0, 0.1, n_periods)
    
    # Dados base (com tendência, sazonalidade e ruído)
    data = trend + seasonal_effect + noise
    
    # Aplicar Holt-Winters para suavizar a série
    model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    return fit.fittedvalues

# Função para gerar um pequeno random walk
def generate_random_walk(n_periods, step_size=0.02):
    steps = np.random.normal(0, step_size, n_periods)
    return np.cumsum(steps)  # Caminhada aleatória com soma cumulativa

# Inicializar a lista para armazenar as séries temporais simuladas
y_values = []

# Gerar uma intensidade de efeito de tratamento diferente para cada unidade tratada (heterogêneo)
treatment_effects = {i: np.random.uniform(0.03, 0.08) for i in treated_units}  # Efeito heterogêneo

# Gerar a série temporal para cada indivíduo
for i in range(n_units):
    # Gerar a série temporal usando Holt-Winters com sazonalidade
    y_series = generate_holt_winters_series(n_periods)
    
    # Adicionar um pequeno random walk à série temporal
    random_walk = generate_random_walk(n_periods, step_size=random_walk_effect)
    y_series += random_walk
    
    # Se o indivíduo for tratado, aplicar o efeito dinâmico e heterogêneo do tratamento
    if i in treated_units:
        treatment_period_start = treatment_start[i]
        treatment_effect = treatment_effects[i]  # Efeito heterogêneo para cada indivíduo
        
        # Efeito dinâmico: O efeito do tratamento aumenta ao longo do tempo, mas varia entre unidades
        for t in range(treatment_period_start, n_periods):
            dynamic_effect = treatment_effect * (t - treatment_period_start) / (n_periods - treatment_period_start)  # Efeito dinâmico ajustado
            y_series[t] += dynamic_effect  # Efeito cumulativo e heterogêneo entre unidades
    
    # Armazenar a série temporal gerada
    y_values.extend(y_series)

# Gerar covariáveis aleatórias
covariate1 = np.random.normal(size=n_units * n_periods)
covariate2 = np.random.normal(size=n_units * n_periods)

# Criar o DataFrame final
hetero = pd.DataFrame({
    'id': ids,
    'time': time,
    'treatment': treatment,
    'y': y_values,
    'covariate1': covariate1,
    'covariate2': covariate2
})

# Exibir as primeiras linhas do dataset gerado
print(hetero.head())

# Salvar em um arquivo CSV, se necessário
hetero.to_csv('synthetic_data_heterogeneous.csv', index=False)
hetero.to_excel('synthetic_data_heterogeneous.xlsx', index=False)


# Detect treatment horizon
hetero = detect_treatment_horizon(hetero, 'treatment', 'id', 'time')

# Define time horizons
horizons = range(0, 20)

# Run Local Projections
results = local_projection(hetero, 'y', 'treatment', horizons, 'id', 'time', covariates=['covariate1', 'covariate2'])

# Extract Effects
effects_df = extract_effects(results)

# Calculate Average Treatment Effect
average_effect = calculate_average_effect(effects_df)
print("\nEstimated Treatment Effects over Horizons:")
print(effects_df)
print(f"\nAverage Treatment Effect: {average_effect}")

# Plot Results
plot_results(effects_df)