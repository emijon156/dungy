# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px
import scipy.stats as stats

st.set_page_config(page_title="Exoplanets & Dark Matter", layout="wide")
st.title("Exoplanets & Dark Matter Analysis")
#hehe
# --- 1. DATA LOAD & CLEAN ---
@st.cache_data
def load_data(path='expl.csv'):
    df = pd.read_csv(path)
    df_clean = df.dropna(subset=['ra', 'dec', 'sy_dist']).copy()
    
    coords = SkyCoord(
        ra=df_clean['ra'].values*u.deg,
        dec=df_clean['dec'].values*u.deg,
        distance=df_clean['sy_dist'].values*u.pc,
        frame='icrs'
    )
    gal_coords = coords.transform_to('galactocentric')
    df_clean['galactocentric_x'] = gal_coords.x.value
    df_clean['galactocentric_y'] = gal_coords.y.value
    df_clean['galactocentric_z'] = gal_coords.z.value
    df_clean['r_galactic'] = np.sqrt(
        df_clean['galactocentric_x']**2 +
        df_clean['galactocentric_y']**2 +
        df_clean['galactocentric_z']**2
    )
    
    rho_0 = 0.0106
    R_s = 12500
    r_over_Rs = df_clean['r_galactic'] / R_s
    df_clean['dark_matter_density'] = rho_0 / (r_over_Rs * (1 + r_over_Rs)**2)
    df_clean['log_dm_density'] = np.log10(df_clean['dark_matter_density'])
    
    planet_cols = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_bmasse', 'pl_rade']
    df_planets = df_clean.dropna(subset=planet_cols).copy()
    
    # Planet mass & period categories
    df_planets['mass_category'] = pd.cut(df_planets['pl_bmasse'],
                                        bins=[0, 10, 100, 1000, 10000],
                                        labels=['Earth-like', 'Neptune-like', 'Jupiter-like', 'Super-Jupiter'])
    df_planets['period_category'] = pd.cut(df_planets['pl_orbper'],
                                          bins=[0, 10, 100, 1000, 10000],
                                          labels=['Short', 'Medium', 'Long', 'Very Long'])
    return df_clean, df_planets

df_clean, df_planets = load_data()
st.subheader("Data Preview")
st.dataframe(df_clean.head())

st.markdown(f"**Working with {len(df_planets)} planets with complete properties**")

# --- 2. MULTIVARIATE ANALYSIS ---
st.subheader("Multivariate OLS Analysis")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_planets[['log_dm_density', 'r_galactic', 'pl_orbper', 'pl_bmasse']])
df_planets[['dm_scaled', 'r_scaled', 'period_scaled', 'mass_scaled']] = scaled_features

model = ols('period_scaled ~ dm_scaled + r_scaled + mass_scaled + dm_scaled:r_scaled', data=df_planets).fit()
st.text(model.summary())

# --- 3. STRATIFIED ANALYSIS BY PLANET TYPE ---
st.subheader("Stratified Analysis (Spearman correlation by mass & period categories)")
results = []
for mass_cat in df_planets['mass_category'].unique():
    for period_cat in df_planets['period_category'].unique():
        subset = df_planets[(df_planets['mass_category'] == mass_cat) &
                           (df_planets['period_category'] == period_cat)]
        if len(subset) > 20:
            corr, pval = spearmanr(subset['pl_orbper'], subset['log_dm_density'])
            results.append({
                'mass_category': mass_cat,
                'period_category': period_cat,
                'sample_size': len(subset),
                'spearman_r': corr,
                'p_value': pval
            })
results_df = pd.DataFrame(results).sort_values('p_value')
st.dataframe(results_df.head(10))

# --- 4. ALTERNATIVE CORRELATIONS ---
st.subheader("Alternative Correlation Methods")
tau, pval_tau = kendalltau(df_planets['pl_orbper'], df_planets['log_dm_density'])
st.write(f"Kendall's tau: {tau:.4f}, p-value: {pval_tau:.4f}")

def partial_correlation(x, y, z):
    x_res = sm.OLS(x, sm.add_constant(z)).fit().resid
    y_res = sm.OLS(y, sm.add_constant(z)).fit().resid
    return pearsonr(x_res, y_res)[0]

part_corr = partial_correlation(df_planets['pl_orbper'], df_planets['log_dm_density'], df_planets['r_galactic'])
st.write(f"Partial correlation (controlling for distance): {part_corr:.4f}")

# --- 5. CLUSTERING ---
st.subheader("Clustering Analysis (DBSCAN)")
cluster_data = df_planets[['log_dm_density', 'r_galactic', 'pl_orbper', 'pl_bmasse']].dropna()
cluster_data_scaled = StandardScaler().fit_transform(cluster_data)
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(cluster_data_scaled)
df_planets.loc[cluster_data.index, 'cluster'] = clusters
st.write(df_planets['cluster'].value_counts())

# --- 6. BAYESIAN CORRELATION ---
st.subheader("Bayesian Correlation Estimation")
def bayesian_correlation(x, y, n_iter=1000):
    corrs = []
    for _ in range(n_iter):
        sample_idx = np.random.choice(len(x), size=len(x), replace=True)
        x_sample = x.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        if len(x_sample) > 10:
            corr, _ = spearmanr(x_sample, y_sample)
            if not np.isnan(corr):
                corrs.append(corr)
    return np.mean(corrs), np.std(corrs)

bayes_mean, bayes_std = bayesian_correlation(df_planets['pl_orbper'], df_planets['log_dm_density'])
st.write(f"Bayesian correlation estimate: {bayes_mean:.4f} ± {bayes_std:.4f}")

# --- 7. SPATIAL AUTOCORRELATION ---
st.subheader("Spatial Autocorrelation (Moran's I)")
model_residuals = sm.OLS(df_planets['pl_orbper'], sm.add_constant(df_planets[['log_dm_density','r_galactic']])).fit().resid

def morans_i(residuals, coords):
    n = len(residuals)
    mean_res = np.mean(residuals)
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis]-coords)**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    weights = 1/dist_matrix
    weights[np.isinf(weights)] = 0
    weights = weights/np.sum(weights)
    numerator = np.sum(weights*(residuals-mean_res)[:,None]*(residuals-mean_res))
    denominator = np.sum((residuals-mean_res)**2)
    if denominator == 0: return np.nan
    return (n/np.sum(weights))*(numerator/denominator)

coords_array = df_planets[['galactocentric_x','galactocentric_y','galactocentric_z']].values
moran_i_val = morans_i(model_residuals.values, coords_array)
st.write(f"Moran's I: {moran_i_val:.4f}")

# --- 8. VISUALIZATIONS ---
st.subheader("Interactive Visualizations")
fig1 = px.scatter(df_planets,
                  x='log_dm_density', y='pl_orbper',
                  color='mass_category', size='pl_bmasse',
                  hover_data=['pl_rade','pl_orbper','pl_bmasse','cluster'],
                  title='Orbital Period vs Log(DM Density) by Planet Mass')
fig1.update_yaxes(type='log')
st.plotly_chart(fig1)

fig2 = px.scatter_3d(df_planets,
                     x='r_galactic', y='log_dm_density', z='pl_orbper',
                     color='mass_category', size='pl_bmasse',
                     hover_data=['pl_rade','pl_orbper','pl_bmasse','cluster'],
                     title='3D: Galactic Radius vs DM Density vs Orbital Period')
fig2.update_layout(scene=dict(zaxis_type='log'))
st.plotly_chart(fig2)

fig3 = px.box(df_planets,
              x='mass_category', y='pl_orbper',
              log_y=True, points="all",
              title="Distribution of Orbital Period by Planet Mass Category")
st.plotly_chart(fig3)

# --- 9. POWER ANALYSIS & RECOMMENDATIONS ---
st.subheader("Power Analysis & Recommendations")
def power_analysis(corr, n, alpha=0.05):
    effect_size = np.abs(corr)
    ncp = effect_size*np.sqrt(n-2)/np.sqrt(1-effect_size**2)
    t_crit = stats.t.ppf(1-alpha/2, n-2)
    power = 1 - stats.t.cdf(t_crit, n-2, ncp) + stats.t.cdf(-t_crit, n-2, ncp)
    return power

st.write(f"Power to detect r=0.07 with n={len(df_planets)}: {power_analysis(0.07, len(df_planets)):.3f}")
for effect_size in [0.1, 0.15, 0.2]:
    st.write(f"Power to detect r={effect_size}: {power_analysis(effect_size,len(df_planets)):.3f}")

st.markdown("""
**Recommendations:**
1. Weak correlations (r ≈ 0.07) suggest any DM effect is subtle  
2. Focus on specific planet subtypes where effects might be stronger  
3. Consider system-level properties rather than individual planets  
4. Relationship might be non-linear – try polynomial or spline models  
5. Spatial autocorrelation should be accounted for
""")
