#--------------------Imports---------------------
import streamlit as st   
import pandas as pd
import pyreadr
import tempfile
import os
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sns

#----------------Cache for streamlit--------------
@st.cache_data

#-----------------Functions-----------------------


def load_data(uploaded_file):
    if uploaded_file is None:
        return None
        
    file_name = uploaded_file.name
    extension = Path(file_name).suffix.lower()
    
    try:
        if extension in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file)
            
        elif extension == '.csv':
            return pd.read_csv(uploaded_file, sep=None, engine='python')
            
        elif extension in ['.rds', '.rda', '.rdata']:
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer()) # getbuffer() ist performanter als getvalue()
                tmp_path = tmp_file.name
            
            try:
                result = pyreadr.read_r(tmp_path)
                
                if result:
                    obj_name = list(result.keys())[0]
                    return result[obj_name]
                else:
                    st.error("Die R-Datei scheint leer zu sein.")
                    return None
            finally:
                
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten ({extension}): {e}")
        return None

def clean_and_convert(df, cols_to_numeric=None):
    # input: DataFrame and columuns that should be numeric
    # returns DataFrame without NaNs and numeric columns
    if df is None: return None
    df_c = df.copy()
    if cols_to_numeric:
        for col in cols_to_numeric:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
    return df_c.dropna().reset_index(drop=True)

def encode_categorical_data(df, columns_to_encode):
    # input: DataFrame and categorical columns
    # returns DataFrame with One-Hot encoded categorical columns
    if not columns_to_encode: return df
    return pd.get_dummies(df, columns=columns_to_encode, drop_first=True, dtype=int)

def get_target_stats(df, target_col):
    # input DataFrame & target column
    # returns first statistical values of target 
    vals = df[target_col]
    pos_vals = vals[vals > 0]
    metrics = ["Count", "Mean", "Median", "Std. Deviation", "Min", "Max"]
    
    stats_data = {
        "Metric": metrics,
        "Total (incl. 0)": [
            len(vals), vals.mean(), vals.median(), vals.std(), vals.min(), vals.max()
        ],
        "Values > 0": [
            len(pos_vals), pos_vals.mean(), pos_vals.median(), pos_vals.std(), pos_vals.min(), pos_vals.max()
        ]
    }
    return pd.DataFrame(stats_data)

def filter_target_range(df, target_col, percentile=1.0):
    # returns filtered target_col >0 and cut-off at percentile
    if df is None or target_col not in df.columns:
        return df, None
    
    df_copy = df.copy()
    
    # For further exploration only values > 0 are used 
    df_positive = df_copy[df_copy[target_col] > 0]
    
    if percentile < 1.0:
        threshold = df_positive[target_col].quantile(percentile)
        df_filtered = df_positive[df_positive[target_col] <= threshold].copy()
    else:
        df_filtered = df_positive.copy()
        threshold = df_positive[target_col].max()
        
    return df_filtered.reset_index(drop=True), threshold

def find_best_distribution1(data, candidate_distributions=None):
    # finds best distribution with aic comparison 
    # returns result of distribution fit
    if candidate_distributions is None:
        candidate_distributions = [
            stats.norm, 
            stats.gamma, 
            stats.lognorm, 
            stats.weibull_min,
            #stats.pareto,
            stats.invgauss,  # Inverse Gauß (Forest) 
            stats.fisk       
        ]
    results = []
    # be sure that data are > 0 and non nan
    clean_data = data[data > 0].dropna().values

    for dist in candidate_distributions:
        try:
            # each distribution separtely 
            if dist.name == 'norm':
                params = dist.fit(clean_data)
            else:
                # Standard-Fit with start at 0
                params = dist.fit(clean_data, floc=0)

            # Log-Liklihood
            logpdf_vals = dist.logpdf(clean_data, *params)
            if np.any(~np.isfinite(logpdf_vals)):
                continue

            loglik = np.sum(logpdf_vals)
            k = len(params)
            aic = 2*k - 2*loglik

            results.append({
                'name': dist.name,
                'dist' : dist,
                'params': params,
                'aic': aic
            })

        except Exception as e:
            pass 

    if not results:
        return None

    # Sort to lowest AIC
    results.sort(key=lambda x: x['aic'])
    return results[0]

def find_best_discrete_distribution(data):
    # Speziell für Clm_Count (0, 1, 2...)
    candidate_distributions = [
        stats.poisson,
        stats.nbinom,
        stats.geom
    ]
    
    results = []
    # WICHTIG: Nullen NICHT löschen, nur NaNs entfernen
    clean_data = data.dropna().values 

    for dist in candidate_distributions:
        try:
            # Fit bei diskreten Verteilungen ist oft anders, 
            # einfacher ist es hier oft über die Momente (Mittelwert/Varianz)
            if dist.name == 'poisson':
                mu = np.mean(clean_data)
                params = (mu,)
            elif dist.name == 'nbinom':
                # Schätzung der Parameter n und p für NegBin
                m = np.mean(clean_data)
                v = np.var(clean_data)
                if v > m:
                    p = m / v
                    n = m**2 / (v - m)
                else:
                    p = 0.99
                    n = m * 100
                params = (n, p)
            
            # Wahrscheinlichkeitsfunktion (PMF) statt Dichte (PDF)
            logpmf_vals = dist.logpmf(clean_data, *params)
            loglik = np.sum(logpmf_vals)
            aic = 2*len(params) - 2*loglik

            results.append({
                'name': dist.name,
                'dist': dist,
                'aic': aic,
                'params': params
            })
        except:
            continue

    results.sort(key=lambda x: x['aic'])
    return results[0] if results else None

def plot_claim_distribution1(ax, data, column, best_fit):
    #plots histogram of target variable and fit from find_best_distribution function
    # 1. Histogramm 
    sns.histplot(data=data, x=column, bins=100, kde=False, 
                 stat="density", ax=ax, color='skyblue', alpha=0.6, label="Data")

    # 2. Calculate X-Values for fit (Min to Max of Data)
    x_min, x_max = data[column].min(), data[column].max()
    x_range = np.linspace(x_min, x_max, 200)

    # 3. PDF of best distribution 
   
    pdf_fitted = best_fit['dist'].pdf(x_range, *best_fit['params'])

    # 4. Plot fit
    ax.plot(x_range, pdf_fitted, color='red', lw=2, 
            label=f"Fit: {best_fit['name']}")

    # Design
    ax.set_title(f"Histogramm with {best_fit['name']}-Fit")
    ax.legend()
    
    return ax

def plot_claim_distribution(ax, data, column, best_fit):
    # Plots Histogramm der Zielvariable und den Fit der find_best_distribution Funktion
    if best_fit is None:
        return ax

    # 1. Histogramm (stat="density" sorgt für Normierung auf 1)
    sns.histplot(data=data, x=column, bins=50, kde=False, 
                 stat="density", ax=ax, color='skyblue', alpha=0.6, label="Data")

    # 2. Unterscheidung: Stetig (PDF) vs. Diskret (PMF)
    # Prüfen, ob die Verteilung diskret ist (z.B. Poisson, nbinom)
    is_discrete = hasattr(best_fit['dist'], 'pmf')

    if is_discrete:
        # X-Werte als ganze Zahlen (Integers) für diskrete Verteilungen
        x_min, x_max = int(data[column].min()), int(data[column].max())
        x_range = np.arange(x_min, x_max + 1)
        
        # Nutze PMF (Probability Mass Function)
        y_fitted = best_fit['dist'].pmf(x_range, *best_fit['params'])
        
        # Diskret plotten wir besser mit Markern oder als Stufenplot
        ax.plot(x_range, y_fitted, 'ro-', lw=2, markersize=5, 
                label=f"Fit (PMF): {best_fit['name']}")
    else:
        # X-Werte als fließende Spanne für stetige Verteilungen (Geld)
        x_min, x_max = data[column].min(), data[column].max()
        x_range = np.linspace(x_min, x_max, 200)
        
        # Nutze PDF (Probability Density Function)
        y_fitted = best_fit['dist'].pdf(x_range, *best_fit['params'])
        
        # Stetiger Linienplot
        ax.plot(x_range, y_fitted, color='red', lw=2, 
                label=f"Fit (PDF): {best_fit['name']}")

    # Design
    ax.set_title(f"Histogram with {best_fit['name']}-Fit")
    ax.legend()
    
    return ax

def plot_qq(ax, data, dist_obj, params, title_suffix=""):
    # Show QQ-Plot of best distribution
    # stats.probplot calculates quantiles
    stats.probplot(data, sparams=params, dist=dist_obj, plot=ax)
    
    # Title and Design 
    ax.set_title(f"Q-Q Plot: {dist_obj.name} {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax

def plot_correlation_heatmap(ax, data, title="Correlation-Heatmap"):
    """
    Berechnet die Korrelation und plottet eine Heatmap.
    """
    # Korrelationsmatrix berechnen (standardmäßig Pearson)
    corr = data.corr()

    # Eine Maske erstellen, um die obere Hälfte zu verstecken (Tri-Map)
    # Das macht die Heatmap deutlich übersichtlicher
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plotten
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True,           # Zeigt die Zahlen in den Kästchen
        fmt=".2f",            # 2 Nachkommastellen
        cmap='coolwarm',      # Blau (negativ) bis Rot (positiv)
        center=0,             # Weißpunkt bei 0 Korrelation
        ax=ax,
        cbar_kws={"shrink": .8}
    )

    ax.set_title(title)
    return ax

def plot_qq_residuals(ax, data, dist_obj, params, title_suffix=""):
    # shows QQ-Plot of residuals

    # 1. CDF of model
    cdf_vals = dist_obj.cdf(data, *params)

    # 2. Numerical stability
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

    # 3. Transformation
    residuals = stats.norm.ppf(cdf_vals)

    # 4. QQ-Plot against norm
    stats.probplot(residuals, dist="norm", plot=ax)

    # 6. Styling
    ax.set_title(f"QQ-Plot Residues ({dist_obj.name}) {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.7)

    return ax

def fit_glm(df, target_col, feature_cols, best_fit):
    #fits GLM model with best distribution

    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df[target_col].copy()
    
    dist_name = best_fit.get('name', 'default')

    try:
        # 1. LOG-NORM
        if dist_name == 'lognorm':
            st.info("🔄 Fitte log-lineares Modell (Log-Normal)...")
            y_log = np.log1p(y)
            model = sm.OLS(y_log, X)
            results = model.fit()
        # 2. GAMMA
        elif dist_name == 'gamma':
            st.info("🔄 Fitte Gamma GLM mit Log-Link...")
            family = sm.families.Gamma(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()
        # 3. INVERSE GAUSSIAN
        elif dist_name == 'invgauss':
            st.info("🔄 Fitte Inverse Gaussian GLM mit Log-Link...")
            family = sm.families.InverseGaussian(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()
        # 4. HEAVY TAILS (Proxy via InvGauss)
        elif dist_name in ['pareto', 'burr', 'fisk', 'weibull_min']:
            st.info(f"🔄 Fitte Inverse Gaussian GLM als Proxy für {dist_name}...")
            family = sm.families.InverseGaussian(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()
        # 5. NORMAL DISTRIBUTION
        else:
            st.info(f"Fitting Gauß-Model (normal distribution) for: {dist_name}...")
            model = sm.GLM(y, X, family=sm.families.Gaussian())
            results = model.fit()

        return results

    except Exception as e:
        st.error(f"GLM-Error at {dist_name}: {e}")
        st.warning("Fallback easier OLS is done.")
        return sm.OLS(y, X).fit()

#-------------------------------------------------------------------
#-------------------Streamlit UserInterface-------------------------
#-------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Actuarial Toolbox")

# 1. Load Data on the left (Sidebar)
uploaded_file = st.sidebar.file_uploader("1. Upload File", type=["csv", "rds", "xlsx", "rda"])

if uploaded_file is not None:
    st.title("Data Analysis")
    #-----------Load Data---------------------------------
    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = load_data(uploaded_file)
    #-----------Show DataFrame on Mainpage------------------
    if 'df_encoded' in st.session_state:
        st.subheader("Final Dataset (Encoded)")
        disp = st.session_state['df_encoded']
    elif 'df_final' in st.session_state:
        st.subheader("Cleaned Datenset")
        disp = st.session_state['df_final']
    elif 'df_filtered' in st.session_state:
        st.subheader("Filtered Dataset")
        disp = st.session_state['df_filtered']
    else:
        disp = st.session_state['df_raw']

    st.dataframe(disp.head(10), use_container_width=True)
    st.caption(f"Dimensions {disp.shape[0]} Rows, {disp.shape[1]} Columns")
    #----------------------------------------------------------------------
    #--------------Decide for relevant columns on sidebar------------------
    #----------------------------------------------------------------------
    st.sidebar.divider()
    all_cols = st.session_state['df_raw'].columns.tolist()
    selected_names = st.sidebar.multiselect("Choose Columns:", all_cols)
    
    if st.sidebar.button("Confirm Selection"):
        st.session_state['df_filtered'] = st.session_state['df_raw'][selected_names].copy()
        for key in ['df_final', 'df_encoded', 'show_stats', 'selected_target']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
    #----------------------------------------------------------------------
    #--------------Clean and Convert DataFrame to numeric------------------
    #----------------------------------------------------------------------
    if 'df_filtered' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Cleaning")
        current_df = st.session_state['df_filtered']
        cols_to_fix = st.sidebar.multiselect("Which columns are numbers?", options=current_df.columns.tolist())
        
        if st.sidebar.button("Clean & Convert"):
            st.session_state['df_final'] = clean_and_convert(current_df, cols_to_numeric=cols_to_fix)
            if 'df_encoded' in st.session_state: del st.session_state['df_encoded']
            st.sidebar.success("Finshed Cleaning")
            st.rerun()
    #----------------------------------------------------------------------
    #--------------Select Target-Variable & gets first stats---------------
    #----------------------------------------------------------------------
    if 'df_final' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Target Variable")
        num_cols = st.session_state['df_final'].select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            target_var = st.sidebar.selectbox("Target Variable:", options=num_cols)
            if st.sidebar.button("Chose Target Variable"):
                st.session_state['show_stats'] = True
                st.session_state['selected_target'] = target_var
                st.rerun()
    if st.session_state.get('show_stats') and 'df_final' in st.session_state:
        st.divider()
        t_col = st.session_state['selected_target']
        st.header(f"Stats for: {t_col}")
        stats_df = get_target_stats(st.session_state['df_final'], t_col)
        st.table(stats_df.style.format(precision=2))
        
        val_col = pd.to_numeric(st.session_state['df_final'][t_col], errors='coerce')
        zeros = (val_col == 0).sum()
        st.metric("Percents of Zero Values", f"{(zeros/len(val_col)*100):.2f} %", f"{zeros} von {len(val_col)} Zeilen")
    #----------------------------------------------------------------------
    #--------------Filter Target Variable----------------------------------
    #----------------------------------------------------------------------
    if 'selected_target' in st.session_state and 'df_final' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Target-Filter & Outlier")
        target = st.session_state['selected_target']
        
        p_slider = st.sidebar.slider(
            "Max-Percentil", 
            0.80, 1.00, 1.00, 0.01,
            help="Cuts values greater than percentil."
        )
        if st.sidebar.button(f"Filter on {target}"):
            df_clean, limit = filter_target_range(st.session_state['df_final'], target, percentile=p_slider)
            st.session_state['df_final'] = df_clean
            if 'df_encoded' in st.session_state: del st.session_state['df_encoded']
            st.sidebar.success(f"Filtered! Limit: {limit:,.2f}")
            st.rerun()
    #----------------------------------------------------------------------
    #--------------One-Hot Encoding of categorical columns-----------------
    #----------------------------------------------------------------------
    if 'df_final' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Encoding")
        df_for_ohe = st.session_state['df_final']
        cat_cols = df_for_ohe.select_dtypes(exclude=['number']).columns.tolist()
        if cat_cols:
            to_encode = st.sidebar.multiselect("Categorical Columns:", options=cat_cols, default=cat_cols)
            if st.sidebar.button("Encoding"):
                st.session_state['df_encoded'] = encode_categorical_data(df_for_ohe, to_encode)
                st.sidebar.success("Encoded!")
                st.rerun()
    #----------------------------------------------------------------------
    #--------------Find best distribution----------------------------------
    #----------------------------------------------------------------------
    if 'selected_target' in st.session_state and 'df_encoded' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Best Distribution")

        df_for_dist = st.session_state['df_encoded']
        target = st.session_state['selected_target']

        if st.sidebar.button("Find the best Fit", use_container_width=True):
            data_to_fit = df_for_dist[target] 
        
            best_fit = find_best_distribution1(data_to_fit)
        
            if best_fit:
                st.session_state['last_analysis'] = best_fit
            else:
                st.sidebar.error("No Distribution found")

        if st.sidebar.button("Find the best discrete Fit", use_container_width=True):
            data_to_fit = df_for_dist[target] 
        
            best_fit = find_best_discrete_distribution(data_to_fit)
        
            if best_fit:
                st.session_state['last_analysis'] = best_fit
            else:
                st.sidebar.error("No Distribution found")
    if 'last_analysis' in st.session_state:
        st.divider()
        res = st.session_state['last_analysis']
        st.write(f"**Distribution:** {res['name']} | **AIC:** {res['aic']:.2f}")
        st.write("**Parameter:**", res['params'])
    #----------------------------------------------------------------------
    #-----Plot Histogram, QQ-Plot, QQ-Plot of residuals & heatmap----------
    #----------------------------------------------------------------------
    if 'last_analysis' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Plots")
        if st.sidebar.button("Plot Histogram and best fit", use_container_width=True):
            
            st.session_state['show_plot'] = 'dist'
        
        if st.sidebar.button("Plot QQ-Plot of best fit", use_container_width=True):
            st.session_state['show_plot'] ='qq'

        if st.sidebar.button("Plot QQ-Plot of Residuals", use_container_width=True):
            st.session_state['show_plot'] ='qqres'

        if st.sidebar.button("Heatmap of correlation Matrix", use_container_width=True):
            st.session_state['show_plot'] ='heatmap'
    #--------------------Hist-Plot------------------------------------------------------
    if st.session_state.get('show_plot') == 'dist':
        fig, ax = plt.subplots()
    
        plot_claim_distribution(ax, st.session_state['df_encoded'], target, st.session_state['last_analysis'])
    
    # Show in Streamlit
        st.pyplot(fig)
    #-------------------QQ-Plot----------------------------------------------------------------
    if st.session_state.get('show_plot') == 'qq':
        
        target = st.session_state['selected_target']
        best_fit = st.session_state['last_analysis']
        fig, ax = plt.subplots()
    
        plot_qq(ax,st.session_state['df_encoded'][target], dist_obj=best_fit['dist'], params=best_fit['params'])
    
    # Show in Streamlit
        st.pyplot(fig)
    #------------------QQ-Plot of Residuals---------------------------------------------------------
    if st.session_state.get('show_plot') == 'qqres':
        
        target = st.session_state['selected_target']
        best_fit = st.session_state['last_analysis']
        fig, ax = plt.subplots()
    
        plot_qq_residuals(ax,st.session_state['df_encoded'][target], dist_obj=best_fit['dist'], params=best_fit['params'])
    
    # Show in Streamlit
        st.pyplot(fig)
    #------------------Heatmap Plot----------------------------------------
    if st.session_state.get('show_plot') == 'heatmap':
        
        target = st.session_state['selected_target']
        best_fit = st.session_state['last_analysis']
        fig, ax = plt.subplots()
    
        plot_correlation_heatmap(ax,st.session_state['df_encoded'])
    
    # Show in Streamlit
        st.pyplot(fig)

    #----------------------------------------------------------------------
    #--------------GLM of best fit-----------------------------------------
    #----------------------------------------------------------------------
    if 'last_analysis' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Perform GLM")
        target = st.session_state['selected_target']
        df_for_glm = st.session_state['df_encoded']
        available_features = [col for col in df_for_glm.columns if col != target]
        feature_cols = st.sidebar.multiselect(
            "Chose features:",
            options=available_features,
            default=available_features[:2] if len(available_features) > 1 else None
        )
        if st.sidebar.button("Führe GLM aus", use_container_width=True):
            if not feature_cols:
                st.sidebar.error("Choose min. 1 feature!")
            else:
                # Wir speichern alles Nötige in den session_state
                st.session_state['glm_trigger'] = True
                st.session_state['glm_target'] = target
                st.session_state['glm_features'] = feature_cols

    if st.session_state.get('glm_trigger'):

        with st.spinner("Berechne Modell..."):
        
            model_results = fit_glm(
                df_for_glm, 
                st.session_state['glm_target'], 
                st.session_state['glm_features'],
                st.session_state['last_analysis']
            )
        
        # Anzeige der Ergebnisse
            st.success("GLM successfully calculated!")
            st.write(model_results.summary())
else:
    st.info("Please upload file.")