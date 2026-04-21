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
#------------------------------------------------------------
# Functions
#------------------------------------------------------------
@st.cache_data
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
                tmp_file.write(uploaded_file.getbuffer()) 
                tmp_path = tmp_file.name
            
            try:
                result = pyreadr.read_r(tmp_path)
                
                if result:
                    obj_name = list(result.keys())[0]
                    return result[obj_name]
                else:
                    st.error("R-Data is empty.")
                    return None
            finally:
                
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    except Exception as e:
        st.error(f"Error while loading data ({extension}): {e}")
        return None
    
def get_processed_data():
    """
    ENGINE: Calculates DataFrame based on conf session_state
    """
    if 'df_raw' not in st.session_state or st.session_state.df_raw is None:
        return None

    # Start with raw data
    df = st.session_state.df_raw.copy()
    conf = st.session_state.config

    # 1. Selection of columns
    if conf['selected_cols']:
        df = df[conf['selected_cols']]

    # 2. Cleaning & Numeric Conversion
    if conf['cols_to_fix']:
        for col in conf['cols_to_fix']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)

    # 3. Scaling
    if conf['scaling_cols'] and conf['divisor'] > 1:
        for col in conf['scaling_cols']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] / conf['divisor']

    # 4. Outlier / Range Filter
    if conf['target'] and conf['target'] in df.columns:
        target = conf['target']
        df = df[df[target] > 0] # Grundregel für aktuarielle Fits
        low = df[target].quantile(conf['p_range'][0])
        high = df[target].quantile(conf['p_range'][1])
        df = df[(df[target] >= low) & (df[target] <= high)].reset_index(drop=True)

    # 5. Encoding (returns DF and Meta-Info)
    encoding_info = {}
    if conf['encode_cols']:
        for col in conf['encode_cols']:
            if col in df.columns:
                original_cats = set(df[col].unique().astype(str))
                df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)
                current_cols = {c.replace(f"{col}_", "") for c in df.columns if c.startswith(f"{col}_")}
                dropped = list(original_cats - current_cols)
                if dropped: encoding_info[col] = dropped[0]
    
    st.session_state['last_encoding_info'] = encoding_info
    return df

@st.cache_data
def find_best_distribution(data, candidate_distributions=None):
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

def plot_claim_distribution(ax, data, column, best_fit):
    # Plots Histogramm of target varible and fit of find_best_distribution function
    if best_fit is None:
        return ax

    # 1. Histogramm (stat="density" normalizes to 1)
    sns.histplot(data=data, x=column, bins=50, kde=False, 
                 stat="density", ax=ax, color='skyblue',edgecolor ='white', alpha=0.6, label="Data")

    # 2.  steadily (PDF) vs. discrete (PMF)
    # Check whether distritbution is discrete (z.B. Poisson, nbinom)
    is_discrete = hasattr(best_fit['dist'], 'pmf')

    if is_discrete:
        # X-Values as Integers for discrete distribution
        x_min, x_max = int(data[column].min()), int(data[column].max())
        x_range = np.arange(x_min, x_max + 1)
        
        # Use PMF (Probability Mass Function)
        y_fitted = best_fit['dist'].pmf(x_range, *best_fit['params'])
        
        # Discrete plot with  markers or stepplot
        ax.plot(x_range, y_fitted, 'ro-', lw=2, markersize=5, 
                label=f"Fit (PMF): {best_fit['name']}")
    else:
        #
        x_min, x_max = data[column].min(), data[column].max()
        x_range = np.linspace(x_min, x_max, 200)
        
        # Use PDF (Probability Density Function)
        y_fitted = best_fit['dist'].pdf(x_range, *best_fit['params'])
        
        # Steady plot
        ax.plot(x_range, y_fitted, color='red', lw=2, 
                label=f"Fit (PDF): {best_fit['name']}")

    # Design
    ax.set_title(f"Histogram with {best_fit['name']}-Fit")
    ax.legend()
    
    return ax

def plot_log_histogram(ax, data, column, best_fit=None):
    # 1. transform data log(x + 1)
    # only use positve values (as in find_best_distribution)
    clean_data = data[column][data[column] > 0].dropna()
    log_data = np.log1p(clean_data)
    
    # 2. plot histogram
    sns.histplot(
        log_data, 
        ax=ax, 
        stat="density", 
        bins=50, 
        color="skyblue", 
        edgecolor="white",
        kde=False  # we plot best_fit instead
    )
    
    # 3. show fit if available 
    if best_fit:
        dist = best_fit['dist']
        params = best_fit['params']
        
        # x-axis
        x_min, x_max = ax.get_xlim()
        y_range = np.linspace(x_min, x_max, 200)
        
        # back transformation to original scale: x = exp(y) - 1
        x_original = np.expm1(y_range)
        
        # PDF calculation on original scale
        pdf_original = dist.pdf(x_original, *params)
        
        # Transformation of PDF in Log-space:
        # pdf_log(y) = pdf_orig(exp(y)-1) * exp(y)
        pdf_log_transformed = pdf_original * np.exp(y_range)
        
        ax.plot(y_range, pdf_log_transformed, 'r-', lw=2, 
                label=f"Fit: {best_fit['name']}")
        ax.legend()
    
    # 4. Axis labeling 
    ax.set_title(f"Distribution of log({column} + 1)")
    ax.set_xlabel(f"log({column} + 1)")
    ax.set_ylabel("Density")
    
    return ax

def plot_qq(ax, data, dist_obj, params, title_suffix=""):
    # Show QQ-Plot of best distribution
    # stats.probplot calculates quantiles
    stats.probplot(data, sparams=params, dist=dist_obj, plot=ax)
    
    # Title and Design 
    ax.set_title(f"Q-Q Plot: {dist_obj.name} {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
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

def plot_correlation_heatmap(ax, data, title="Correlation-Heatmap"):
    #calculates correlation heatmap
    
    corr = data.corr()

    
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True,           # shows number
        fmt=".2f",           
        cmap='coolwarm',      
        center=0,             
        ax=ax,
        cbar_kws={"shrink": .8}
    )

    ax.set_title(title)
    return ax

def fit_glm(df, target_col, feature_cols, best_fit, exposure_col=None):
    # Daten vorbereiten
    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df[target_col].copy()
    
    # Calculate Offset: In GLMs with Log-Link one uses log(Exposure)
    offset = None
    if exposure_col and exposure_col in df.columns:
        # log(0) is not defined
        offset = np.log(df[exposure_col].clip(lower=1e-6))
    
    dist_name = best_fit.get('name', 'default')

    try:
        # 1. LOG-NORM (Special case via OLS)
        if dist_name == 'lognorm':
            st.info("🔄 Fitting log-linear model (Log-Normal)...")
            y_log = np.log1p(y)
            # log(y) - log(E) = X*beta
            if offset is not None:
                y_log = y_log - offset
            model = sm.OLS(y_log, X)
            results = model.fit()

        # 2. GAMMA
        elif dist_name == 'gamma':
            st.info("🔄 Fitting Gamma GLM with Log-Link...")
            family = sm.families.Gamma(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family, offset=offset)
            results = model.fit()

        # 3. INVERSE GAUSSIAN
        elif dist_name == 'invgauss':
            st.info("🔄 Fitting Inverse Gaussian GLM with Log-Link...")
            family = sm.families.InverseGaussian(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family, offset=offset)
            results = model.fit()

        # 4. HEAVY TAILS (Proxy via Gamma)
        elif dist_name in ['fisk','pareto', 'burr', 'weibull_min']:
            st.info(f"🔄 Fitting Gamma-GLM as robust Proxy for {dist_name}...")
            
            family = sm.families.Gamma(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()

        # elif dist_name in ['pareto', 'burr', 'fisk', 'weibull_min']:
        #     st.info(f"🔄 Fitting Inverse Gaussian GLM as Proxy for {dist_name}...")
        #     family = sm.families.InverseGaussian(link=sm.families.links.Log())
        #     model = sm.GLM(y, X, family=family, offset=offset)
        #     results = model.fit()

        # 5. NORMAL DISTRIBUTION
        else:
            st.info(f"🔄 Fitting Gauß-Model (normal distribution) for: {dist_name}...")
            # Attention: Gaussian uses Standard-Identity-Link. 
            model = sm.GLM(y, X, family=sm.families.Gaussian(), offset=offset)
            results = model.fit()

        return results

    except Exception as e:
        st.error(f"GLM-Error at {dist_name}: {e}")
        st.warning("Fallback easier OLS is done.")
        return sm.OLS(y, X).fit()

def get_target_stats(df, target_col):
    """
    Berechnet statistische Kennzahlen für eine Spalte im übergebenen DataFrame.
    Es wird keine interne Filterung (wie > 0) vorgenommen.
    """
    vals = pd.to_numeric(df[target_col], errors='coerce').dropna()
    
    metrics = [
        "Count", "Mean", "Median", "Std. Deviation", 
        "Min", "Max", "95% Percentile", "99% Percentile"
    ]
    
    values = [
        len(vals), 
        vals.mean(), 
        vals.median(), 
        vals.std(), 
        vals.min(), 
        vals.max(),
        vals.quantile(0.95), 
        vals.quantile(0.99)
    ]
    
    stats_data = {
        "Metric": metrics,
        "Value": values
    }
    
    return pd.DataFrame(stats_data)


st.set_page_config(layout="wide", page_title="Actuarial Toolbox Pro")

default_config = {
                'selected_cols': [],
                'cols_to_fix': [],
                'scaling_cols': [],
                'divisor': 1000.0,
                'target': None,
                'p_range': (0.0, 1.0),
                'encode_cols': [],
                'show_plot': None
            }
if 'config' not in st.session_state:
    st.session_state.config = default_config

#-------------------------------------------------------------------
# SIDEBAR (Configuration) for streamlit
#-------------------------------------------------------------------
with st.sidebar:
    st.title("Data Preparation")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "rds", "xlsx","rda"])
#-------------------------------------------------------------------
# Load data and convert to DF
#-------------------------------------------------------------------
    if uploaded_file:
        if st.session_state.get('file_name') != uploaded_file.name:
            # 1. Load new Data
            st.session_state.df_raw = load_data(uploaded_file)
            st.session_state.file_name = uploaded_file.name
        
            # 2. reset complete config if new data file is uploaded
            st.session_state.config = default_config
            # 3. also all analysis session_states
            keys_to_clear = ['last_fit', 'last_analysis', 'glm_trigger']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()


    if 'df_raw' in st.session_state:
        all_cols = st.session_state.df_raw.columns.tolist()
        st.divider()
    #-------------------------------------------------------------------
    # Select relevant columns
    #-------------------------------------------------------------------
        temp_selection = st.multiselect("Choose relevant columns:", options=all_cols, default=st.session_state.config['selected_cols'])
        # Button as "Gatekeeper"
        if st.button("Confirm columns", type="primary"):
            # now session_state is set up 
            st.session_state.config['selected_cols'] = temp_selection
        
            # reset later settings if depending column was deleted 
            if st.session_state.config['target'] not in temp_selection:
                st.session_state.config['target'] = None
            
            st.success("Auswahl übernommen!")
            st.rerun() 

        st.divider()
    #-------------------------------------------------------------------
    # Select numeric columns
    #-------------------------------------------------------------------
        temp_selection_num = st.multiselect(
            "Choose numeric columns:", 
            options=st.session_state.config['selected_cols'], 
            default=None
            )
        if st.button("Confirm numeric columns", type="primary"):
            
            st.session_state.config['cols_to_fix'] = temp_selection_num
        
            # reset later settings if depending column was deleted 
            if st.session_state.config['target'] not in temp_selection_num:
                st.session_state.config['target'] = None
            
            st.success("Selection confirmed!")
            st.rerun() 

        st.divider()
    #-------------------------------------------------------------------
    # Select categorical columns
    #-------------------------------------------------------------------
        # find potential categorical columns
        cat_cols = [c for c in st.session_state.config['selected_cols'] if c not in st.session_state.config['cols_to_fix']]
        temp_selection_enc = st.multiselect("Choose columns to One-Hot encode:", options=cat_cols, default=cat_cols)
        if st.button("Confirm columns to encode", type="primary"):
            
            st.session_state.config['encode_cols'] = temp_selection_enc
        
            st.success("Selction confirmed!")
            st.rerun() 
        
        st.divider()
    #-------------------------------------------------------------------
    # Select columns to scale and divisor
    #-------------------------------------------------------------------
        st.subheader("Scaling")
        # choose columns to scale and divisor for scaling
        temp_selection_scale = st.multiselect("Choose columns to scale:",options=all_cols, default= st.session_state.config['scaling_cols'])
        temp_selection_div = st.number_input("Divisor", value=1000.0)
        if st.button("Confirm columns to scale", type="primary"):
            
            st.session_state.config['scaling_cols'] = temp_selection_scale
            st.session_state.config['divisor'] = temp_selection_div
            
            st.success("Selection confirmed!")
            st.rerun() 

        st.divider()
    #-------------------------------------------------------------------
    # Select Target Variable
    #-------------------------------------------------------------------
        st.subheader("Target Variable")
        num_cols = st.session_state.config['selected_cols']
        target_sel = st.selectbox("Target Variable:", options=[None] + num_cols,index=0)

        if st.button("Choose target"):
            st.session_state.config['target'] = target_sel
            
            # get processed DataFrame and find the best distribution for chosen target
            df_final = get_processed_data()
            st.session_state['last_fit'] = find_best_distribution(df_final[st.session_state.config['target']])
            st.rerun()
        
        st.divider()
    #-------------------------------------------------------------------
    # Filter Target Variable
    #-------------------------------------------------------------------
        st.subheader("Filter Target Variable")
        if st.session_state.config['target']:
            # temporary variable for slider
            # starting point from config state
            temp_p_range = st.slider("Percentile Range", min_value=0.0, max_value=1.0, value=st.session_state.config['p_range'])
    
            # Button for confirmation 
            if st.button("Apply filter"):
                # update session_state of p_range
                st.session_state.config['p_range'] = temp_p_range
        
                st.success(f"Filter set to {temp_p_range}!")
                # process new DataFrame and calculate best fit for new DataFrame
                df_final = get_processed_data()
                st.session_state['last_fit'] = find_best_distribution(df_final[st.session_state.config['target']])
                
                st.rerun()

#-------------------------------------------------------------------
# MAINPAGE (Calculation & Visualization)
#-------------------------------------------------------------------

if 'df_raw' in st.session_state:
    # get final DataFrame with all constraints 
    df_final = get_processed_data()
#-------------------------------------------------------------------
# Show actual DataFrame
#-------------------------------------------------------------------    
    st.header("Actual DataFrame")
    st.dataframe(df_final.head(10), width='stretch')
    st.caption(f"Actual dimensions: {df_final.shape[0]} Rows, {df_final.shape[1]} Columns")

    #-------------------------------------------------------------------
    # Show statistics for Target and Target > 0
    #-------------------------------------------------------------------
    if st.session_state.config['target']:
        t_col = st.session_state.config['target']
        #st.header(f"Stats for target variable: {t_col}")
    
        # Raw DataFrame with values <= 0
        df_all = st.session_state.df_raw[st.session_state.config['selected_cols']].copy()
        # Comparison of raw DataFrame and final DataFrame
        stats_all = get_target_stats(df_all, t_col).set_index("Metric")
        stats_final = get_target_stats(df_final, t_col).set_index("Metric")

        # merge tabular
        comparison_df = pd.concat([stats_all, stats_final], axis=1)
        comparison_df.columns = ["Raw Data (incl. 0)", "Final Data (only > 0)"]

        # show on Mainpage
        st.header(f"Statistical Comparison of {t_col}")
        st.table(comparison_df.style.format(precision=2))

        # count where target is 0 or <0
        val_col = pd.to_numeric(df_all[t_col], errors='coerce')
        total = len(val_col)

        zeros = (val_col == 0).sum()
        negatives = (val_col < 0).sum()

        # Layout with two columns
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Zero Values (0)", value=f"{(zeros/total*100):.2f} %", delta=f"{zeros} of {total} Rows")

        with m_col2:
            st.metric(label="Negative Values (<0)", value=f"{(negatives/total*100):.2f} %", delta=f"{negatives} of {total} Rows")

        # Conclusion
        excluded_total = zeros + negatives
        st.info(f"**Total rows dropped:** {excluded_total} Rows ({ (excluded_total/total*100):.2f}%) are ignored for GLM-Fit (Target > 0).")

    #-------------------------------------------------------------------
    # Find best distribution & show plots
    #-------------------------------------------------------------------
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Distribution Fit")
            # if st.button("Find best fit", width='stretch'):
            #     with st.spinner("Berechne..."):
            #         res = find_best_distribution(df_final[st.session_state.config['target']])
            #         st.session_state['last_fit'] = res
            
            if 'last_fit' in st.session_state:
                fit = st.session_state.last_fit
                st.info(f"Modell: **{fit['name']}** \nAIC: `{fit['aic']:.2f}`")
                
                st.divider()
                if st.button("Histogramm"): st.session_state.config['show_plot'] = 'hist'
                if st.button("Log-Hist"): st.session_state.config['show_plot'] = 'loghist'
                if st.button("QQ-Plot"): st.session_state.config['show_plot'] = 'qq'
                if st.button("QQ-Residuals"): st.session_state.config['show_plot'] = 'qqres'
                if st.button("Correlation"): st.session_state.config['show_plot'] = 'corr'

        with col2:
            if st.session_state.config['show_plot']:
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_type = st.session_state.config['show_plot']
                target = st.session_state.config['target']
                
                if plot_type == 'hist':
                    plot_claim_distribution(ax, df_final, target, fit)
                elif plot_type == 'loghist':
                    plot_log_histogram(ax, df_final, target, fit)
                elif plot_type == 'qq':
                    plot_qq(ax,df_final[target], dist_obj=fit['dist'], params=fit['params'])
                elif plot_type =='qqres':
                    plot_qq_residuals(ax,df_final[target], dist_obj=fit['dist'], params=fit['params'])
                elif plot_type == 'corr':
                    plot_correlation_heatmap(ax,df_final)
                
                st.pyplot(fig)

    #-------------------------------------------------------------------
    # GLM SECTION
    #-------------------------------------------------------------------
    st.divider()
    
    st.header("GLM Modelling ")
    
    if st.session_state.config['target'] and 'last_fit' in st.session_state:
        features = st.multiselect("Features für GLM:", [c for c in df_final.columns if c != st.session_state.config['target']])
        if 'last_encoding_info' in st.session_state and st.session_state['last_encoding_info']:
            st.subheader("Encoding Info")
            #  Dictionarys to DataFrame for Visualization
            enc_df = pd.DataFrame(
                list(st.session_state['last_encoding_info'].items()), 
                columns=['Variable', 'Reference Categorie (Dropped)']
                )
            st.dataframe(enc_df, hide_index=True, width='stretch')     
        if st.button("Calculate GLM") and features:
            model_results = model_results = fit_glm(
                df_final, 
                target, 
                features,
                st.session_state['last_fit']
            )
            
            st.write(model_results.summary())

else:
    st.info("Please upload data file.")   