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
    # and info about dropped columns as reference for GLM
    if not columns_to_encode: 
        return df, {}
    
    df_processed = df.copy()
    dropped_info = {}

    for col in columns_to_encode:
        original_categories = set(df_processed[col].unique().astype(str))
        df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=True, dtype=int)
        
        current_cols = {c.replace(f"{col}_", "") for c in df_processed.columns if c.startswith(f"{col}_")}
        dropped = list(original_categories - current_cols)
        if dropped:
            dropped_info[col] = dropped[0]

    return df_processed, dropped_info

def get_target_stats(df, target_col):
    # input DataFrame & target column
    # returns first statistical values of target 
    vals = df[target_col]
    pos_vals = vals[vals > 0]
    metrics = ["Count", "Mean", "Median", "Std. Deviation", "Min", "Max", "95% Percentile", "99% Percentile"]
    
    stats_data = {
        "Metric": metrics,
        "Total (incl. 0)": [
            len(vals), vals.mean(), vals.median(), vals.std(), vals.min(), vals.max(),
            vals.quantile(0.95), vals.quantile(0.99)  
        ],
        "Values > 0": [
            len(pos_vals), pos_vals.mean(), pos_vals.median(), pos_vals.std(), pos_vals.min(), pos_vals.max(),
            pos_vals.quantile(0.95), pos_vals.quantile(0.99)  
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

def plot_qq(ax, data, dist_obj, params, title_suffix=""):
    # Show QQ-Plot of best distribution
    # stats.probplot calculates quantiles
    stats.probplot(data, sparams=params, dist=dist_obj, plot=ax)
    
    # Title and Design 
    ax.set_title(f"Q-Q Plot: {dist_obj.name} {title_suffix}")
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

def fit_glm1(df, target_col, feature_cols, best_fit):
    #fits GLM model with best distribution

    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df[target_col].copy()
    
    dist_name = best_fit.get('name', 'default')

    try:
        # 1. LOG-NORM
        if dist_name == 'lognorm':
            st.info("Fitting log-linear model (Log-Normal)...")
            y_log = np.log1p(y)
            model = sm.OLS(y_log, X)
            results = model.fit()
        # 2. GAMMA
        elif dist_name == 'gamma':
            st.info("Fitting Gamma GLM with Log-Link...")
            family = sm.families.Gamma(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()
        # 3. INVERSE GAUSSIAN
        elif dist_name == 'invgauss':
            st.info("Fitting Inverse Gaussian GLM with Log-Link...")
            family = sm.families.InverseGaussian(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family)
            results = model.fit()
        # 4. HEAVY TAILS (Proxy via InvGauss)
        elif dist_name in ['pareto', 'burr', 'fisk', 'weibull_min']:
            st.info(f"Fitting Inverse Gaussian GLM als Proxy für {dist_name}...")
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

def fit_glm(df, target_col, feature_cols, best_fit, exposure_col=None):
    # Daten vorbereiten
    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df[target_col].copy()
    
    # Offset berechnen: In GLMs mit Log-Link nutzt man log(Exposure)
    offset = None
    if exposure_col and exposure_col in df.columns:
        # Sicherstellen, dass keine 0 enthalten ist (log(0) ist nicht definiert)
        offset = np.log(df[exposure_col].clip(lower=1e-6))
    
    dist_name = best_fit.get('name', 'default')

    try:
        # 1. LOG-NORM (Spezialfall via OLS)
        if dist_name == 'lognorm':
            st.info("🔄 Fitting log-linear model (Log-Normal)...")
            y_log = np.log1p(y)
            # Mathematisch: log(y) - log(E) = X*beta
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

        # 4. HEAVY TAILS (Proxy via InvGauss)
        elif dist_name in ['pareto', 'burr', 'fisk', 'weibull_min']:
            st.info(f"🔄 Fitting Inverse Gaussian GLM as Proxy for {dist_name}...")
            family = sm.families.InverseGaussian(link=sm.families.links.Log())
            model = sm.GLM(y, X, family=family, offset=offset)
            results = model.fit()

        # 5. NORMAL DISTRIBUTION
        else:
            st.info(f"🔄 Fitting Gauß-Model (normal distribution) for: {dist_name}...")
            # Achtung: Gaussian nutzt Standard-Identity-Link. 
            # Wenn Offset gewünscht, sollte man auch hier über link=Log() nachdenken.
            model = sm.GLM(y, X, family=sm.families.Gaussian(), offset=offset)
            results = model.fit()

        return results

    except Exception as e:
        st.error(f"GLM-Error at {dist_name}: {e}")
        st.warning("Fallback easier OLS is done.")
        return sm.OLS(y, X).fit()

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

def scale_columns(df, columns, divisor):
    # scales columns by dividing by divisor
    if not columns:
        return df
    df_new = df.copy()
    
    for col in columns:
        if col in df_new.columns:
            # check if numeric
            if pd.api.types.is_numeric_dtype(df_new[col]):
                df_new[col] = df_new[col] / divisor
            else:
                st.warning(f"Column '{col}' is not numeric.")     
    return df_new

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
    elif 'df_scaled' in st.session_state:
        st.subheader("scaled Datenset")
        disp = st.session_state['df_scaled']
    elif 'df_final' in st.session_state:
        st.subheader("Cleaned Datenset")
        disp = st.session_state['df_final']
    elif 'df_filtered' in st.session_state:
        st.subheader("Filtered Dataset")
        disp = st.session_state['df_filtered']
    else:
        disp = st.session_state['df_raw']

    st.dataframe(disp.head(10), width='stretch')
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
    #--------------eventually divide column by x---------------------------
    #----------------------------------------------------------------------
    if 'df_final' in st.session_state:
        if 'df_scaled' not in st.session_state:
            st.session_state['df_scaled'] = st.session_state['df_final'].copy()
        
        st.sidebar.divider()
        st.sidebar.subheader("Divide Columns by")
        div_cols = st.session_state['df_final'].columns.tolist()
    
        if div_cols:
            selected_div_cols = st.sidebar.multiselect("Select columns to scale", 
                options=div_cols)

            divisor = st.sidebar.number_input(
                "Enter Divisor", 
                min_value=1.0, 
                value=1000.0, 
                step=10.0,
                format="%.1f"
            )
            btn_col1, btn_col2 = st.sidebar.columns(2)
            if btn_col1.button("Apply Scale"):
                if selected_div_cols:
                    # scale from actual df_scaled
                    st.session_state['df_scaled'] = scale_columns(
                        st.session_state['df_scaled'], 
                        selected_div_cols, 
                        divisor
                    )

                    if 'df_encoded' in st.session_state:
                        del st.session_state['df_encoded']
                    
                    st.sidebar.success(f"Divided by {divisor}")
                    st.rerun()
                else:
                    st.sidebar.error("Please select columns!")
        
            if btn_col2.button("Reset Scaling"):
                st.session_state['df_scaled'] = st.session_state['df_final'].copy()
                if 'df_encoded' in st.session_state:
                    del st.session_state['df_encoded']
                st.rerun()
    #----------------------------------------------------------------------
    #--------------Select Target-Variable & gets first stats---------------
    #----------------------------------------------------------------------
    if 'df_scaled' in st.session_state:
        if 'df_encoded' not in st.session_state:
            st.session_state['df_encoded'] = st.session_state['df_scaled']
        st.sidebar.divider()
        st.sidebar.subheader("Target Variable")
        num_cols = st.session_state['df_scaled'].select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            target_var = st.sidebar.selectbox("Target Variable:", options=num_cols)
            if st.sidebar.button("Chose Target Variable"):
                st.session_state['show_stats'] = True
                st.session_state['selected_target'] = target_var
                st.rerun()
    if st.session_state.get('show_stats') and 'df_scaled' in st.session_state:
        st.divider()
        t_col = st.session_state['selected_target']
        st.header(f"Stats for: {t_col}")
        stats_df = get_target_stats(st.session_state['df_scaled'], t_col)
        st.table(stats_df.style.format(precision=2))
        
        val_col = pd.to_numeric(st.session_state['df_scaled'][t_col], errors='coerce')
        zeros = (val_col == 0).sum()
        st.metric("Percents of Zero Values", f"{(zeros/len(val_col)*100):.2f} %", f"{zeros} of {len(val_col)} Rows")
    #----------------------------------------------------------------------
    #--------------Filter Target Variable----------------------------------
    #----------------------------------------------------------------------
    if 'selected_target' in st.session_state and 'df_scaled' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Target-Filter & Outlier")
        target = st.session_state['selected_target']
        
        p_slider = st.sidebar.slider(
            "Max-Percentil", 
            0.80, 1.00, 1.00, 0.01,
            help="Cuts values greater than percentil."
        )
        if st.sidebar.button(f"Filter on {target}"):
            df_clean, limit = filter_target_range(st.session_state['df_scaled'], target, percentile=p_slider)
            st.session_state['df_scaled'] = df_clean
            if 'df_encoded' in st.session_state: del st.session_state['df_encoded']
            st.sidebar.success(f"Filtered! Limit: {limit:,.2f}")
            st.rerun()
    #----------------------------------------------------------------------
    #--------------One-Hot Encoding of categorical columns-----------------
    #----------------------------------------------------------------------
    
    if 'df_scaled' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Encoding")
        df_for_ohe = st.session_state['df_scaled']
        cat_cols = df_for_ohe.select_dtypes(exclude=['number']).columns.tolist()
    
        if cat_cols:
            to_encode = st.sidebar.multiselect("Categorical Columns:", options=cat_cols, default=cat_cols)
        
            if st.sidebar.button("Encoding"):
                # return encoded df and info about dropped column
                df_enc, info = encode_categorical_data(df_for_ohe, to_encode)
            
                st.session_state['df_encoded'] = df_enc
                st.session_state['encoding_info'] = info  
            
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

        if st.sidebar.button("Find the best Fit", width='stretch'):
            data_to_fit = df_for_dist[target] 
        
            best_fit = find_best_distribution(data_to_fit)
        
            if best_fit:
                st.session_state['last_analysis'] = best_fit
            else:
                st.sidebar.error("No Distribution found")

        if st.sidebar.button("Find the best discrete Fit", width='stretch'):
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
    #-Plot Histogram, QQ-Plot, QQ-Plot of residuals, heatmap, log-hist-----
    #----------------------------------------------------------------------
    if 'last_analysis' in st.session_state:
        st.sidebar.divider()
        st.sidebar.subheader("Plots")
        if st.sidebar.button("Plot Histogram and best fit", width='stretch'):
            st.session_state['show_plot'] = 'dist'
        
        if st.sidebar.button("Log-Distribution",width='stretch'):
            st.session_state['show_plot'] = "logdist"

        if st.sidebar.button("Plot QQ-Plot of best fit", width='stretch'):
            st.session_state['show_plot'] ='qq'

        if st.sidebar.button("Plot QQ-Plot of Residuals", width='stretch'):
            st.session_state['show_plot'] ='qqres'

        if st.sidebar.button("Heatmap of correlation Matrix", width='stretch'):
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
    #--------------------------Log-Dist Plot-------------------------------
    if st.session_state.get('show_plot') == 'logdist':

        fig,ax =plt.subplots()

        plot_log_histogram(ax,st.session_state['df_encoded'],target,st.session_state['last_analysis'])

        st.pyplot(fig)

    #-------------------show which variable are dropped and are now reference for GLM-----------------
    if 'encoding_info' in st.session_state:
        st.info("### Analysis of reference variable (Baseline)")
        cols = st.columns(len(st.session_state['encoding_info']))
    
        for idx, (col_name, ref_val) in enumerate(st.session_state['encoding_info'].items()):
            with cols[idx]:
                st.metric(label=f"Reference for {col_name}", value=ref_val)
                st.caption(f"All Coefficients for {col_name} are relative to **{ref_val}**.")

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
        if st.sidebar.button("Perform GLM", width='stretch'):
            if not feature_cols:
                st.sidebar.error("Choose min. 1 feature!")
            else:
                # store in session_state
                st.session_state['glm_trigger'] = True
                st.session_state['glm_target'] = target
                st.session_state['glm_features'] = feature_cols

    if st.session_state.get('glm_trigger'):

        with st.spinner("Calculate model..."):
        
            model_results = fit_glm(
                df_for_glm, 
                st.session_state['glm_target'], 
                st.session_state['glm_features'],
                st.session_state['last_analysis']
            )
        
        # show results of GLM
            st.success("GLM successfully calculated!")
            st.write(model_results.summary())
    
else:
    st.info("Please upload file.")