import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Page Configuration ---
st.set_page_config(page_title="ISE 315 Statistics Hub", page_icon="📈", layout="wide")

st.title("📈 ISE 315 Statistics Hub")
st.write("An interactive statistical analysis tool for Engineering Statistics.")
st.markdown("##### *Your A+ in ISE 315*")
st.markdown("---")

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Navigation:", [
    "Descriptive Statistics", 
    "Inference for Two Samples (Ch 10)",
    "Simple Linear Regression (Ch 11)", 
    "Multiple Linear Regression (Ch 12)", 
    "One-Way ANOVA (Ch 13)",      # صفحة الواجبات
    "Course Resources 📚",       # صفحة المصادر
    "Grade Calculator 🧮"        # صفحة الحاسبة
])

# --- Global Helper Function for Data Input ---
def get_statistical_data(input_labels, key_prefix, is_regression=False, n_random=15):
    st.markdown("#### 📥 Data Input Method")
    method = st.radio("Choose how to input data:", 
                      ["Interactive Table (Enter values)", "Upload File (CSV/Excel)", "Generate Random Data 🎲"], 
                      key=f"method_{key_prefix}", horizontal=True)
    
    if method == "Interactive Table (Enter values)":
        st.caption("💡 Tip: Type a number, press **Enter** to go to the next row.")
        init_df = pd.DataFrame({label: [np.nan]*10 for label in input_labels})
        edited_df = st.data_editor(init_df, num_rows="dynamic", key=f"editor_{key_prefix}", use_container_width=True)
        return edited_df.dropna(how='all')

    elif method == "Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key=f"file_{key_prefix}")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                missing_cols = [col for col in input_labels if col not in df.columns]
                if missing_cols:
                    st.error(f"Error: File is missing required columns: {missing_cols}.")
                    return None
                st.success("File uploaded successfully!")
                return df[input_labels].dropna()
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None
        return None

    else:
        st.info(f"Generated {n_random} random data points for demonstration.")
        random_data = {}
        for label in input_labels:
            if "Y" in label or "Score" in label or "Values" in label:
                random_data[label] = np.random.normal(loc=75, scale=10, size=n_random).round(2)
            elif "Sample" in label or "Treatment" in label:
                random_data[label] = np.random.normal(loc=np.random.choice([70, 75, 80]), scale=8, size=n_random).round(2)
            else:
                random_data[label] = np.random.uniform(10, 50, size=n_random).round(2)
        
        df_random = pd.DataFrame(random_data)
        st.dataframe(df_random, use_container_width=True)
        return df_random

# ==========================================
# 1. Descriptive Statistics
# ==========================================
if menu == "Descriptive Statistics":
    st.header("📊 Descriptive Statistics")
    st.write("Calculate mean, median, variance, and standard deviation for a dataset.")
    
    df_data = get_statistical_data(["Values"], "desc")
    
    if df_data is not None and not df_data.empty:
        data = df_data["Values"].dropna().tolist()
        if data:
            df = pd.DataFrame(data, columns=["Values"])
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Metrics")
                
                # إعداد الحسابات
                mean_val = df["Values"].mean()
                var_val = df["Values"].var(ddof=1)
                std_val = df["Values"].std(ddof=1)
                
                # العرض مع أزرار الشرح (Popovers)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Mean (Average)", f"{mean_val:.4f}")
                    with st.popover("ℹ️ Mean"):
                        st.markdown("**Sample Mean ($\\overline{X}$)**")
                        st.markdown(r"$$\overline{X} = \frac{\sum x_i}{n}$$")
                        st.write("The arithmetic average of the data.")
                        
                with m2:
                    st.metric("Median", f"{df['Values'].median():.4f}")
                    with st.popover("ℹ️ Median"):
                        st.write("The middle value when the data is sorted. Less affected by outliers than the mean.")
                
                m3, m4 = st.columns(2)
                with m3:
                    st.metric("Variance", f"{var_val:.4f}")
                    with st.popover("ℹ️ Variance"):
                        st.markdown("**Sample Variance ($S^2$)**")
                        st.markdown(r"$$S^2 = \frac{\sum (x_i - \overline{X})^2}{n-1}$$")
                        st.write("Measures how far the numbers are spread out from their average.")
                        
                with m4:
                    st.metric("Standard Deviation", f"{std_val:.4f}")
                    with st.popover("ℹ️ Std Dev"):
                        st.markdown("**Sample Standard Deviation ($S$)**")
                        st.markdown(r"$$S = \sqrt{S^2}$$")
                        st.write("The square root of the variance.")

            with col2:
                st.subheader("Boxplot")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.boxplot(data, vert=False, patch_artist=True)
                st.pyplot(fig)

# ==========================================
# 2. Inference for Two Samples (Chapter 10)
# ==========================================
elif menu == "Inference for Two Samples (Ch 10)":
    st.header("⚖️ Inference for Two Samples")
    
    test_type = st.radio("Select Test Type:", ["Independent T-Test (Variances Unknown)", "Paired T-Test"])
    df_data = get_statistical_data(["Sample 1", "Sample 2"], "ch10")
    alpha = st.number_input("Significance Level (Alpha):", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    
    if st.button("Run Hypothesis Test", type="primary"):
        if df_data is not None and not df_data.empty:
            sample1, sample2 = df_data["Sample 1"].dropna().tolist(), df_data["Sample 2"].dropna().tolist()
            if len(sample1) > 1 and len(sample2) > 1:
                if test_type == "Independent T-Test (Variances Unknown)":
                    stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
                else:
                    if len(sample1) == len(sample2):
                        stat, p_val = stats.ttest_rel(sample1, sample2)
                    else:
                        st.error("Error: Paired T-Test requires both samples to have the same number of observations.")
                        st.stop()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Test Statistic (t-value)", f"{stat:.4f}")
                    with st.popover("ℹ️ Formula"):
                        if test_type == "Independent T-Test (Variances Unknown)":
                            st.markdown(r"$$t_0 = \frac{\overline{X}_1 - \overline{X}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}$$")
                        else:
                            st.markdown(r"$$t_0 = \frac{\overline{d}}{S_d / \sqrt{n}}$$")
                with c2:
                    st.metric("P-Value", f"{p_val:.4f}")
                    with st.popover("ℹ️ P-Value"):
                        st.write("The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct.")
                
                st.markdown("### 💡 Conclusion")
                if p_val < alpha:
                    st.success(f"**Reject the Null Hypothesis** at α = {alpha}.")
                else:
                    st.info(f"**Fail to Reject the Null Hypothesis** at α = {alpha}.")

# ==========================================
# 3. Simple Linear Regression (Chapter 11)
# ==========================================
elif menu == "Simple Linear Regression (Ch 11)":
    st.header("📈 Simple Linear Regression")
    
    df_data = get_statistical_data(["X", "Y"], "ch11", is_regression=True)
    
    if st.button("Run Regression", type="primary"):
        if df_data is not None and not df_data.empty:
            x, y = df_data["X"].dropna().tolist(), df_data["Y"].dropna().tolist()
            if len(x) == len(y) and len(x) > 1:
                X_const = sm.add_constant(x)
                model = sm.OLS(y, X_const).fit()
                
                st.info(f"**Model Equation: Y = {model.params[0]:.4f} + {model.params[1]:.4f} * X**")
                with st.popover("ℹ️ Equation Formulas"):
                    st.markdown("**Slope ($\\hat{\\beta}_1$):**")
                    st.markdown(r"$$\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}$$")
                    st.markdown("**Intercept ($\\hat{\\beta}_0$):**")
                    st.markdown(r"$$\hat{\beta}_0 = \overline{y} - \hat{\beta}_1\overline{x}$$")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("SSR", f"{model.ess:.4f}")
                    with st.popover("ℹ️ SSR"):
                        st.markdown(r"$$SS_R = \hat{\beta}_1 S_{xy}$$")
                with c2:
                    st.metric("SSE", f"{model.ssr:.4f}")
                    with st.popover("ℹ️ SSE"):
                        st.markdown(r"$$SS_E = SS_T - SS_R$$")
                with c3:
                    st.metric("SST", f"{model.centered_tss:.4f}")
                    with st.popover("ℹ️ SST"):
                        st.markdown(r"$$SS_T = \sum_{i=1}^n y_i^2 - n\overline{y}^2$$")
                with c4:
                    st.metric("R-squared", f"{model.rsquared:.4f}")
                    with st.popover("ℹ️ R²"):
                        st.markdown(r"$$R^2 = \frac{SS_R}{SS_T} = 1 - \frac{SS_E}{SS_T}$$")
                
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.scatter(x, y, color='blue', label='Data points')
                ax.plot(x, model.predict(X_const), color='red', label='Fitted line')
                ax.legend()
                st.pyplot(fig)

# ==========================================
# 4. Multiple Linear Regression (Chapter 12)
# ==========================================
elif menu == "Multiple Linear Regression (Ch 12)":
    st.header("📊 Multiple Linear Regression")
    
    df_data = get_statistical_data(["Y", "X1", "X2"], "ch12", is_regression=True)
    
    if st.button("Run Multiple Regression", type="primary"):
        if df_data is not None and not df_data.empty:
            clean_df = df_data.dropna()
            if len(clean_df) > 2:
                X_const = sm.add_constant(clean_df[['X1', 'X2']])
                model = sm.OLS(clean_df['Y'], X_const).fit()
                
                st.info(f"**Y = {model.params['const']:.4f} + {model.params['X1']:.4f}*X1 + {model.params['X2']:.4f}*X2**")
                with st.popover("ℹ️ Matrix Formula"):
                    st.markdown("**Least Squares Estimation:**")
                    st.markdown(r"$$\hat{\beta} = (X'X)^{-1}X'y$$")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("R-squared", f"{model.rsquared:.4f}")
                with c2:
                    st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
                    with st.popover("ℹ️ Adj R²"):
                        st.markdown(r"$$R_{adj}^2 = 1 - \frac{SS_E / (n-p)}{SS_T / (n-1)}$$")
                with c3:
                    st.metric("F-Statistic", f"{model.fvalue:.4f}")
                    with st.popover("ℹ️ F-Stat"):
                        st.markdown(r"$$F_0 = \frac{MSR}{MSE} = \frac{SSR / k}{SSE / (n-p)}$$")
                
                summary_df = pd.DataFrame({
                    "Coefficient": model.params,
                    "Std. Error": model.bse,
                    "t-value": model.tvalues,
                    "p-value": model.pvalues
                })
                st.dataframe(summary_df)

# ==========================================
# 5. One-Way ANOVA (Chapter 13)
# ==========================================
elif menu == "One-Way ANOVA (Ch 13)":
    st.header("📑 One-Way ANOVA")
    
    df_data = get_statistical_data(["Treatment 1", "Treatment 2", "Treatment 3"], "ch13")
    
    if st.button("Generate ANOVA Table", type="primary"):
        if df_data is not None and not df_data.empty:
            t1, t2, t3 = df_data["Treatment 1"].dropna().tolist(), df_data["Treatment 2"].dropna().tolist(), df_data["Treatment 3"].dropna().tolist()
            if t1 and t2 and t3:
                df_anova = pd.DataFrame({
                    'Score': t1 + t2 + t3,
                    'Treatment': ['T1']*len(t1) + ['T2']*len(t2) + ['T3']*len(t3)
                })
                
                model = ols('Score ~ C(Treatment)', data=df_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
                anova_table = anova_table[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
                anova_table.columns = ['Sum of Sq. (SS)', 'df', 'Mean Sq. (MS)', 'F-value', 'p-value']
                anova_table.index = ['Treatments (Between)', 'Error (Within)']
                
                st.markdown("### 📊 ANOVA Table")
                st.dataframe(anova_table)
                
                # أزرار الشرح تحت الجدول
                c1, c2, c3 = st.columns(3)
                with c1:
                    with st.popover("ℹ️ SS Formulas"):
                        st.markdown(r"**$SS_T$ (Total):**")
                        st.markdown(r"$$SS_T = \sum_{i=1}^a \sum_{j=1}^n y_{ij}^2 - \frac{y_{..}^2}{N}$$")
                        st.markdown(r"**$SS_{Treatment}$:**")
                        st.markdown(r"$$SS_{Treatment} = \sum_{i=1}^a \frac{y_{i.}^2}{n} - \frac{y_{..}^2}{N}$$")
                with c2:
                    with st.popover("ℹ️ MS Formulas"):
                        st.markdown(r"**Mean Square (MS):**")
                        st.markdown(r"$$MS_{Treatment} = \frac{SS_{Treatment}}{a-1}$$")
                        st.markdown(r"$$MSE = \frac{SS_E}{N-a}$$")
                with c3:
                    with st.popover("ℹ️ F-value Formula"):
                        st.markdown(r"$$F_0 = \frac{MS_{Treatment}}{MSE}$$")
                
                p_val = anova_table.iloc[0]['p-value']
                if p_val < 0.05:
                    st.success(f"**Reject Null Hypothesis** (p-value = {p_val:.4f}).")
                else:
                    st.info(f"**Fail to Reject Null Hypothesis** (p-value = {p_val:.4f}).")


#===============================================
elif menu == "Course Resources 📚":
    st.header("📚 Course Resources")
    # هنا حط كود الروابط (الكتاب والفورملا شيت) فقط
  #  st.link_button("Home work",'')
  #  st.link_button("📖 Textbook", "رابط_الكتاب")
  #  st.link_button("📝 Formula Sheet", "رابط_الفورملا")
    st.link_button('All Resources','https://drive.google.com/drive/folders/1bzneQWBdPb00j2qNSZgLXlQ3GYdcgyzU')
# ==========================================
# 6. Course Resources & Grade Calculator
# ==========================================
elif menu == "Grade Calculator 🧮":
    st.header("🧮 ISE 315 Grade Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Your Scores (%)")
        with st.container(border=True):
            m1_p = st.number_input("First Exam (%)", 0.0, 100.0, )
            m2_p = st.number_input("Second Exam (%)", 0.0, 100.0, 0)
            final_p = st.number_input("Final Exam (%)", 0.0, 100.0, 0)
            hw_p = st.number_input("HW Raw Score (%)", 0.0, 100.0, 0)
            att_p = st.number_input("Attendance Raw Score (%)", 0.0, 100.0, 0)
            curve = st.slider("Normalization Impact (%)", -10, 10, 0)
    with col2:
        st.subheader("📊 Your Performance Dashboard")
        
        # حساب المجموع الكلي
        major_total = (m1_p * 0.28) + (m2_p * 0.28) + (final_p * 0.30)
        att_hw_total = ((hw_p * 0.10) + (att_p * 0.04)) * (1 + curve/100)
        total = major_total + att_hw_total
        
        # نظام القريدات مع تفاصيل دقيقة
        def get_grade_details(s):
            if s >= 95: return "A+", "Excellent - Keep it up!", "green"
            if s >= 86: return "A", "Very Good - Great work!", "green"
            if s >= 81: return "B+", "Good - Solid performance!", "blue"
            if s >= 76: return "B", "Above Average - Doing well!", "blue"
            if s >= 71: return "C+", "Average - Stay focused!", "orange"
            if s >= 65: return "C", "Pass - Needs more effort!", "red"
            return "F", "Fail - Please contact your professor!", "red"

        grade, feedback, color = get_grade_details(total)
        
        # العرض البصري الجميل
       # التعديل: استخدام ألوان متوافقة مع الـ Dark Mode
        # استخدمنا ألوان متباينة (Contrast) مع خلفية داكنة خفيفة
     # يجب أن ينتهي الكود بهذا السطر تحديداً:
  # انسخ هذا الجزء بالكامل واستبدل أي شيء مشابه له في ملفك
        st.markdown(f"""
        <div style="background-color: #262730; padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #4a4a4a;">
            <p style="color: #ffffff; font-size: 18px; margin-bottom: 5px;">Total Score</p>
            <h1 style="color: {color}; font-size: 45px; margin: 0;">{total:.1f} / 100</h1>
            <h2 style="color: #ffffff; margin-top: 10px;">Grade: {grade}</h2>
            <p style="color: #cccccc; font-style: italic; margin-top: 10px;">{feedback}</p>
        </div>
        """, unsafe_allow_html=True)
        # شريط تقدم (Progress Bar)
        st.progress(min(total/100, 1.0))
        
        st.write("---")
        # الـ Z-Score الاختياري
      
# Footer / Credits
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "Developed with ❤️ by <a href='https://twitter.com/yasserqbs' target='_blank' style='text-decoration: none; color: #1DA1F2; font-weight: bold;'>Yasser (@yasserqbs)</a>"
    "</div>", 
    unsafe_allow_html=True
)
