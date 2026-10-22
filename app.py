from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Modelling", layout="wide")

OUTPUTS = Path(__file__).parent / "outputs"

# Load Saved Results
results = pd.read_csv(OUTPUTS / "model_results.csv")
coefficients = pd.read_csv(OUTPUTS / "logistic_coefficients.csv")
importance = pd.read_csv(OUTPUTS / "lightgbm_importance.csv")
predictions = pd.read_csv(OUTPUTS / "test_predictions.csv")
shap_global = pd.read_csv(OUTPUTS / "lightgbm_shap_global.csv")
shap_local = pd.read_csv(OUTPUTS / "lightgbm_shap_applicant.csv")

st.title("Credit Risk Modelling")
st.caption(
    "An end-to-end credit risk modelling project comparing Logistic Regression, "
    "LightGBM, and a Neural Network. The project covers data preparation, "
    "model development, validation, interpretation, and model comparison."
)
st.markdown("---")

# Navigation Sidebar
page = st.sidebar.radio(
    "Navigation Workspace",
    ["Overview","Model Comparison","Explainability","Conclusion"]
)

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "The objective of this project was to build a small credit-risk modelling workflow "
        "using the Home Credit Default Risk dataset. I compare three different modelling "
        "approaches using the same engineered features and Train/Test population."
    )
    
    st.write("")  

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="background-color:#46627F; padding:25px; border-radius:14px; border-bottom:6px solid #B8CAD8;">
                <p style="margin:0; font-size:14px; color:#EAF0F5; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">Total Applications</p>
                <h2 style="margin:8px 0 0 0; font-size:34px; color:#FFFFFF; font-weight:700;">307,511</h2>
                <p style="margin:5px 0 0 0; font-size:12px; color:#D7E2EA;">Full Dataset Volume</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            """
            <div style="background-color:#607D78; padding:25px; border-radius:14px; border-bottom:6px solid #BFD0C8;">
                <p style="margin:0; font-size:14px; color:#EAF2EF; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">Target Default Rate</p>
                <h2 style="margin:8px 0 0 0; font-size:34px; color:#FFFFFF; font-weight:700;">8.07%</h2>
                <p style="margin:5px 0 0 0; font-size:12px; color:#DDE8E3;">Imbalanced Target Risk</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            """
            <div style="background-color:#6B7185; padding:25px; border-radius:14px; border-bottom:6px solid #C8CDD8;">
                <p style="margin:0; font-size:14px; color:#EFF1F5; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">Evaluated Models</p>
                <h2 style="margin:8px 0 0 0; font-size:34px; color:#FFFFFF; font-weight:700;">3 Approaches</h2>
                <p style="margin:5px 0 0 0; font-size:12px; color:#E0E3EA;">Parallel Risk Pipelines</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    st.subheader("Technical Pipeline Approach")
    st.write("The sequential architecture tracking data assets downstream into performance validation runs:")
    
    st.write("")

    st.markdown(
        """
        <div style="background-color:#46627F; padding:20px; border-radius:10px; text-align:center; border-left:5px solid #B8CAD8; margin-bottom:10px;">
            <strong style="color:#FFFFFF; font-size:16px;">1. Data Engineering Pipeline</strong><br>
            <span style="font-size:13px; color:#EAF0F5; font-weight:500;">Data Ingestion &amp; Feature Engineering Integration</span>
        </div>
        <div style="text-align:center; color:#708090; font-size:22px; font-weight:bold; margin:10px 0;">↓</div>
        """,
        unsafe_allow_html=True
    )
    
    
    st.markdown(
        """
        <div style="background-color:#607D78; padding:20px; border-radius:10px; text-align:center; border-left:5px solid #BFD0C8; margin-bottom:10px;">
            <strong style="color:#FFFFFF; font-size:16px;">2. Data Exploration Phase</strong><br>
            <span style="font-size:13px; color:#EAF2EF; font-weight:500;">Interactive Analysis via exploration.ipynb</span>
        </div>
        <div style="text-align:center; color:#708090; font-size:22px; font-weight:bold; margin:10px 0;">↓</div>
        """,
        unsafe_allow_html=True
    )
    
    

    
    st.markdown(
        """
        <p style="color:#243546; font-weight:bold; font-size:15px; margin-bottom:12px; text-align:center; text-transform:uppercase; letter-spacing:0.5px;">
            3. Parallel Model Execution Branch
        </p>
        """,
        unsafe_allow_html=True
    )
    
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            """
            <div style="background-color:#46627F; padding:18px; border-radius:8px; text-align:center; border-bottom:4px solid #B8CAD8; min-height:85px;">
                <span style="color:#FFFFFF; font-weight:600; font-size:15px;">Logistic Regression</span><br>
                <span style="font-size:12px; color:#EAF0F5; opacity:0.9;">Baseline Model</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            """
            <div style="background-color:#607D78; padding:18px; border-radius:8px; text-align:center; border-bottom:4px solid #BFD0C8; min-height:85px;">
                <span style="color:#FFFFFF; font-weight:600; font-size:15px;">LightGBM Tree Model</span><br>
                <span style="font-size:12px; color:#EAF2EF; opacity:0.9;">Challenger Model 1</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            """
            <div style="background-color:#6B7185; padding:18px; border-radius:8px; text-align:center; border-bottom:4px solid #C8CDD8; min-height:85px;">
                <span style="color:#FFFFFF; font-weight:600; font-size:15px;">Neural Network</span><br>
                <span style="font-size:12px; color:#EFF1F5; opacity:0.9;">Challenger Model 2</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown(
        """
        <div style="text-align:center; color:#708090; font-size:22px; font-weight:bold; margin:10px 0;">↓</div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="background-color:#6B7185; padding:20px; border-radius:10px; text-align:center; border-left:5px solid #C8CDD8;">
            <strong style="color:#FFFFFF; font-size:16px;">4. Concluding Diagnostics Workspace</strong><br>
            <span style="font-size:13px; color:#EFF1F5; font-weight:500;">Model Validation, Gini Matrix Comparisons &amp; Final Production Deployment Decision</span>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Model Comparison":
    st.header("Model Comparison")

    st.write(
        "All three models use the same engineered dataset and Test population. "
        "This allows the modelling approaches to be compared on broadly the same "
        "underlying information."
    )

    st.subheader("Performance Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    st.write("")

    st.subheader("Test Performance")

    col1, col2, col3 = st.columns(3)

    lr = results[
        results["Model"] == "Logistic Regression"
    ].iloc[0]

    lgbm = results[
        results["Model"] == "LightGBM"
    ].iloc[0]

    nn = results[
        results["Model"] == "Neural Network"
    ].iloc[0]


    with col1:
        st.markdown(
            f"""
            <div style="background-color: #46627F; padding: 22px; border-radius: 10px; text-align: center; border-bottom: 5px solid #B8CAD8;">
                <div style="color:#EAF0F5; font-size:15px; font-weight:600;">
                    Logistic Regression
                </div>
                <div style="color:white; font-size:30px; font-weight:700; margin-top:10px;">
                    {lr["Test AUC"]:.3f}
                </div>
                <div style="color:#DCE8F2; font-size:12px; margin-top:4px;">
                    Test AUC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:
        st.markdown(
            f"""
            <div style="background-color: #607D78; padding: 22px; border-radius: 10px; text-align: center; border-bottom: 5px solid #BFD0C8;">
                <div style="color:#EAF2EF; font-size:15px; font-weight:600;">
                    LightGBM
                </div>
                <div style="color:white; font-size:30px; font-weight:700; margin-top:10px;">
                    {lgbm["Test AUC"]:.3f}
                </div>
                <div style="color:#DDE8E3; font-size:12px; margin-top:4px;">
                    Test AUC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:
        st.markdown(
            f"""
            <div style="background-color: #46627F; padding: 22px; border-radius: 10px; text-align: center; border-bottom: 5px solid #B8CAD8;">
                <div style="color:#EAF0F5; font-size:15px; font-weight:600;">
                    Neural Network
                </div>
                <div style="color:white; font-size:30px; font-weight:700; margin-top:10px;">
                    {nn["Test AUC"]:.3f}
                </div>
                <div style="color:#DCE8F2; font-size:12px; margin-top:4px;">
                    Test AUC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")
    st.write("")


    st.subheader("Results Summary")

    st.markdown(
        """
        **Logistic Regression** provides the most stable performance and acts as the main benchmark.

        **LightGBM** achieves the highest Test AUC and KS and the lowest Brier score, giving the strongest overall predictive performance. Its Train performance  is noticeably higher than its Test performance, suggesting some overfitting.
        
        **Neural Network** improves on the baseline Logistic Regression but does not outperform LightGBM in the current experiment.
        """
    )

elif page == "Explainability":

    import matplotlib.pyplot as plt

    st.header("Model Explainability")

    st.write(
        "The models have different levels of interpretability. "
        "For Logistic Regression I can inspect the fitted coefficients directly. "
        "For LightGBM I use feature importance and SHAP."
    )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    st.subheader("1. Logistic Regression")

    st.write(
        "Each Logistic Regression coefficient shows the direction and size of a "
        "feature's relationship with the modelled log-odds of Default, holding the "
        "other variables constant. Positive coefficients increase the predicted "
        "Default odds and negative coefficients reduce them."
    )

    top_coefficients = (
        coefficients
        .head(12)
        .sort_values("coefficient")
    )

    colours = [
        "#B55252" if value > 0 else "#46627F"
        for value in top_coefficients["coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.barh(
        top_coefficients["feature"],
        top_coefficients["coefficient"],
        color=colours
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel("Coefficient")
    ax.set_title("Largest Logistic Regression Coefficients")

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.caption(
        "Red = higher Default odds · Blue = lower Default odds"
    )

    st.write(
        "The external credit scores appear among the stronger risk-reducing "
        "features, while credit and previous payment information also contribute "
        "to the model."
    )


    st.divider()


    # ========================================================
    # LIGHTGBM
    # ========================================================

    st.subheader("2. LightGBM")

    st.write(
        "LightGBM is not directly interpretable in the same way as Logistic Regression. "
        "I therefore use two views of the model. Feature importance shows which variables "
        "are used most by the trees, while SHAP estimates how features contribute to the "
        "model output across the sample."
    )

    col1, col2 = st.columns(2)


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    with col1:

        st.markdown("#### Feature Importance")

        st.write(
            "This ranks the variables according to how heavily they are used "
            "by the fitted tree model."
        )

        top_importance = (
            importance
            .head(12)
            .sort_values("importance")
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        ax.barh(
            top_importance["feature"],
            top_importance["importance"],
            color="#607D78"
        )

        ax.set_xlabel("Importance")
        ax.set_title("LightGBM Feature Importance")

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )


    # ========================================================
    # GLOBAL DIRECTIONAL SHAP
    # ========================================================

    with col2:

        st.markdown("#### SHAP Direction")

        st.write(
            "This shows the average direction of each feature's SHAP contribution "
            "across the sampled applicants."
        )

        top_shap = (
            shap_global
            .head(12)
            .sort_values("mean_shap")
        )

        colours = [
            "#B55252" if value > 0 else "#46627F"
            for value in top_shap["mean_shap"]
        ]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        ax.barh(
            top_shap["feature"],
            top_shap["mean_shap"],
            color=colours
        )

        ax.axvline(
            0,
            color="black",
            linewidth=1
        )

        ax.set_xlabel("Average SHAP Contribution")
        ax.set_title("Global SHAP Direction")

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )


    st.caption(
        "SHAP: Red = pushes predictions towards higher Default risk · "
        "Blue = pushes predictions towards lower Default risk"
    )

    st.write(
        "The external credit scores are important in both Logistic Regression "
        "and LightGBM, which gives some consistency between the two model results."
    )


    st.divider()


    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    st.subheader("3. Neural Network")

    st.write(
        "The Neural Network is harder to interpret directly because its prediction "
        "is built through several layers of weights and nonlinear transformations. "
        "Methods such as SHAP can also be used for neural networks, but I left this "
        "out of this project and use the Neural Network mainly as a performance challenger."
    )
elif page == "Conclusion":

    st.header("Conclusion")

    st.write(
        "The three models provide different trade-offs between predictive "
        "performance, complexity and interpretability."
    )

    col1, col2, col3 = st.columns(3)


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    with col1:

        with st.container(border=True):

            st.markdown(
                """
                <div style="background-color:#46627F; padding:14px; border-radius:8px; margin-bottom:14px;">
                    <div style="color:white; font-size:20px; font-weight:700; text-align:center;">
                        Logistic Regression
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("**Strengths**")

            st.markdown(
                """
                - Directly interpretable coefficients
                - Consistent performance
                - Simple statistical benchmark
                """
            )

            st.markdown("**Limitations**")

            st.markdown(
                """
                - Lowest predictive performance
                - Linear structure in log-odds
                """
            )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    with col2:

        with st.container(border=True):

            st.markdown(
                """
                <div style="background-color:#607D78; padding:14px; border-radius:8px; margin-bottom:14px;">
                    <div style="color:white; font-size:20px; font-weight:700; text-align:center;">
                        LightGBM
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("**Strengths**")

            st.markdown(
                """
                - Strongest overall performance
                - Captures nonlinear relationships
                - Feature importance and SHAP available
                """
            )

            st.markdown("**Limitations**")

            st.markdown(
                """
                - More overfitting
                - Less directly interpretable
                """
            )


    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    with col3:

        with st.container(border=True):

            st.markdown(
                """
                <div style="background-color:#46627F; padding:14px; border-radius:8px; margin-bottom:14px;">
                    <div style="color:white; font-size:20px; font-weight:700; text-align:center;">
                        Neural Network
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("**Strengths**")

            st.markdown(
                """
                - Flexible nonlinear model
                - Performs slightly better than Logistic Regression
                - Useful alternative challenger
                """
            )

            st.markdown("**Limitations**")

            st.markdown(
                """
                - Does not outperform LightGBM
                - Hardest to interpret
                """
            )


    st.write("")

    # ========================================================
    # CONCLUSION
    # ========================================================

    st.subheader("Conclusion")

    st.markdown(
        """
        **LightGBM** performs best overall, with the strongest AUC and KS and
        the lowest Brier score. It also shows more overfitting compared to other models, so more work is needed to epxlore this.

        **Logistic Regression** still wins for overall simplicity and Interpretability, which could potentially improve performance if more work on feature engineering is done using techniqeus such as DFS.

        **Neural Network** performs slightly better than Logistic Regression,
        but does not outperform LightGBM despite being more complex, although this was a simple implementation, where a more thorough model selection could potentially outperfrom both models.
        """
    )


    # ========================================================
    # FURTHER WORK
    # ========================================================

    st.subheader("Further Work")

    st.markdown(
        """
        - Cross-validation
        - Regularisation
        - Calibration testing
        - Begin workign on LGD and EAD
        - Stability testing and model monitoring
        """
    )