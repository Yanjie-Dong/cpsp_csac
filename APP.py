#!/usr/bin/env python
# coding: utf-8

# 工作环境
import os
import warnings
import joblib
import streamlit as st
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)

# 加载模型和数据
model = joblib.load("simple_model.pkl")
min_max_params = joblib.load("min_max_params_app.pkl")
selected_features = joblib.load("selected_features_app.pkl")  # 25

feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]

# 定义每个特征的输入范围
feature_ranges = {
    "Subacute pain NRS score at POD30": {"min": 0, "max": 10, "step": 1},
    "The 10th percentile of postoperative NRS score": {"min": 0, "max": 10, "step": 1},
    "Postoperative NRS CWT (coeff=2, width=2, scales=(2/5/10/20))": {"min": 0.0, "max": 1.0, "step": 0.1},
    "Rehabilitation feeling at POD30": {"min": 0, "max": 10, "step": 1},
    "Surgical month": {"min": 1, "max": 12, "step": 1},
    "Consumption of intraoperative opioid": {"min": 0.0, "max": None, "step": 0.1},
    "Intraoperative crystalloid": {"min": 0, "max": None, "step": 1},
    "Treponema pallidum antibody": {"min": 0.0, "max": None, "step": 0.1},
    "Preoperative pain NRS score": {"min": 0, "max": 10, "step": 1},
    "Hospitalizing expenses": {"min": 0, "max": None, "step": 1},
    "Preoperative serummagnesium": {"min": 0.0, "max": None, "step": 0.1},
    "Consumption of intraoperative sevoflurane": {"min": 0.0, "max": None, "step": 0.1},
    "Consumption of intraoperative propofol": {"min": 0.0, "max": None, "step": 0.1},
    "Preoperative fibrinogen": {"min": 0.0, "max": None, "step": 0.1}
}

# 定义哪些特征是二元的
binary_features = ["Drainage tube placement", "Open surgery", "Male gender",
                  "Abdominal surgery", 'Operation grading IV',
                  'PHQ9-Trouble in sleep', 'Junior school and below',
                  'PSQI-Feel too hot when sleep',
                  'Middle thrombus risk',
                  'Surface or limb surgery',
                  'No thrombus risk']

# 设置页面为全宽模式
st.set_page_config(layout="wide")

# 自定义CSS优化三列布局
st.markdown("""
<style>
    /* 主容器全宽 */
    .main .block-container {
        max-width: 100%;
        padding: 2rem 4rem;
    }
    
    /* 输入控件样式 */
    div.stNumberInput, div.stSelectbox {
        width: 100% !important;
    }
    
    /* 三列样式优化 */
    div[data-testid="column"] {
        padding: 0rem 1rem;
        min-width: 30%;
    }
    
    /* 标签样式优化 */
    label[data-testid="stWidgetLabel"] p {
        font-size: 14px;
        line-height: 1.4;
        margin-bottom: 0.5rem;
        word-break: break-word;
    }
    
    /* 输入框样式 */
    div[data-baseweb="input"]>div, 
    div[data-baseweb="select"]>div {
        border-radius: 4px;
        padding: 0.25rem 0.75rem;
    }
    
    /* 按钮样式 */
    div.stButton>button {
        width: 100%;
        margin-top: 2rem;
        padding: 0.5rem;
    }
    
    /* 响应式调整 */
    @media screen and (max-width: 1200px) {
        div[data-testid="column"] {
            min-width: 45%;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("CPSP Prediction")

# 创建3列，使用全宽布局
col1, col2, col3 = st.columns(3)

inputs = {}
for i, feature in enumerate(selected_features):
    current_col = i % 3  # 改为3列循环
    with [col1, col2, col3][current_col]:  # 三列上下文
        # 使用原始特征名称
        display_name = feature
        
        if feature in binary_features:
            inputs[feature] = st.selectbox(
                display_name,
                options=[0, 1],
                format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)',
                key=f"binary_{feature}"
            )
        else:
            min_val = feature_ranges[feature]["min"]
            max_val = feature_ranges[feature]["max"]
            step = feature_ranges[feature]["step"]
            
            if max_val is None:
                inputs[feature] = st.number_input(
                    display_name,
                    min_value=min_val,
                    step=step,
                    value=min_val,
                    key=f"num_{feature}"
                )
            else:
                inputs[feature] = st.number_input(
                    display_name,
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                    value=min_val,
                    key=f"num_{feature}"
                )

# 预测按钮（全宽按钮）
if st.button("Predict", key="predict_button"):
    user_input = np.array([inputs[f] for f in selected_features])
    min_vals = min_max_params["min"][selected_indices]
    max_vals = min_max_params["max"][selected_indices]
    normalized_input = (user_input - min_vals) / (max_vals - min_vals)
                            
    prediction = model.predict([normalized_input])
    predicted_proba = model.predict_proba([normalized_input])[0]

    risk_probability = predicted_proba[1]
    st.success(f"Based on this model, the output probability of CPSP risk is {risk_probability * 100:.2f}%. A predicted probability ≥12.40% (the optimal threshold determined by Youden's index) is classified as high risk for CPSP.")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))

    # 显示SHAP力图（全宽显示）
    st.subheader("Feature Impact Analysis")
    plt.figure(figsize=(20, 6))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([normalized_input], columns=selected_features), matplotlib=True)
    plt.tight_layout()
    st.pyplot(plt)
