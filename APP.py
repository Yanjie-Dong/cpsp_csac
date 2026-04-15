#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 工作环境
import os
import warnings
current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)

import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 页面配置（宽布局） ====================
st.set_page_config(
    page_title="CPSP Prediction",
    layout="wide"
)

# ==================== CSS 美化（沿用毕业论文风格，内容适配英文） ====================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        color: #1a237e;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* 子标题样式 */
    .section-title {
        color: #37474f;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 1rem 0 0.5rem 0;
    }
    
    /* 输入卡片样式 */
    .input-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* 结果卡片样式 */
    .result-container {
        background: #f5f7fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* 预测按钮样式 */
    .predict-button {
        width: 100%;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
        background: #1a237e;
        color: white;
        font-weight: 500;
        font-size: 1rem;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    
    .predict-button:hover {
        background: #283593;
    }
    
    /* 输入框统一样式 */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* 特征标签 */
    .feature-label {
        font-weight: 500;
        color: #455a64;
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 设置 Matplotlib 的全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# ==================== 加载模型与参数（SCI版本） ====================
model = joblib.load("simple_model.pkl")
min_max_params = joblib.load("min_max_params_app.pkl")
selected_features = joblib.load("selected_features_app.pkl")  # 25

feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]

# 明确保留 SCI 版本的特征列表（确保与加载的模型一致）
selected_features = ['Subacute pain NRS score on postoperative day thirty', 
                     'Drainage tube placement', 
                     'Acute pain NRS score on postoperative day three', 
                     'Gender', 
                     'Rehabilitation NRS score on postoperative day thirty', 
                     'Open surgery', 
                     'Abdominal surgery', 
                     'Treponema pallidum antibody', 
                     'Consumption of intraoperative opioid', 
                     'Operation grading IV', 
                     'Preoperative pain NRS score', 
                     'Consumption of intraoperative propofol', 
                     'Summer season', 
                     'Preoperative PSQI score', 
                     'Surface or limb surgery']

# 定义每个特征的输入范围（SCI 版本原有）
feature_ranges = {
    "Subacute pain NRS score on postoperative day thirty": {"min": 0.0, "max": 10.0, "step": 1.0},
    "Acute pain NRS score on postoperative day three": {"min": 0.0, "max": 10.0, "step": 1.0},
    "Rehabilitation NRS score on postoperative day thirty": {"min": 0.0, "max": 10.0, "step": 1.0},
    "Consumption of intraoperative opioid": {"min": 0.0, "max": None, "step": 0.1},
    "Treponema pallidum antibody": {"min": 0.0, "max": 1.0, "step": 0.1},
    "Acute pain NRS score on postoperative day one": {"min": 0.0, "max": 10.0, "step": 1.0},
    "Preoperative pain NRS score": {"min": 0.0, "max": 10.0, "step": 1.0},
    "Hospitalizing expenses": {"min": 0.0, "max": None, "step": 0.1},
    "Consumption of intraoperative propofol": {"min": 0.0, "max": None, "step": 0.1},
    "Preoperative PSQI score": {"min": 0.0, "max": 20.0, "step": 1.0}
}

# 定义二元特征（SCI 版本原有）
binary_features = ["Drainage tube placement", "Open surgery", "Gender",
                   "Abdominal surgery", 'Operation grading IV',
                   'PHQ9-Trouble in sleep', 'Summer season',
                   'Surface or limb surgery']

# ==================== 界面布局 ====================
st.markdown('<div class="main-header">CPSP Prediction</div>', unsafe_allow_html=True)

# 两列布局
col1, col2 = st.columns(2, gap="large")

inputs = {}

# 将特征分成两列
mid_point = len(selected_features) // 2
left_features = selected_features[:mid_point]
right_features = selected_features[mid_point:]

# 左侧列
with col1:
    for feature in left_features:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        if feature in binary_features:
            inputs[feature] = st.selectbox(
                label="",
                options=[0, 1],
                format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)',
                key=f"left_{feature}",
                label_visibility="collapsed"
            )
        else:
            if feature in feature_ranges:
                range_info = feature_ranges[feature]
                min_val = range_info["min"]
                max_val = range_info["max"]
                step = range_info["step"]
                if max_val is None:
                    inputs[feature] = st.number_input(
                        label="",
                        min_value=min_val,
                        step=step,
                        value=min_val,
                        key=f"left_{feature}",
                        label_visibility="collapsed"
                    )
                else:
                    # 使用 slider 提升体验（毕业论文风格）
                    inputs[feature] = st.slider(
                        label="",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=step,
                        key=f"left_{feature}",
                        label_visibility="collapsed"
                    )
            else:
                inputs[feature] = st.number_input(
                    label="",
                    min_value=0.0,
                    step=0.1,
                    value=0.0,
                    key=f"left_{feature}",
                    label_visibility="collapsed"
                )

# 右侧列
with col2:
    for feature in right_features:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        if feature in binary_features:
            inputs[feature] = st.selectbox(
                label="",
                options=[0, 1],
                format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)',
                key=f"right_{feature}",
                label_visibility="collapsed"
            )
        else:
            if feature in feature_ranges:
                range_info = feature_ranges[feature]
                min_val = range_info["min"]
                max_val = range_info["max"]
                step = range_info["step"]
                if max_val is None:
                    inputs[feature] = st.number_input(
                        label="",
                        min_value=min_val,
                        step=step,
                        value=min_val,
                        key=f"right_{feature}",
                        label_visibility="collapsed"
                    )
                else:
                    inputs[feature] = st.slider(
                        label="",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=step,
                        key=f"right_{feature}",
                        label_visibility="collapsed"
                    )
            else:
                inputs[feature] = st.number_input(
                    label="",
                    min_value=0.0,
                    step=0.1,
                    value=0.0,
                    key=f"right_{feature}",
                    label_visibility="collapsed"
                )

# ==================== 预测按钮 ====================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("Predict", use_container_width=True, type="primary"):
        # 构建输入向量
        user_input = np.array([inputs[f] for f in selected_features])
        min_vals = min_max_params["min"][selected_indices]
        max_vals = min_max_params["max"][selected_indices]
        normalized_input = (user_input - min_vals) / (max_vals - min_vals)
        
        # 预测概率
        prediction = model.predict([normalized_input])
        predicted_proba = model.predict_proba([normalized_input])[0]
        risk_probability = predicted_proba[1]
        
        # 显示结果卡片
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(f"### CPSP Risk Probability: **{risk_probability * 100:.2f}%**")
        st.markdown(f"Predicted Class: **{'High Risk' if prediction[0] == 1 else 'Low Risk'}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP 分析
        st.markdown('<div class="section-title">Feature Contribution Analysis (SHAP)</div>', unsafe_allow_html=True)
        with st.spinner("Generating SHAP force plot..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))
            
            plt.figure(figsize=(18, 8))
            shap.force_plot(explainer.expected_value, shap_values[0],
                           pd.DataFrame([normalized_input], columns=selected_features),
                           matplotlib=True)
            plt.tight_layout()
            st.pyplot(plt)
            st.caption("Red: increases risk, Blue: decreases risk")

# 页脚
st.markdown("---")
st.caption("Chronic Post-Surgical Pain (CPSP) Prediction Model")



