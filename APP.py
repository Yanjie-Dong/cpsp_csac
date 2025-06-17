#!/usr/bin/env python
# coding: utf-8

# In[82]:


# 工作环境
import os
import warnings
current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)


# In[83]:

import joblib
import streamlit as st
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


# In[84]:


# 
model = joblib.load("simple_model.pkl")
min_max_params = joblib.load("min_max_params_app.pkl")
selected_features = joblib.load("selected_features_app.pkl")  # 25


# In[85]:

feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]


# In[ ]:


# 定义每个特征的输入范围
feature_ranges = {
    "Subacute pain NRS score at POD30": {"min": 0, "max": 10,"step": 1,},
    "The 10th percentile of postoperative NRS score": {"min": 0, "max": 10,"step": 1},
    "Postoperative NRS CWT (coeff=2, width=2, scales=(2/5/10/20))": {"min": 0.0, "max": 1.0,"step":0.1},
    "Rehabilitation feeling at POD30": {"min": 0, "max": 10,"step": 1},
    "Surgical month": {"min": 1, "max": 12,"step": 1},
    "Consumption of intraoperative opioid": {"min": 0.0, "max": None,"step":0.1},
    "Intraoperative crystalloid": {"min": 0, "max": None,"step": 1},
    "Treponema pallidum antibody": {"min": 0.0, "max": None,"step":0.1},
    "Preoperative pain NRS score": {"min": 0, "max": 10,"step": 1},
    "Hospitalizing expenses": {"min": 0, "max": None,"step": 1},
    "Preoperative serummagnesium": {"min": 0.0, "max": None,"step": 0.1},
    "Consumption of intraoperative sevoflurane": {"min": 0.0, "max": None,"step":0.1},
    "Consumption of intraoperative propofol": {"min": 0.0, "max": None,"step":0.1},
    "Preoperative fibrinogen": {"min": 0.0, "max": None,"step":0.1}
    #连续特征的范围
}


# In[86]:


st.title("CPSP Prediction")
inputs = {}
# 定义哪些特征是二元的
binary_features = ["Drainage tube placement", "Open surgery", "Male gender",
                  "Abdominal surgery", 'Operation grading IV',
                  'PHQ9-Trouble in sleep','Junior school and below',
                  'PSQI-Feel too hot when sleep',
                  'Middle thrombus risk',
                  'Surface or limb surgery',
                  'No thrombus risk']  # 示例：这些特征只能取0或1

st.title("CPSP Prediction")

# 创建3列
col1, col2, col3 = st.columns([4, 4, 4])  # 增加每列的宽度比例

inputs = {}
current_col = 0  # 用于跟踪当前列

for i, feature in enumerate(selected_features):
    # 确定当前特征应该放在哪一列
    current_col = i % 4
    
    # 根据列号选择对应的列上下文管理器
    with [col1, col2, col3][current_col]:
        if feature in binary_features:
            # 二元特征，只能取0或1
            inputs[feature] = st.selectbox(
                f"{feature}",
                options=[0, 1],
                format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
            )
        else:
            # 连续特征，使用手动定义的范围和步长
            min_val = feature_ranges[feature]["min"]
            max_val = feature_ranges[feature]["max"]
            step = feature_ranges[feature]["step"]
            
            if max_val is None:
                # 如果没有设置最大值，允许用户输入任意大的值
                inputs[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    step=step,
                    value=min_val  # 默认值设置为最小值
                )
            else:
                # 如果设置了最大值，使用范围限制
                inputs[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                    value=min_val  # 默认值设置为最小值
                )



# In[87]:


if st.button("Predict"):
    # (1)
    user_input = np.array([inputs[f] for f in selected_features])
    min_vals = min_max_params["min"][selected_indices]  # 25min
    max_vals = min_max_params["max"][selected_indices]  # 25max
    normalized_input = (user_input - min_vals) / (max_vals - min_vals)
                            
     # (2) 预测（注意输入是2D数组）
    prediction = model.predict([normalized_input])
    predicted_proba = model.predict_proba([normalized_input])[0]

    # 显示预测结果
    risk_probability = predicted_proba[1]  # 正类的概率
    st.success(f"Based on this model, the output probability of CPSP risk is {risk_probability * 100:.2f}%.  A predicted probability ≥12.40% (the optimal threshold determined by Youden's index) is classified as high risk for CPSP.")

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))

    # 显示SHAP力图
    plt.figure(figsize=(20, 12))  # 调整图像大小
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([normalized_input], columns=selected_features), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")

