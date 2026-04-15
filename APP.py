#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 工作环境
import os
import warnings
current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)


# In[2]:


import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# 设置 Matplotlib 的全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'


# In[4]:


# 
model = joblib.load("simple_model.pkl")
min_max_params = joblib.load("min_max_params_app.pkl")
selected_features = joblib.load("selected_features_app.pkl")  # 25


# In[5]:


feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]


# In[ ]:


selected_features = ['Subacute pain NRS score on postoperative day thirty', 'Drainage tube placement', 'Acute pain NRS score on postoperative day three', 'Gender', 'Rehabilitation NRS score on postoperative day thirty', 'Open surgery', 'Abdominal surgery', 'Treponema pallidum antibody', 'Consumption of intraoperative opioid', 'Operation grading IV', 'Preoperative pain NRS score', 'Consumption of intraoperative propofol', 'Summer season', 'Preoperative PSQI score', 'Surface or limb surgery']


# In[6]:


# 定义每个特征的输入范围
feature_ranges = {
    "Subacute pain NRS score on postoperative day thirty": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Acute pain NRS score on postoperative day three": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Rehabilitation NRS score on postoperative day thirty": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Consumption of intraoperative opioid": {"min": 0.0, "max": None,"step": 0.1},
    "Treponema pallidum antibody": {"min": 0.0, "max": 1.0,"step": 0.1},
    "Acute pain NRS score on postoperative day one": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Preoperative pain NRS score": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Hospitalizing expenses": {"min": 0.0, "max": None,"step":0.1},
    "Consumption of intraoperative propofol": {"min": 0.0, "max": None,"step": 0.1},
    "Preoperative PSQI score": {"min": 0.0, "max": 20.0,"step": 1.0}
    #连续特征的范围
}


# In[7]:


st.title("CPSP Prediction")
inputs = {}
# 定义哪些特征是二元的
binary_features = ["Drainage tube placement", "Open surgery", "Gender",
                  "Abdominal surgery", 'Operation grading IV',
                  'PHQ9-Trouble in sleep','Summer season',
                  
                  'Surface or limb surgery']  # 示例：这些特征只能取0或1

for feature in selected_features:
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


# In[8]:


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
    st.success(f"Based on this model, the CPSP risk of this patient is {risk_probability * 100:.2f}%")

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))

    # 显示SHAP力图
    plt.figure(figsize=(20, 12))  # 调整图像大小
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([normalized_input], columns=selected_features), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")


# In[ ]:




