import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载标准器和模型
scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
    'P': joblib.load('scaler_standard_P.pkl'),
    'U': joblib.load('scaler_standard_U.pkl')
}

models = {
    'C': joblib.load('lightgbm_C.pkl'),
    'P': joblib.load('lightgbm_P.pkl'),
    'U': joblib.load('lightgbm_U.pkl')
}

# 定义特征名称
display_features_to_scale = [
    'Age (years)',                                  # Age (e.g., 52 years)
    'Endometrial thickness (mm)',                   # Endometrial thickness in mm
    'HE4 (pmol/L)',                                 # HE4 level in pmol/L
    'Menopause (1=yes)',                            # Menopause status (1=yes)
    'HRT (Hormone Replacement Therapy, 1=yes)',     # HRT status (1=yes)
    'Endometrial heterogeneity (1=yes)',            # Endometrial heterogeneity (1=yes)
    'Uterine cavity occupation (1=yes)',            # Uterine cavity occupation (1=yes)
    'Uterine cavity occupying lesion with rich blood flow (1=yes)', # Uterine cavity occupying lesion with rich blood flow (1=yes)
    'Uterine cavity fluid (1=yes)'                  # Uterine cavity fluid (1=yes)
]

# 原始特征名称，用于标准化器
original_features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

additional_features = {
    'C': ['CM5141.0', 'CM6160.0', 'CM7441.0', 'CM7439.0', 'CM7438.0', 'CM5139.0', 'CM6557.0', 'CM4088.0'],
    'P': ['PM733.0', 'PM285.0', 'PM673.0', 'PM469.0', 'PP14.0', 'PP344.0', 'PP526.0', 'PP443.0', 'PM787.0', 'PM722.0'],
    'U': ['UM7578.0', 'UM510.0', 'UM507.0', 'UM670.0', 'UM351.0', 'UM5905.0', 'UM346.0', 'UM355.0', 
          'UM8899.0', 'UM1152.0', 'UM5269.0', 'UM6437.0', 'UM5906.0', 'UM7622.0', 'UM8898.0', 'UM2132.0', 
          'UM3513.0', 'UM790.0', 'UM8349.0', 'UM2093.0', 'UM4210.0', 'UM3935.0', 'UM4256.0']
}

# Streamlit界面
st.title("疾病风险预测器")

# 模型选择
selected_models = st.multiselect(
    "选择要使用的模型（可以选择一个或多个）",
    options=['U', 'C', 'P'],
    default=['U']
)

# 获取用户输入
user_input = {}

# 定义特征输入
for i, feature in enumerate(display_features_to_scale):
    if "1=yes" in feature:  # 对于分类变量，限制输入为0或1
        user_input[original_features_to_scale[i]] = st.selectbox(f"{feature}:", options=[0, 1])
    else:  # 对于连续变量，使用数值输入框
        user_input[original_features_to_scale[i]] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# 为每个选定的模型定义额外特征
for model_key in selected_models:
    for feature in additional_features[model_key]:
        user_input[feature] = st.number_input(f"{feature} ({model_key}):", min_value=0.0, value=0.0)

# 定义模型预测结果存储字典
model_predictions = {}

# 对选定的每个模型进行标准化和预测
for model_key in selected_models:
    # 针对每个模型构建专用的输入数据
    model_input_df = pd.DataFrame([user_input])
    
    # 只保留该模型的特征列
    model_features = original_features_to_scale + additional_features[model_key]
    model_input_df = model_input_df[model_features]
    
    # 对需要标准化的特征进行标准化
    model_input_df[original_features_to_scale] = scalers[model_key].transform(model_input_df[original_features_to_scale])
    
    # 使用模型进行预测
    predicted_proba = models[model_key].predict_proba(model_input_df)[0]
    predicted_class = models[model_key].predict(model_input_df)[0]
    
    # 保存预测结果
    model_predictions[model_key] = {
        'proba': predicted_proba,
        'class': predicted_class
    }
    
    # 显示每个模型的预测概率
    st.write(f"**模型 {model_key} 预测概率:**")
    st.write(f"类别 0（无疾病）: {predicted_proba[0]:.2f}")
    st.write(f"类别 1（有疾病）: {predicted_proba[1]:.2f}")

# 确定最终预测类别
final_class = None

if len(selected_models) == 1:
    # 若只选择一个模型，则采用该模型的预测结果
    final_class = list(model_predictions.values())[0]['class']

elif len(selected_models) == 2:
    # 若选择两个模型，按排名优先级 U > C > P 选择预测结果
    if 'U' in selected_models:
        final_class = model_predictions['U']['class']
    elif 'C' in selected_models:
        final_class = model_predictions['C']['class']
    else:
        final_class = model_predictions['P']['class']

elif len(selected_models) == 3:
    # 若选择三个模型，以预测类别 0 或 1 出现次数最多的为最终结果
    classes = [model_predictions[model]['class'] for model in selected_models]
    final_class = max(set(classes), key=classes.count)

# 显示最终预测结果
st.write("**最终预测结果:**")
if final_class == 1:
    st.write(
        "**结果: 您有较高的患病风险。** 根据模型的预测结果，建议您住院接受进一步的专业医疗评估。"
    )
else:
    st.write(
        "**结果: 您的患病风险较低。** 建议您定期进行健康检查，以便随时监控您的健康状况。"
    )
