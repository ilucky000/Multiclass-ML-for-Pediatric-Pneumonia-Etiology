
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import shap
import os
st.set_page_config(page_title="Pneumonia pathogens in Children", layout="wide")
st.markdown("""
    <style>
        @font-face {
            font-family: 'Times New Roman';
        }
        /* 全局字体设置 */
        html, body, [class*="css"], div, p, h1, h2, h3, h4, h5, h6, 
        .stMarkdown, .stButton, .stTable, .stMetric, .stAlert, .stInfo, 
        .stSuccess, .stWarning, .stError, .stSelectbox, .stMultiSelect, 
        .stTextInput, .stNumberInput, .stDateInput, .stTimeInput, 
        .stHeader, .stSidebar, .stTabs, .stTab, .stDataFrame,
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
        div[data-baseweb], button, input, select, textarea {
            font-family: 'Times New Roman', serif !important;
        }
        /* 确保表格内容也使用Times New Roman */
        .dataframe td, .dataframe th {
            font-family: 'Times New Roman', serif !important;
        }
    </style>
""", unsafe_allow_html=True)

plt.rcParams['font.family'] = 'Times New Roman'
st.title("Pneumonia pathogens in Children")
st.markdown("""
### Instructions:
1. Enter the children's clinical indicators in the input fields above.
2. Click the "Predict" button to get results.
3. Retain two significant figures for units.
4. SHAP values show how each feature contributes to the prediction.
""")
st.write("---")
st.write("Please input children's clinical indicators:")

@st.cache_resource
def load_model():
    # 获取当前脚本的目录
    script_dir = os.path.dirname(__file__)
    # 构建模型路径
    model_file = os.path.join(script_dir, 'model.pkl')
    
    # 检查文件是否存在
    if not os.path.exists(model_file):
        st.error(f"模型文件不存在: {model_file}")
        return None
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

col1, col2, col3 = st.columns(3)

###########["Age","Season","NE_per",'ALP','EOS_per','Mg','Ca','PAB','MO_obs','RDW','MCH','LDH','DBil',"BUN"]] 
with col1:
    Age = st.number_input('Age [year]', min_value=0.00, max_value=18.00, value=None, format="%.2f")
    ALP = st.number_input('Alkaline phosphatase (ALP) [IU/L]', value=None, format="%.2f")
    Ca = st.number_input('Calcium (Ca) [mmol/L]', value=None, format="%.2f")
    RDW = st.number_input('Red blood cell distribution width (RDW) [fL]', value=None, format="%.2f")
    DBil= st.number_input('Direct bilirubin (DBil) [μmol/L]', value=None, format="%.2f")
      
with col2:
    # Season 选择框 - 显示名称，存储对应的数字
    season_options = {
        "Spring (Mar-May)": 1,
        "Summer (Jun-Aug)": 2,
        "Autumn (Sep-Nov)": 3,
        "Winter (Dec-Feb)": 4
    }
    selected_season = st.selectbox('Season', list(season_options.keys()))
    EOS_per = st.number_input('Eosinophil ratio (EOS%) [%]', value=None, format="%.2f")
    PAB = st.number_input('Prealbumin (PAB) [mg/L]', value=None, format="%.2f")
    MCH = st.number_input('Mean corpuscular hemoglobin (MCH) [pg]', value=None, format="%.2f")
    BUN= st.number_input('Blood urea nitrogen (BUN) [mmol/L]', value=None, format="%.2f")
    
with col3:
    NE_per = st.number_input('Neutrophil ratio (NE%) [%]', value=None, format="%.2f")
    Mg = st.number_input('Magnesium (Mg) [mmol/L]', value=None, format="%.2f")
    MO_obs = st.number_input('Absolute value of monocyte (MO#) [10*9/L]', value=None, format="%.2f")
    LDH = st.number_input('Lactate dehydrogenase (LDH) [IU/L]', value=None, format="%.2f")
    
if st.button('Predict'):
    # 检查所有输入是否完整
    input_values = [Age,NE_per,ALP,EOS_per,Mg,Ca,PAB,MO_obs,RDW,MCH,LDH,DBil,BUN]
    if any(v is None or v == '' for v in input_values):
        st.error("Please fill in all input fields!")
    else:
        # 将季节名称转换为数字（关键步骤）
        season_number = season_options[selected_season]
        
        input_data = pd.DataFrame([[Age, season_number, NE_per,ALP,EOS_per,Mg,Ca,PAB,MO_obs,RDW,MCH,LDH,DBil,BUN]],
                                   columns=["Age","Season","NE_per",'ALP','EOS_per','Mg','Ca','PAB','MO_obs','RDW','MCH','LDH','DBil',"BUN"])

        input_data["Season"] = pd.Categorical(input_data["Season"],categories=[1,2,3,4])

        prediction_proba = model.predict_proba(input_data)[0]
        prediction_class = model.predict(input_data)[0]
    
    pathogen_types = {
        1: "Bacterial Pneumonia",
        2: "Viral Pneumonia", 
        3: "Atypical Pneumonia"
        }
    
    st.write("---")


    st.subheader("Prediction Results")
        
    # 显示三个类别的概率
    st.write(f"**Bacterial Pneumonia**: {prediction_proba[0]:.1%}")
    st.progress(prediction_proba[0])
        
    st.write(f"**Viral Pneumonia**: {prediction_proba[1]:.1%}")
    st.progress(prediction_proba[1])
        
    st.write(f"**Atypical Pneumonia**: {prediction_proba[2]:.1%}")
    st.progress(prediction_proba[2])

    # 最终预测结果
    #st.success(f"**Prediction Result：Type {prediction_class}** (Confidence: {max(prediction_proba):.1%})")
    st.success(f"**Prediction: {pathogen_types[prediction_class]}** (Confidence: {max(prediction_proba):.1%})")

    # Chart 1: Pie Chart using matplotlib
    # Chart 2: Model Interpretation
    col1, col2 = st.columns([1.1, 1.3])  # 左右宽度比例，可调整

    with col1:
      #st.subheader("Probability Distribution")
      st.markdown("<h3 style='text-align: center;'>Probability Distribution</h3>",unsafe_allow_html=True)
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
        
      import matplotlib.pyplot as plt
      fig1, ax = plt.subplots(figsize=(4, 4))

      colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
      labels = ['Bacterial Pneumonia', 'Viral Pneumonia', 'Atypical Pneumonia']
      explode = (0.05, 0.05, 0.05)

      wedges, texts, autotexts = ax.pie(
          prediction_proba,
          explode=explode,
          labels=labels,
          autopct='%1.1f%%',
          colors=colors,
          textprops={'fontsize': 10, 'fontweight': 'bold'},
          shadow=True,
          startangle=90
      )

      for autotext in autotexts:
          autotext.set_color('white')
          autotext.set_fontsize(11)
          autotext.set_fontweight('bold')

      #ax.set_title('Probability Distribution', fontsize=14, fontweight='bold', pad=20)

      st.pyplot(fig1)
      plt.close(fig1)
 

    with col2:
      st.markdown("<h3 style='text-align: center;'>Model Interpretation</h3>",unsafe_allow_html=True)
      #st.subheader("Model Interpretation")

      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(input_data)

      pred_label = model.predict(input_data)[0]
 
      pred_idx = np.where(
          model.classes_ == pred_label
      )[0][0]

      sv = shap.Explanation(
          values=shap_values[0, :, pred_idx],
          base_values=explainer.expected_value[pred_idx],
          data=input_data.iloc[0].values,
          feature_names=input_data.columns.tolist()
      )

      plt.figure(figsize=(8, 6))

      shap.plots.waterfall(
          sv,
          max_display=14,
          show=False
      )

      fig2 = plt.gcf()
      st.pyplot(fig2)
      plt.close(fig2)
      
st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: LightGBM Classifier
- Training Data: Clinical Data
- Target Variable: Pneumonia pathogens
- Number of Features: 14 Clinical Indicators
""")


st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- NE%: 7.10-90.60 (%)
- ALP: 63.00-539.00 (IU/L)
- EOS%: 0.00-10.10 (%)
- Mg: 0.67-1.29 (mmol/L)
- Ca: 1.95-2.83 (mmol/L)
- PAB: 52.50-353.20 (mg/L)
- MO#: 0.70-3.13 (10*9/L)
- RDW: 31.90-60.80 (fL)
- MCH: 18.50-39.00 (pg)
- LDH:165.00-1024.00 (IU/L)
- DBIL:0.50-17.90 (μmol/L)
- BUN:0.70-7.90 (mmol/L)
""")

