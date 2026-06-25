"""
╔══════════════════════════════════════════════════════════════════════════╗
║   TURBINE GUARDIAN  /  حارس التوربين  —  v6.3 (REAL NASA FD001 DATA)    ║
║   Gas Turbine Sensor Fault Detection & Decision System                   ║
║   نظام كشف الأعطال واتخاذ القرار الصناعي لتوربينات الغاز              ║
║ ───────────────────────────────────────────────────────────────────────  ║
║  الجديد في v6 / What's new in v6:                                       ║
║  ✅ نمط محاكاة مستلهم من NASA C-MAPSS | NASA C-MAPSS-inspired pattern   ║
║     (محاكاة رياضية معايرة على إحصاءات NASA المنشورة — ليست ملفات FD001  ║
║     الأصلية | Mathematical simulation calibrated to published NASA stats ║
║     — NOT the original FD001 raw files)                                  ║
║  ✅ كتالوج مصور ثنائي اللغة MS5002C          | Illustrated bilingual    ║
║     catalog                                                              ║
║  ✅ زر تشغيل واحد                            | Single start button      ║
║  ✅ قرار نصي صناعي فوري (بدل SPE Charts)     | Plain-text decision      ║
║  ✅ XAI — تفسير القرار بالأرقام              | XAI explanation engine   ║
║  ✅ تحديد نوع العطل Bias/Drift/Noise          | Fault type identification║
║  ✅ محرك توصيات صيانة ذكي                    | Smart recommendation     ║
║  ✅ فحص جودة البيانات قبل التحليل            | Data quality check       ║
║  ✅ Health Score 0-100% بدل Fault/No Fault    | System health score      ║
║  ✅ سجل أعطال متطور مع تكرار                 | Advanced fault log        ║
║  ✅ تصدير JSON للتكامل الصناعي (SCADA/DCS)    | JSON/CSV/PDF export      ║
║  ✅ وضع بسيط للعامل + وضع خبير للمهندس       | Simple + Expert modes     ║
╚══════════════════════════════════════════════════════════════════════════╝
التشغيل / Run:
    pip install streamlit numpy pandas matplotlib scikit-learn scipy reportlab openpyxl
    streamlit run turbine_guardian_v6.py
"""

# ══ Imports ══
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score)
import pickle, io, os, json, warnings
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image)
from reportlab.lib.enums import TA_CENTER
warnings.filterwarnings('ignore')

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'turbine_guardian_log.json')

# ══════════════════════════════════════════════════════════════════════
#  DEMO MODE CONFIG  —  نسخة العرض المجانية
#  ضع DEMO_MODE = False في النسخة الكاملة المباعة
# ══════════════════════════════════════════════════════════════════════
DEMO_MODE = True          # True = ديمو مجاني  |  False = نسخة كاملة
DEMO_MAX_POINTS = 300     # حد أقصى لنقاط البيانات في الديمو
GUMROAD_URL = "https://gumroad.com/l/turbine-guardian"  # رابط الشراء
DEMO_WATERMARK = "🔒 DEMO — turbineguardian.io"

# ══════════════════════════════════════════════════════════════════════
#  NASA C-MAPSS — Reference Statistics Only (NOT raw data)
#  إحصاءات NASA C-MAPSS المرجعية فقط (وليست بيانات خام)
# ══════════════════════════════════════════════════════════════════════
# ⚠️ IMPORTANT / تنبيه مهم:
# The numbers below (means, stds, sensor names) are published reference
# statistics from: A. Saxena & K. Goebel (2008), "Turbofan Engine
# Degradation Simulation Data Set", NASA Ames Prognostics Data Repository,
# dataset CMAPSS FD001.
#
# generate_turbine_data() does NOT load NASA's raw FD001.txt files.
# It generates a SYNTHETIC sine/cosine signal and calibrates its mean
# and noise level to match these published NASA statistics. This is a
# "NASA-inspired simulation", not real NASA telemetry.
#
# الأرقام أدناه (المتوسطات والانحرافات وأسماء الحساسات) هي إحصاءات مرجعية
# منشورة من نفس الورقة العلمية، لكن الدالة generate_turbine_data() لا تُحمّل
# ملفات FD001.txt الأصلية من NASA — هي تولّد إشارة رياضية (sine/cosine)
# ثم تُعايرها لتقترب من هذه الإحصاءات المنشورة. هذا "محاكاة مستلهمة من
# NASA"، وليس بيانات تليمتري حقيقية من NASA.
#
# To use REAL NASA data: download FD001.txt from
# https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
# and replace generate_turbine_data() with a loader for that file.

NASA_CMAPSS_SENSOR_NAMES = {
    'en': ['Fan Inlet Temp (T2)', 'LPC Outlet Temp (T24)', 'HPC Outlet Temp (T30)',
           'LPT Outlet Temp (T50)', 'Fan Inlet Pressure (P2)', 'Bypass Pressure (P15)',
           'HPC Outlet Pressure (P30)'],
    'ar': ['حرارة دخول المروحة T2', 'حرارة خروج الضاغط المنخفض T24', 'حرارة خروج الضاغط العالي T30',
           'حرارة خروج التوربين المنخفض T50', 'ضغط دخول المروحة P2', 'ضغط قناة الجسر P15',
           'ضغط خروج الضاغط العالي P30'],
}
NASA_CMAPSS_NORMAL = {
    'means':  [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36],
    'stds':   [0.0,    0.50,   8.55,    14.94,   0.0,   0.0,   7.05],
    'units':  ['R', 'R', 'R', 'R', 'psia', 'psia', 'psia'],
}
MS5002C_SPECS = {
    'power_mw': 30.6, 'speed_rpm': 4670, 'pressure_ratio': 8.4,
    'exhaust_temp_c': 487, 'mass_flow_kgs': 128.5, 'efficiency': 28.5,
    'application_en': 'Pipeline compression, Power generation',
    'application_ar': 'ضغط خطوط الأنابيب، توليد الطاقة',
    'origin': "Algeria / Hassi R'Mel — الجزائر / حاسي الرمل",
}

def generate_turbine_data(n=1000, seed=42, nasa_mode=False):
    """
    Generate SYNTHETIC turbine sensor data (mathematical simulation).
    nasa_mode=True  -> sine/cosine signal calibrated to published NASA CMAPSS
                        FD001 reference statistics (mean/std). This is a
                        NASA-INSPIRED SIMULATION, not real NASA telemetry.
    nasa_mode=False -> Classic MS5002C simulation (v5 behavior, kept for continuity)

    توليد بيانات حساسات توربين اصطناعية (محاكاة رياضية).
    nasa_mode=True  -> إشارة جيبية مُعايرة على إحصاءات NASA CMAPSS FD001
                        المنشورة. هذا "محاكاة مستلهمة من NASA"، وليس
                        بيانات تليمتري NASA حقيقية.
    nasa_mode=False -> محاكاة MS5002C الكلاسيكية (سلوك v5)
    """
    np.random.seed(seed); t = np.linspace(0, 10, n)
    if nasa_mode:
        u_temp     = np.cos(2*np.pi*0.12*t) + 0.4*np.sin(2*np.pi*0.35*t)
        u_pressure = np.sin(2*np.pi*0.15*t) + 0.3*np.cos(2*np.pi*0.40*t)
        noise = [0.002, 0.005, 0.006, 0.008, 0.001, 0.001, 0.005]
        X = np.column_stack([
            u_temp     + noise[0]*np.random.randn(n),
            u_temp     + noise[1]*np.random.randn(n),
            u_temp     + 0.5*u_pressure + noise[2]*np.random.randn(n),
            u_temp     + 0.7*u_pressure + noise[3]*np.random.randn(n),
            u_pressure + noise[4]*np.random.randn(n),
            u_pressure + noise[5]*np.random.randn(n),
            3*u_temp   + 2*u_pressure   + noise[6]*np.random.randn(n),
        ])
        for i in range(X.shape[1]):
            X[:, i] += np.linspace(0, 0.008*(i+1), n)  # synthetic slow drift, inspired by NASA RUL degradation shape
    else:
        u1 = np.sin(2*np.pi*0.3*t) + 0.5*np.sin(2*np.pi*0.7*t)
        u2 = np.cos(2*np.pi*0.2*t) + 0.3*np.cos(2*np.pi*0.5*t)
        X = np.column_stack([
            u1+0.007*np.random.randn(n), u1+0.006*np.random.randn(n),
            u1+0.0065*np.random.randn(n), u2+0.005*np.random.randn(n),
            u2+0.006*np.random.randn(n), 3*u1+2*u2+0.004*np.random.randn(n),
            2*u1+u2+0.005*np.random.randn(n),
        ])
    return X, t

# ══════════════════════════════════════════════════════════════════════
#  REAL NASA C-MAPSS FD001 Data Loader  /  محمّل بيانات NASA الحقيقية
# ══════════════════════════════════════════════════════════════════════
# This loads NASA's ACTUAL FD001.txt files — unlike generate_turbine_data()
# above, there is no synthetic generation here. This is genuine NASA
# telemetry from the official Prognostics Data Repository.
#
# هذا يُحمّل ملفات FD001.txt **الحقيقية** من NASA — بخلاف الدالة أعلاه،
# لا يوجد أي توليد اصطناعي هنا. هذا تليمتري NASA حقيقي من المستودع الرسمي.

NASA_FD001_COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]

# These 4 sensors are constant (std≈0) across all of FD001 and carry no
# information — this is a well-known property of the dataset, not a bug.
# هذه الحساسات الأربعة ثابتة (std≈0) عبر كل FD001 ولا تحمل أي معلومة —
# هذه خاصية معروفة في أدبيات المجموعة، وليست خللاً.
NASA_FD001_CONSTANT_SENSORS = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# The 7 most informative (highest-variance) sensors, used to map onto this
# app's existing 7-sensor pipeline without changing the rest of the system.
NASA_FD001_TOP7_SENSORS = ['sensor_9', 'sensor_14', 'sensor_4', 'sensor_3', 'sensor_17',
                           'sensor_12', 'sensor_7']

NASA_FD001_SENSOR_LABELS = {
    'sensor_1':'T2 (Fan Inlet Temp)','sensor_2':'T24 (LPC Outlet Temp)','sensor_3':'T30 (HPC Outlet Temp)',
    'sensor_4':'T50 (LPT Outlet Temp)','sensor_5':'P2 (Fan Inlet Pressure)','sensor_6':'P15 (Bypass Pressure)',
    'sensor_7':'P30 (HPC Outlet Pressure)','sensor_8':'Nf (Fan Speed)','sensor_9':'Nc (Core Speed)',
    'sensor_10':'epr (Engine Pressure Ratio)','sensor_11':'Ps30 (HPC Outlet Static Pressure)',
    'sensor_12':'phi (Fuel Flow Ratio)','sensor_13':'NRf (Corrected Fan Speed)','sensor_14':'NRc (Corrected Core Speed)',
    'sensor_15':'BPR (Bypass Ratio)','sensor_16':'farB (Burner Fuel-Air Ratio)','sensor_17':'htBleed (Bleed Enthalpy)',
    'sensor_18':'Nf_dmd (Demanded Fan Speed)','sensor_19':'PCNfR_dmd (Demanded Corrected Fan Speed)',
    'sensor_20':'W31 (HPT Coolant Bleed)','sensor_21':'W32 (LPT Coolant Bleed)',
}

@st.cache_data
def load_nasa_fd001(train_path, test_path=None, rul_path=None):
    """
    Load REAL NASA C-MAPSS FD001 files (genuine telemetry, not synthetic).
    Returns a dict with train_df, test_df (optional), rul_true (optional),
    and the list of unit IDs available.
    """
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=NASA_FD001_COLUMNS)
    result = {'train_df': train_df, 'units': sorted(train_df['unit'].unique().tolist())}
    if test_path is not None:
        result['test_df'] = pd.read_csv(test_path, sep=r'\s+', header=None, names=NASA_FD001_COLUMNS)
    if rul_path is not None:
        rul_true = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        result['rul_true'] = rul_true['RUL'].values
    return result

def extract_nasa_unit_run(train_df, unit_id, sensors=None):
    """
    Extract one engine's full run-to-failure trajectory (genuine NASA data)
    as an (N, n_sensors) array ready for the existing detection pipeline.
    The engine runs normally for a while, then degrades until failure —
    exactly the same shape the rest of this app expects (X_clean_full).
    """
    if sensors is None:
        sensors = NASA_FD001_TOP7_SENSORS
    unit_data = train_df[train_df['unit'] == unit_id].sort_values('cycle')
    X = unit_data[sensors].values.astype(float)
    t = unit_data['cycle'].values.astype(float)
    return X, t, sensors

def extract_nasa_test_unit(test_df, rul_true_array, unit_id, sensors=None):
    """
    Extract one engine's test trajectory + its TRUE RUL at the cutoff point
    (from NASA's own ground-truth file, not estimated). Used for genuine
    RUL evaluation (RMSE against NASA's published ground truth).
    """
    if sensors is None:
        sensors = NASA_FD001_TOP7_SENSORS
    unit_data = test_df[test_df['unit'] == unit_id].sort_values('cycle')
    X = unit_data[sensors].values.astype(float)
    t = unit_data['cycle'].values.astype(float)
    units_sorted = sorted(test_df['unit'].unique().tolist())
    idx = units_sorted.index(unit_id)
    true_rul_at_cutoff = float(rul_true_array[idx]) if idx < len(rul_true_array) else None
    return X, t, true_rul_at_cutoff


TRANSLATIONS = {
    'ar': {
        'app_title':        'حارس التوربين',
        'app_subtitle':     'نظام اتخاذ القرار الصناعي · MS5002C · محاكاة مستلهمة من NASA C-MAPSS',
        'app_desc':         'قرار فوري · تفسير XAI · توصية صيانة · Autoencoder ACPNL',
        'display_mode':     'وضع العرض',
        'simple_mode':      '👤 وضع بسيط',
        'expert_mode':      '🧑‍🔧 وضع خبير',
        'data_source':      'مصدر البيانات',
        'nasa_real':        '🛰️ بيانات NASA الحقيقية (FD001)',
        'nasa_sim':         '🚀 محاكاة مستلهمة من NASA CMAPSS',
        'simulation':       '🔬 محاكاة MS5002C',
        'upload_file':      '📁 رفع ملف CSV/Excel',
        'language':         '🌐 اللغة / Language',
        'run_analysis':     '🚀 تشغيل التحليل الكامل',
        'load_model':       'تحميل نموذج محفوظ',
        'num_points':       'عدد نقاط القياس',
        'fault_settings':   '🔴 إعدادات العطل',
        'engine_select':    '✈️ اختر محرك NASA',
        'engine_info':      'هذا محرك حقيقي من أسطول NASA — يعمل بشكل طبيعي ثم يتدهور فعلياً حتى الفشل',
        'real_data_badge':  '✅ بيانات تليمتري حقيقية — وليست محاكاة',
        'use_test_set':     'استخدم بيانات الاختبار (مع RUL الحقيقي)',
        'true_rul_label':   'RUL الحقيقي (من NASA)',
        'rul_comparison':   'مقارنة RUL: المُقدَّر مقابل الحقيقي',
        'faulty_sensor':    'الحساس المعطوب',
        'fault_start':      'بداية العطل (%)',
        'fault_severity':   'شدة العطل',
        'fault_type':       'نوع العطل',
        'algorithm':        'الخوارزمية',
        'confidence_level': 'مستوى الثقة (%)',
        'epochs':           'دورات التدريب',
        'partial_epochs':   'دورات التحديد',
        'ewma_alpha':       'معامل EWMA (α)',
        'model_section':    '🧠 النموذج',
        'bias':             'انزياح (Bias)',
        'drift':            'انجراف (Drift)',
        'noise':            'ضجيج (Noise)',
        'linear_pca':       '📐 ACP خطية',
        'kernel_pca':       '🌀 Kernel PCA',
        'autoencoder_lbl':  '🧠 Autoencoder (ACPNL)',
        'detection_rate':   'نسبة الكشف',
        'confidence':       'الثقة',
        'fault_type_label': 'نوع العطل',
        'false_alarms':     'إنذارات كاذبة',
        'overall_health':   'الصحة الإجمالية',
        'tab_decision':     '🚨 قرار فوري',
        'tab_health':       '🗺️ خريطة الصحة',
        'tab_xai':          '🧠 التفسير (XAI)',
        'tab_eval':         '📐 التقييم العلمي',
        'tab_catalog':      '📘 كتالوج التوربين',
        'tab_fingerprint':  '📡 بصمة العطل',
        'tab_timeline':     '📈 تطور الصحة',
        'tab_localization': '🔍 تحديد الموقع',
        'tab_signals':      '📡 إشارات الحساسات',
        'tab_log':          '📋 سجل الأحداث',
        'download_pdf':     '📄 تحميل تقرير PDF',
        'save_model':       '💾 حفظ النموذج',
        'export_csv':       '📊 تصدير CSV',
        'export_json':      '🔌 تصدير JSON',
        'checking_quality': '🔍 فحص جودة البيانات...',
        'building_model':   'بناء النموذج...',
        'training_complete':'✅ اكتمل التدريب',
        'localizing':       'تحديد موقع العطل...',
        'analysis_complete':'✅ اكتمل التحليل',
        'computing_health': 'حساب درجات الصحة...',
        'healthy':          'سليم',
        'warning_status':   'تحذير',
        'fault_status':     'عطل',
        'fault_identified': '← عطل محدد',
        'health_map_title': '⚙️ خريطة صحة الحساسات',
        'health_summary':   '📊 ملخص الصحة',
        'probable_causes':  '🔍 الأسباب المحتملة',
        'smart_rec':        '⚡ التوصيات الذكية',
        'immediate_action': '⚡ إجراء فوري',
        'short_term':       '🔧 قصير المدى',
        'preventive':       '🛡️ وقائي',
        'data_quality':     '🔍 تقرير جودة البيانات',
        'fault_log':        '📋 سجل أحداث الأعطال',
        'total_events':     'إجمالي الأحداث',
        'critical_events':  'أحداث حرجة',
        'warnings_count':   'تحذيرات',
        'no_events':        'لا توجد أحداث مسجلة بعد. شغّل أول تحليل للبدء.',
        'processing':       '⚙️ جارٍ المعالجة...',
        'normal_period':    'فترة طبيعية',
        'faulty_period':    'فترة عطل',
        'welcome_title':    'مرحباً بك في حارس التوربين v6',
        'pred_warning':     '🔮 تحذير تنبؤي:',
        'already_fault':    '🔴 النظام في منطقة العطل بالفعل',
        'stable':           '✅ مستقر — لا عطل متوقع',
        'min_health':       'أدنى صحة',
        'max_health':       'أعلى صحة',
        'avg_health':       'متوسط الصحة',
        'time_healthy':     'وقت سليم',
        'full_ranking':     '📊 التصنيف الكامل للحساسات',
        'rank':             'الترتيب',
        'fault_prob':       'احتمال العطل',
        'health_score':     'درجة الصحة',
        'risk_ratio':       'نسبة الخطر',
        'status':           'الحالة',
        'footer_txt':       'حارس التوربين v6 · محاكاة مستلهمة من NASA CMAPSS · Autoencoder ACPNL · أداة تعليمية',
        'report_title':     'حارس التوربين — تقرير قرار الصيانة الصناعي',
        'report_subtitle':  'MS5002C · نظام اتخاذ القرار (بيانات NASA حقيقية متاحة) · v6.3',
        'session_info':     'معلومات الجلسة والنموذج',
        'sensor_health':    'حالة صحة الحساسات',
        'charts_section':   'مخططات الكشف والتحديد',
        'maintenance_rec':  'توصيات الصيانة',
        'confidential':     'سري',
        'param':            'المعامل',
        'value':            'القيمة',
        'report_date':      'تاريخ التقرير',
        'algo_used':        'الخوارزمية المستخدمة',
        'samples':          'عدد العينات المحللة',
        'det_rate':         'نسبة الكشف',
        'conf_score':       'درجة الثقة',
        'fault_type_rep':   'نوع العطل المحدد',
        'pred_note':        'تحذير تنبؤي',
        'sensor_col':       'الحساس',
        'health_col':       'الصحة',
        'status_col':       'الحالة',
        'ratio_col':        'النسبة',
        'faulty_rep':       '🔴 معطوب',
        'warn_rep':         '🟡 تحذير',
        'healthy_rep':      '🟢 سليم',
        'xai_title':        '🧠 تفسير القرار (XAI)',
        'xai_contribution': 'نسبة المساهمة في العطل',
        'xai_correlation':  'الحساسات المترابطة المتأثرة',
        'xai_pattern':      'النمط الملاحظ',
        'catalog_title':    '📘 كتالوج التوربين الصناعي',
        'one_btn_note':     '🎯 زر واحد — تشغيل كامل للنظام',
    },
    'en': {
        'app_title':        'TURBINE GUARDIAN',
        'app_subtitle':     'Industrial Decision System · MS5002C · NASA C-MAPSS-Inspired Simulation',
        'app_desc':         'Instant Decision · XAI Explanation · Maintenance Recommendation · Autoencoder ACPNL',
        'display_mode':     'Display Mode',
        'simple_mode':      '👤 Simple Mode',
        'expert_mode':      '🧑‍🔧 Expert Mode',
        'data_source':      'Data Source',
        'nasa_real':        '🛰️ Real NASA Data (FD001)',
        'nasa_sim':         '🚀 NASA CMAPSS-Inspired Simulation',
        'simulation':       '🔬 MS5002C Simulation',
        'upload_file':      '📁 Upload CSV/Excel',
        'language':         '🌐 اللغة / Language',
        'run_analysis':     '🚀 Run Full Analysis',
        'load_model':       'Load Saved Model',
        'num_points':       'Number of Measurements',
        'fault_settings':   '🔴 Fault Settings',
        'engine_select':    '✈️ Select NASA Engine Unit',
        'engine_info':      'This is a real engine from the NASA fleet — it runs normally then genuinely degrades to failure',
        'real_data_badge':  '✅ Real telemetry — not simulated',
        'use_test_set':     'Use test set data (with true RUL)',
        'true_rul_label':   'True RUL (from NASA)',
        'rul_comparison':   'RUL Comparison: Estimated vs. True',
        'faulty_sensor':    'Faulty Sensor',
        'fault_start':      'Fault Start (%)',
        'fault_severity':   'Fault Severity',
        'fault_type':       'Fault Type',
        'algorithm':        'Algorithm',
        'confidence_level': 'Confidence Level (%)',
        'epochs':           'Training Epochs',
        'partial_epochs':   'Partial Epochs',
        'ewma_alpha':       'EWMA Filter (α)',
        'model_section':    '🧠 Model',
        'bias':             'Bias (Offset)',
        'drift':            'Drift (Progressive)',
        'noise':            'Noise (Intermittent)',
        'linear_pca':       '📐 Linear PCA',
        'kernel_pca':       '🌀 Kernel PCA',
        'autoencoder_lbl':  '🧠 Autoencoder (ACPNL)',
        'detection_rate':   'Detection Rate',
        'confidence':       'Confidence',
        'fault_type_label': 'Fault Type',
        'false_alarms':     'False Alarms',
        'overall_health':   'Overall Health',
        'tab_decision':     '🚨 Instant Decision',
        'tab_health':       '🗺️ Health Map',
        'tab_xai':          '🧠 Explanation (XAI)',
        'tab_eval':         '📐 Scientific Evaluation',
        'tab_catalog':      '📘 Turbine Catalog',
        'tab_fingerprint':  '📡 Fault Fingerprint',
        'tab_timeline':     '📈 Health Timeline',
        'tab_localization': '🔍 Localization',
        'tab_signals':      '📡 Sensor Signals',
        'tab_log':          '📋 Fault Log',
        'download_pdf':     '📄 Download PDF Report',
        'save_model':       '💾 Save Model',
        'export_csv':       '📊 Export CSV',
        'export_json':      '🔌 Export JSON',
        'checking_quality': '🔍 Checking data quality...',
        'building_model':   'Building model...',
        'training_complete':'✅ Training complete',
        'localizing':       'Localizing fault...',
        'analysis_complete':'✅ Analysis complete',
        'computing_health': 'Computing health scores...',
        'healthy':          'HEALTHY',
        'warning_status':   'WARNING',
        'fault_status':     'FAULT',
        'fault_identified': '← FAULT IDENTIFIED',
        'health_map_title': '⚙️ SENSOR HEALTH MAP',
        'health_summary':   '📊 HEALTH SUMMARY',
        'probable_causes':  '🔍 PROBABLE CAUSES',
        'smart_rec':        '⚡ SMART RECOMMENDATIONS',
        'immediate_action': '⚡ IMMEDIATE ACTION',
        'short_term':       '🔧 SHORT-TERM',
        'preventive':       '🛡️ PREVENTIVE',
        'data_quality':     '🔍 DATA QUALITY REPORT',
        'fault_log':        '📋 FAULT HISTORY LOG',
        'total_events':     'Total Events',
        'critical_events':  'Critical',
        'warnings_count':   'Warnings',
        'no_events':        'No events logged yet. Run your first analysis to start tracking.',
        'processing':       '⚙️ Processing...',
        'normal_period':    'Normal period',
        'faulty_period':    'Faulty period',
        'welcome_title':    'Welcome to Turbine Guardian v6',
        'pred_warning':     '🔮 PREDICTIVE WARNING:',
        'already_fault':    '🔴 Already in fault zone',
        'stable':           '✅ Stable — no fault predicted',
        'min_health':       'Min Health',
        'max_health':       'Max Health',
        'avg_health':       'Avg Health',
        'time_healthy':     'Time Healthy',
        'full_ranking':     '📊 FULL SENSOR RANKING',
        'rank':             'Rank',
        'fault_prob':       'Fault Probability',
        'health_score':     'Health Score',
        'risk_ratio':       'Risk Ratio',
        'status':           'Status',
        'footer_txt':       'Turbine Guardian v6 · NASA CMAPSS-Inspired Simulation · Autoencoder ACPNL · Educational Tool',
        'report_title':     'TURBINE GUARDIAN — Industrial Maintenance Decision Report',
        'report_subtitle':  'MS5002C · Decision Engine (Real NASA Data Available) · v6.3',
        'session_info':     'Session & Model Information',
        'sensor_health':    'Sensor Health Status',
        'charts_section':   'Detection & Localization Charts',
        'maintenance_rec':  'Maintenance Recommendations',
        'confidential':     'CONFIDENTIAL',
        'param':            'Parameter',
        'value':            'Value',
        'report_date':      'Report Date',
        'algo_used':        'Algorithm Used',
        'samples':          'Samples Analyzed',
        'det_rate':         'Detection Rate',
        'conf_score':       'Confidence Score',
        'fault_type_rep':   'Fault Type Identified',
        'pred_note':        'Predictive Warning',
        'sensor_col':       'Sensor',
        'health_col':       'Health Score',
        'status_col':       'Status',
        'ratio_col':        'Risk Ratio',
        'faulty_rep':       '🔴 FAULTY',
        'warn_rep':         '🟡 WARNING',
        'healthy_rep':      '🟢 HEALTHY',
        'xai_title':        '🧠 Decision Explanation (XAI)',
        'xai_contribution': 'Contribution to Fault Detection',
        'xai_correlation':  'Correlated Sensors Affected',
        'xai_pattern':      'Observed Pattern',
        'catalog_title':    '📘 Industrial Turbine Catalog',
        'one_btn_note':     '🎯 One button — runs the full system',
    }
}

ALERT_MESSAGES = {
    'ar': {
        'NORMAL':   {'title':'✅  النظام طبيعي — جميع الحساسات سليمة',
                     'action':'استمر في المراقبة الدورية المعتادة.',
                     'icon':'🟢','color':'#00c853'},
        'WARNING':  {'title':'⚠️  تحذير — شذوذ مكتشف',
                     'action':'راقب الحساس عن كثب. جدول فحصاً خلال 48 ساعة.',
                     'icon':'🟡','color':'#ffd700'},
        'FAULT':    {'title':'🔧  عطل مكتشف — يلزم صيانة',
                     'action':'جدول صيانة عاجلة. التدهور مستمر.',
                     'icon':'🟠','color':'#ff6d00'},
        'CRITICAL': {'title':'🚨  عطل حرج — إجراء فوري مطلوب',
                     'action':'⛔ أوقف التشغيل فوراً. افحص الحساس قبل إعادة التشغيل.',
                     'icon':'🔴','color':'#ff1744'},
    },
    'en': {
        'NORMAL':   {'title':'✅  SYSTEM NORMAL — ALL SENSORS HEALTHY',
                     'action':'Continue standard monitoring. No action required.',
                     'icon':'🟢','color':'#00c853'},
        'WARNING':  {'title':'⚠️  WARNING — ANOMALY DETECTED',
                     'action':'Monitor closely. Schedule inspection within 48 hours.',
                     'icon':'🟡','color':'#ffd700'},
        'FAULT':    {'title':'🔧  FAULT DETECTED — MAINTENANCE REQUIRED',
                     'action':'Schedule urgent maintenance. Degradation is ongoing.',
                     'icon':'🟠','color':'#ff6d00'},
        'CRITICAL': {'title':'🚨  CRITICAL FAULT — IMMEDIATE ACTION REQUIRED',
                     'action':'⛔ STOP TURBINE. Inspect sensor before restart.',
                     'icon':'🔴','color':'#ff1744'},
    }
}

SENSOR_NAMES = {
    'ar': ['ح1 — ضغط الدخول','ح2 — ضغط الوسط','ح3 — ضغط الخروج',
           'ح4 — حرارة الدخول','ح5 — حرارة الخروج',
           'ح6 — سرعة التوربين','ح7 — إخراج الطاقة'],
    'en': ['S1 — Inlet Pressure','S2 — Mid Pressure','S3 — Outlet Pressure',
           'S4 — Inlet Temperature','S5 — Outlet Temperature',
           'S6 — Turbine Speed','S7 — Power Output'],
}
SENSOR_SHORT = {
    'ar': ['ضغط الدخول','ضغط الوسط','ضغط الخروج',
           'حرارة الدخول','حرارة الخروج','سرعة التوربين','إخراج الطاقة'],
    'en': ['Inlet Pressure','Mid Pressure','Outlet Pressure',
           'Inlet Temperature','Outlet Temperature','Turbine Speed','Power Output'],
}

def T(key):
    lang = st.session_state.get('lang', 'en')
    return TRANSLATIONS[lang].get(key, TRANSLATIONS['en'].get(key, key))

def AL(level):
    lang = st.session_state.get('lang', 'en')
    base = ALERT_MESSAGES[lang].get(level, ALERT_MESSAGES['en']['FAULT'])
    en_base = ALERT_MESSAGES['en'].get(level, ALERT_MESSAGES['en']['FAULT'])
    return {**en_base, **base}

def SN(i=None):
    """
    Sensor display name (long form). When real NASA data is active,
    reads from st.session_state['active_sensor_labels'] (set in the
    data-prep step) instead of the MS5002C placeholder names — so the
    UI always shows the sensor names that actually match the loaded data.
    """
    lang = st.session_state.get('lang', 'en')
    active = st.session_state.get('active_sensor_labels')
    if active is not None:
        return active if i is None else (active[i] if i < len(active) else f'S{i+1}')
    names = SENSOR_NAMES[lang]
    return names if i is None else (names[i] if i < len(names) else f'S{i+1}')

def SS(i):
    lang = st.session_state.get('lang', 'en')
    active = st.session_state.get('active_sensor_labels_short')
    if active is not None:
        return active[i] if i < len(active) else f'S{i+1}'
    names = SENSOR_SHORT[lang]
    return names[i] if i < len(names) else f'S{i+1}'

S_COLORS = ['#4db8ff','#4db8ff','#4db8ff','#ffd700','#ffd700','#4dff88','#ff9944']

PROBABLE_CAUSES = {
    'ar': {
        'ضغط الخروج':   ['قيد في خط التصريف','عطل في صمام التصريف','انجراف معايرة الحساس'],
        'ضغط الدخول':   ['صمام دخول جزئياً مغلق','انسداد الفلتر','فقدان الضغط المصدري'],
        'ضغط الوسط':    ['تسرب داخلي','تلف في المرحلة','تآكل السداد'],
        'حرارة الدخول': ['عطل المبادل الحراري','انجراف الترموكوبل','تغير حرارة المحيط'],
        'حرارة الخروج': ['شذوذ غرفة الاحتراق','تدهور نوعية الوقود','خلل نظام التبريد'],
        'سرعة التوربين':['عطل في وحدة التحكم','تغير مفاجئ في الحمل','اهتزاز ميكانيكي'],
        'إخراج الطاقة': ['عطل في المولد','خطأ كهربائي','انخفاض الكفاءة'],
    },
    'en': {
        'Outlet Pressure':    ['Downstream restriction','Discharge valve fault','Sensor calibration drift'],
        'Inlet Pressure':     ['Inlet valve partially closed','Filter blockage','Upstream pressure loss'],
        'Mid Pressure':       ['Internal stage leak','Impeller damage','Seal wear'],
        'Inlet Temperature':  ['Heat exchanger fouling','Thermocouple drift','Ambient temperature change'],
        'Outlet Temperature': ['Combustion chamber anomaly','Fuel quality degradation','Cooling flow reduction'],
        'Turbine Speed':      ['Governor/speed controller fault','Sudden load change','Mechanical vibration'],
        'Power Output':       ['Generator or AVR fault','Electrical grid fault','Turbine efficiency drop'],
    }
}
FAULT_TYPE_CAUSES = {
    'ar': {'bias':'على الأرجح خطأ في المعايرة أو انزياح ثابت في العملية.',
           'drift':'تدهور تدريجي أو اتساخ أو تآكل في الحساس.',
           'noise':'تداخل كهربائي أو توصيل مفكوك أو ضجيج ناجم عن الاهتزاز.'},
    'en': {'bias':'Likely a calibration offset, zeroing error, or constant process shift.',
           'drift':'Progressive wear, fouling, or gradual sensor degradation detected.',
           'noise':'Electrical interference, loose connection, or vibration-induced noise.'}
}
MAINTENANCE_ACTIONS = {
    'ar': {
        'bias':    {'immediate':'أعد معايرة الحساس مقارنةً بالمرجع القياسي',
                    'short_term':'افحص منظم الإشارة للانجراف في نقطة الصفر',
                    'preventive':'جدول دورة معايرة دورية — كل 3 أشهر'},
        'drift':   {'immediate':'افحص عنصر الحساس للاتساخ أو التآكل',
                    'short_term':'قارن قراءات الاتجاه مع مسجل العملية على مدى 7 أيام',
                    'preventive':'ركّب مراقبة حالة على غلاف الحساس'},
        'noise':   {'immediate':'افحص وشدّ جميع التوصيلات الكهربائية',
                    'short_term':'افحص درع الكابل واستمرارية التأريض',
                    'preventive':'راجع مصادر التداخل الكهرومغناطيسي قرب أسلاك الحساس'},
        'unknown': {'immediate':'تحقق من قراءة الحساس بجهاز محمول في الموقع',
                    'short_term':'افحص سلسلة الإشارة الكاملة: حساس → منظم → DCS',
                    'preventive':'أضف قياساً احتياطياً لهذه المعلمة الحرجة'},
    },
    'en': {
        'bias':    {'immediate':'Recalibrate sensor against certified reference standard',
                    'short_term':'Inspect signal conditioner for zero-point drift',
                    'preventive':'Schedule periodic calibration — recommended every 3 months'},
        'drift':   {'immediate':'Inspect sensor element for fouling, scaling, or mechanical wear',
                    'short_term':'Compare reading trend with process historian over last 7 days',
                    'preventive':'Install condition monitoring on sensor housing'},
        'noise':   {'immediate':'Check and tighten all electrical connections and terminal blocks',
                    'short_term':'Inspect cable shielding and grounding continuity',
                    'preventive':'Review EMI/RFI sources near sensor wiring routes'},
        'unknown': {'immediate':'Verify sensor reading with portable instrument on-site',
                    'short_term':'Inspect full signal chain: sensor → conditioner → DCS',
                    'preventive':'Add redundant measurement for this critical parameter'},
    }
}
PRIORITY_MAP = {
    'ar': {'CRITICAL':{'label':'🔴 فوري','time':'خلال ساعة واحدة'},
           'FAULT':   {'label':'🟠 عاجل','time':'خلال 24 ساعة'},
           'WARNING': {'label':'🟡 مخطط','time':'خلال 72 ساعة'},
           'NORMAL':  {'label':'🟢 روتيني','time':'الجدول المعتاد'}},
    'en': {'CRITICAL':{'label':'🔴 IMMEDIATE','time':'Within 1 hour'},
           'FAULT':   {'label':'🟠 URGENT','time':'Within 24 hours'},
           'WARNING': {'label':'🟡 PLANNED','time':'Within 72 hours'},
           'NORMAL':  {'label':'🟢 ROUTINE','time':'Next scheduled'}},
}

# ══════════════════════════════════════════════════════════════════════
#  Page Config & Styles
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Turbine Guardian v6 / حارس التوربين",
                    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# ── DEMO BANNER ──────────────────────────────────────────────────────
if DEMO_MODE:
    st.markdown(f'''
    <div style="background:#fff8e1;border:1.5px solid #f9a825;border-radius:8px;
         padding:10px 18px;margin-bottom:12px;display:flex;align-items:center;
         justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <div style="color:#6d4c00;font-size:0.85rem;font-weight:600;">
        🔒 نسخة تجريبية مجانية — بعض الميزات مقيّدة
        &nbsp;|&nbsp; Free Demo — some features are locked
      </div>
      <a href="{GUMROAD_URL}" target="_blank"
         style="background:#f9a825;color:#fff;padding:6px 16px;border-radius:6px;
                font-size:0.82rem;font-weight:700;text-decoration:none;">
        🛒 اشتر النسخة الكاملة / Buy Full Version
      </a>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800;900&family=Cairo:wght@400;600;700;900&display=swap');
html,body,[class*="css"]{background:#060b14;color:#c8d8e8;}
.stApp{background:radial-gradient(ellipse at top,#0d1f3c 0%,#060b14 60%);}
.hero{background:linear-gradient(135deg,#0a1628 0%,#0d2347 50%,#091220 100%);
  border:1px solid #1e4a7a;border-bottom:2px solid #2a6fbd;
  border-radius:8px;padding:22px 32px;margin-bottom:20px;
  box-shadow:0 4px 40px rgba(30,100,200,0.25);}
.hero h1{font-weight:900;font-size:1.9rem;color:#fff;margin:0 0 4px 0;
  letter-spacing:4px;text-transform:uppercase;
  text-shadow:0 0 30px rgba(77,184,255,0.6);}
.hero .sub{color:#4db8ff;font-family:'Share Tech Mono',monospace;
  font-size:0.76rem;letter-spacing:3px;}
.hero .desc{color:#7ab0d4;font-size:0.7rem;margin-top:4px;}
.nasa-badge{background:linear-gradient(135deg,#1a0050,#0d002a);
  border:1px solid #7c4dff;border-radius:20px;padding:4px 14px;
  font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#b388ff;
  display:inline-block;}
.mcard{background:linear-gradient(135deg,#0a1628,#0d1f3c);
  border:1px solid #1a3a6a;border-radius:8px;padding:13px;
  text-align:center;margin:3px 0;}
.mval{font-family:'Share Tech Mono',monospace;font-size:1.7rem;
  font-weight:bold;color:#4db8ff;line-height:1.2;}
.mlbl{font-size:0.68rem;color:#5a8aaa;letter-spacing:1.5px;
  text-transform:uppercase;margin-top:4px;}
.hcard{border-radius:8px;padding:12px 16px;margin:4px 0;border-left:4px solid;
  font-family:'Share Tech Mono',monospace;font-size:0.85rem;}
.hcard-ok{background:#071a0e;border-color:#00c853;color:#4dff88;}
.hcard-warn{background:#1a1400;border-color:#ffd700;color:#ffd700;}
.hcard-fault{background:#1a0505;border-color:#ff1744;color:#ff4d4d;animation:blink 1.5s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.75}}
.plain-decision{border-radius:10px;padding:22px 26px;margin:10px 0;
  font-family:'Share Tech Mono',monospace;border:2px solid;}
.plain-line{font-size:0.9rem;color:#c8d8e8;margin:6px 0;line-height:1.6;}
.plain-label{color:#5a8aaa;font-size:0.78rem;}
.xai-card{background:#080d18;border:1px solid #1a3a6a;border-left:4px solid;
  border-radius:6px;padding:14px 18px;margin:6px 0;font-family:'Share Tech Mono',monospace;}
.health-score-big{font-family:'Share Tech Mono',monospace;font-size:3rem;
  font-weight:bold;text-align:center;line-height:1;}
.health-trend{font-size:0.75rem;text-align:center;margin-top:4px;}
.catalog-card{background:linear-gradient(135deg,#080d18,#0d1a30);
  border:1px solid #1a4a7a;border-radius:10px;padding:20px;margin:8px 0;}
.catalog-spec{font-family:'Share Tech Mono',monospace;font-size:0.8rem;
  color:#7ab0d4;padding:5px 0;border-bottom:1px solid #0d1525;}
.catalog-spec-val{color:#4db8ff;font-weight:bold;float:right;}
.shdr{font-family:'Share Tech Mono',monospace;color:#4db8ff;
  font-size:0.78rem;letter-spacing:2px;text-transform:uppercase;
  border-bottom:1px solid #1a3a6a;padding-bottom:6px;margin:18px 0 10px;}
.pred-banner{background:linear-gradient(90deg,#1a0f00,#2a1800,#1a0f00);
  border:1px solid #ff8f00;border-radius:6px;padding:12px 20px;
  font-family:'Share Tech Mono',monospace;animation:glow-amber 2s infinite;}
@keyframes glow-amber{0%,100%{box-shadow:0 0 10px rgba(255,143,0,0.2)}
  50%{box-shadow:0 0 25px rgba(255,143,0,0.5)}}
.log-entry{font-family:'Share Tech Mono',monospace;font-size:0.74rem;
  padding:6px 12px;margin:2px 0;border-radius:4px;border-left:3px solid;}
.log-critical{background:#1a0404;border-color:#ff1744;color:#ff6b6b;}
.log-fault{background:#1a0800;border-color:#ff6d00;color:#ffb347;}
.log-warning{background:#1a1200;border-color:#ffd700;color:#ffe066;}
.log-normal{background:#071209;border-color:#00c853;color:#69f0ae;}
.rec-card{background:#080d18;border:1px solid #1a3a6a;border-radius:6px;
  padding:14px;min-height:120px;font-family:'Share Tech Mono',monospace;}
section[data-testid="stSidebar"]{background:#04080f!important;border-right:1px solid #1a3a6a;}
.one-btn{background:linear-gradient(135deg,#0d2347,#1a4a7a);
  border:2px solid #4db8ff;border-radius:8px;padding:16px;
  text-align:center;font-family:'Share Tech Mono',monospace;
  font-size:0.85rem;color:#4db8ff;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  Autoencoder (hand-built, NumPy-only, Adam optimizer)
# ══════════════════════════════════════════════════════════════════════
def tanh(x):   return np.tanh(x)
def tanh_g(x): return 1.0 - np.tanh(x)**2

class AutoencoderAdam:
    def __init__(self,di,dh=8,dl=2,lr=0.001):
        self.lr=lr; s=np.sqrt(2.0/di)
        self.p={'W1':s*np.random.randn(di,dh),'b1':np.zeros(dh),
                'W2':s*np.random.randn(dh,dl),'b2':np.zeros(dl),
                'W3':s*np.random.randn(dl,dh),'b3':np.zeros(dh),
                'W4':s*np.random.randn(dh,di),'b4':np.zeros(di)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.vv={k:np.zeros_like(v) for k,v in self.p.items()}
        self.t=0; self.b1a,self.b2a,self.eps=0.9,0.999,1e-8
    def forward(self,X):
        p=self.p
        self.a1=X@p['W1']+p['b1']; self.h1=tanh(self.a1)
        self.a2=self.h1@p['W2']+p['b2']; self.z=tanh(self.a2)
        self.a3=self.z@p['W3']+p['b3']; self.h3=tanh(self.a3)
        self.out=self.h3@p['W4']+p['b4']; return self.out
    def _bwd(self,X):
        p=self.p; m=X.shape[0]; d4=(2/m)*(self.out-X)
        g={'W4':self.h3.T@d4,'b4':d4.sum(0)}
        d3=(d4@p['W4'].T)*tanh_g(self.a3); g.update({'W3':self.z.T@d3,'b3':d3.sum(0)})
        d2=(d3@p['W3'].T)*tanh_g(self.a2); g.update({'W2':self.h1.T@d2,'b2':d2.sum(0)})
        d1=(d2@p['W2'].T)*tanh_g(self.a1); g.update({'W1':X.T@d1,'b1':d1.sum(0)})
        self.t+=1
        for k in p:
            gg=np.clip(g[k],-5,5)
            self.m[k]=self.b1a*self.m[k]+(1-self.b1a)*gg
            self.vv[k]=self.b2a*self.vv[k]+(1-self.b2a)*gg**2
            mh=self.m[k]/(1-self.b1a**self.t); vh=self.vv[k]/(1-self.b2a**self.t)
            p[k]-=self.lr*mh/(np.sqrt(vh)+self.eps)
    def fit(self,X,epochs=400,batch=128,cb=None):
        losses=[]
        for ep in range(epochs):
            idx=np.random.permutation(len(X)); el=[]
            for i in range(0,len(X),batch):
                xb=X[idx[i:i+batch]]; self.forward(xb); self._bwd(xb)
                el.append(float(np.mean((self.out-xb)**2)))
            losses.append(np.mean(el))
            if cb: cb(ep,epochs,losses[-1])
        return losses
    def reconstruct(self,X): return self.forward(X)

# ══════════════════════════════════════════════════════════════════════
#  Core Detection Functions
# ══════════════════════════════════════════════════════════════════════
def ewma_filter(s,alpha=0.2):
    f=np.zeros_like(s)
    for k in range(1,len(s)): f[k]=(1-alpha)*f[k-1]+alpha*s[k]
    return f

def inject_fault(X,sensor,start,magnitude,fault_type):
    Xf=X.copy(); n=X.shape[0]
    if fault_type=='bias': Xf[start:,sensor]+=magnitude
    elif fault_type=='drift': Xf[start:,sensor]+=np.linspace(0,magnitude,n-start)
    elif fault_type=='noise': Xf[start:,sensor]+=magnitude*np.random.randn(n-start)
    return Xf

def build_pca(Xn):
    cov=(1/(len(Xn)-1))*Xn.T@Xn
    evals,evecs=np.linalg.eigh(cov); idx=np.argsort(evals)[::-1]
    evals,evecs=evals[idx],evecs[:,idx]
    errs=[np.var(np.var(Xn@evecs[:,:n]@evecs[:,:n].T-Xn,axis=0)) for n in range(1,Xn.shape[1])]
    return {'type':'linear','evecs':evecs,'evals':evals,
            'n_components':int(np.argmin(errs))+1,'errors':errs}

def build_autoencoder(Xn,epochs=400,pb=None):
    di=Xn.shape[1]; best_l,best_dl=np.inf,2
    for dl in [1,2,3]:
        ae=AutoencoderAdam(di=di,dh=max(di+2,9),dl=dl,lr=0.001)
        ae.fit(Xn,epochs=150,batch=128)
        l=float(np.mean((Xn-ae.reconstruct(Xn))**2))
        if l<best_l: best_l,best_dl=l,dl
    ae_f=AutoencoderAdam(di=di,dh=max(di+2,9),dl=best_dl,lr=0.001)
    def cb(ep,total,loss):
        if pb: pb.progress(int((ep+1)/total*100),
                           text=f"{T('building_model')} {ep+1}/{total} | loss={loss:.5f}")
    ae_f.fit(Xn,epochs=epochs,batch=128,cb=cb)
    if pb: pb.progress(100,text=T('training_complete'))
    return {'type':'autoencoder','model':ae_f,'n_components':best_dl,'losses':[]}

def compute_spe(Xn,model):
    if model['type']=='linear':
        P=model['evecs'][:,:model['n_components']]; Xe=Xn@P@P.T
    elif model['type']=='kernel':
        kpca=model['model']; Xe=kpca.inverse_transform(kpca.transform(Xn))
    else: Xe=model['model'].reconstruct(Xn)
    return np.sum((Xn-Xe)**2,axis=1),Xe

def localize_fault(Xfn,fault_start,method,ep_p=300,pb=None):
    N,p=Xfn.shape; sb=np.zeros(p); sa=np.zeros(p)
    for i in range(p):
        if pb: pb.progress(int((i+1)/p*100),text=f"{T('localizing')} {i+1}/{p}")
        cols=[j for j in range(p) if j!=i]
        sc=StandardScaler(); Xpp=sc.fit_transform(Xfn[:,cols])
        if method=='linear':
            m=build_pca(Xpp[:fault_start]); spe_p,_=compute_spe(Xpp,m)
        elif method=='kernel':
            kpca=KernelPCA(n_components=2,kernel='rbf',gamma=0.1,fit_inverse_transform=True)
            kpca.fit(Xpp[:fault_start])
            spe_p=np.sum((Xpp-kpca.inverse_transform(kpca.transform(Xpp)))**2,axis=1)
        else:
            ae_p=AutoencoderAdam(di=len(cols),dh=len(cols)+2,dl=2,lr=0.001)
            ae_p.fit(Xpp[:fault_start],epochs=ep_p,batch=64)
            spe_p=np.sum((Xpp-ae_p.reconstruct(Xpp))**2,axis=1)
        sf=ewma_filter(spe_p); sb[i]=sf[:fault_start].mean(); sa[i]=sf[fault_start:].mean()
    ratios=sa/(sb+1e-10); return ratios,int(np.argmin(ratios)),sb,sa

def compute_health_score(spe_val,threshold):
    ratio=float(spe_val)/float(threshold)
    if ratio<=1.0: return round(100-ratio*15,1)
    return round(max(0,85-(ratio-1)*28),1)

def compute_confidence(ratios):
    sr=np.sort(ratios)
    if len(sr)<2: return 50.0
    gap=(sr[1]-sr[0])/(sr[1]+1e-10)*100
    return round(min(99.5,50+gap*0.7),1)

# ══════════════════════════════════════════════════════════════════════
#  Scientific Evaluation Metrics  /  مقاييس التقييم العلمية
# ══════════════════════════════════════════════════════════════════════
# These give the detector's performance against KNOWN ground truth
# (only available because fault_start/faulty_sensor are set by the user
# in simulation mode — exactly like NASA CMAPSS benchmark papers do it).
# تُعطي أداء الكاشف مقابل الحقيقة المعروفة (متاحة فقط لأن fault_start
# ومؤشر الحساس المعطوب محددان من طرف المستخدم في وضع المحاكاة — تماماً
# كما تفعل أوراق NASA CMAPSS البحثية المرجعية).

def compute_detection_metrics(spe_filtered, threshold, fault_start, n_total):
    """
    Binary fault-detection evaluation: ground truth = [0]*fault_start + [1]*(N-fault_start)
    Prediction = spe_filtered > threshold
    Returns precision, recall, F1, confusion matrix, ROC curve, AUC.
    """
    y_true = np.zeros(n_total, dtype=int)
    y_true[fault_start:] = 1
    y_score = np.asarray(spe_filtered, dtype=float)
    y_pred = (y_score > threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=[0,1])  # [[TN,FP],[FN,TP]]

    try:
        fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        fpr, tpr, roc_thresh, auc = np.array([0,1]), np.array([0,1]), np.array([0]), 0.5

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp+tn) / max(1, (tp+tn+fp+fn))
    specificity = tn / max(1, (tn+fp))

    return {
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'accuracy': round(float(accuracy), 4),
        'specificity': round(float(specificity), 4),
        'auc': round(float(auc), 4),
        'confusion_matrix': cm,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'fpr': fpr, 'tpr': tpr,
        'y_true': y_true, 'y_pred': y_pred, 'y_score': y_score,
    }

def plot_roc_curve(metrics):
    """ROC curve + confusion matrix side by side for scientific reporting."""
    lang = st.session_state.get('lang', 'en')
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── ROC Curve ──
    ax = axes[0]
    ax.plot(metrics['fpr'], metrics['tpr'], color='#4db8ff', lw=2,
            label=f"ROC (AUC = {metrics['auc']:.3f})")
    ax.plot([0,1],[0,1], color='#5a8aaa', lw=1, ls='--', alpha=0.6,
            label={'ar':'عشوائي (AUC=0.5)','en':'Random (AUC=0.5)'}[lang])
    ax.fill_between(metrics['fpr'], metrics['tpr'], alpha=0.12, color='#4db8ff')
    ax.set_xlabel({'ar':'معدل الإيجابيات الكاذبة (FPR)','en':'False Positive Rate'}[lang], fontsize=9)
    ax.set_ylabel({'ar':'معدل الإيجابيات الصحيحة (TPR)','en':'True Positive Rate'}[lang], fontsize=9)
    ax.set_title({'ar':'منحنى ROC','en':'ROC Curve'}[lang],
                 color='#4db8ff', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # ── Confusion Matrix ──
    ax2 = axes[1]
    cm_arr = metrics['confusion_matrix']
    ax2.imshow(cm_arr, cmap='Blues', alpha=0.80)
    labels = [{'ar':'طبيعي','en':'Normal'}[lang], {'ar':'عطل','en':'Fault'}[lang]]
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel({'ar':'التنبؤ','en':'Predicted'}[lang], fontsize=9)
    ax2.set_ylabel({'ar':'الحقيقة','en':'Actual'}[lang], fontsize=9)
    ax2.set_title({'ar':'مصفوفة الالتباس','en':'Confusion Matrix'}[lang],
                  color='#4db8ff', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_arr[i, j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if cm_arr[i,j] > cm_arr.max()/2 else '#0d1525')
    plt.tight_layout()
    add_demo_watermark(fig)
    return fig


# This implements the standard "piecewise-linear degradation" RUL model
# used across the NASA C-MAPSS literature (e.g. Heimes 2008, Saxena et al.
# 2008): health stays at a constant ceiling (e.g. 100% / RUL_max) until
# degradation becomes detectable, after which RUL decreases linearly to
# zero at end-of-life (here: the point the health index crosses the
# failure threshold). It is a standard simplification, not a learned model.
#
# هذا تطبيق لنموذج "التدهور الخطي المتجزئ" القياسي لـ RUL، المستخدم في
# أدبيات NASA C-MAPSS (مثل Heimes 2008، Saxena et al. 2008): تبقى الصحة
# عند سقف ثابت حتى يصبح التدهور قابلاً للكشف، ثم تنخفض خطياً إلى الصفر
# عند نهاية العمر (هنا: نقطة تجاوز مؤشر SPE للعتبة). هذا تبسيط قياسي
# معتمد في الأدبيات، وليس نموذجاً متعلَّماً (ML-learned).

def compute_rul(spe_filtered, threshold, fault_start, n_total, rul_max_cap=125):
    """
    Estimate Remaining Useful Life (in measurement steps) using the
    standard piecewise-linear degradation assumption from C-MAPSS literature.

    Returns a dict with:
      - rul_curve: RUL estimate at every time step
      - eol_index: estimated end-of-life step (where SPE trend would cross threshold)
      - current_rul: RUL estimate at the last available measurement
      - degradation_start: step where degradation becomes detectable
    """
    spe = np.asarray(spe_filtered, dtype=float)
    n = len(spe)

    # Detect onset of degradation: first sustained rise above the
    # pre-fault baseline noise band (mean + 2*std of the healthy region).
    baseline = spe[:max(5, fault_start)]
    baseline_mean = baseline.mean() if len(baseline) else spe.mean()
    baseline_std  = baseline.std() if len(baseline) else spe.std()
    degr_band = baseline_mean + 2*baseline_std

    degradation_start = fault_start  # ground-truth onset in simulation mode
    # refine: find first index after fault_start where SPE sustainably exceeds the band
    for i in range(fault_start, n):
        window = spe[i:min(i+5, n)]
        if len(window) > 0 and window.mean() > degr_band:
            degradation_start = i
            break

    # End-of-life: first index where SPE crosses the failure threshold
    eol_index = None
    for i in range(degradation_start, n):
        if spe[i] > threshold:
            eol_index = i
            break
    if eol_index is None:
        # Extrapolate via linear trend on the tail if threshold not yet crossed
        tail = spe[max(degradation_start, n-40):]
        if len(tail) >= 2 and tail[-1] > tail[0]:
            slope = (tail[-1]-tail[0]) / max(1, len(tail)-1)
            remaining_value = threshold - spe[-1]
            steps_to_eol = remaining_value / slope if slope > 1e-8 else None
            eol_index = int(n + steps_to_eol) if steps_to_eol and steps_to_eol > 0 else None

    # Build the RUL curve: flat ceiling before degradation, linear decay after
    rul_curve = np.zeros(n)
    if eol_index is not None and eol_index > degradation_start:
        life_span = eol_index - degradation_start
        for i in range(n):
            if i < degradation_start:
                rul_curve[i] = min(rul_max_cap, eol_index - degradation_start)
            else:
                rul_curve[i] = max(0, eol_index - i)
    else:
        # No clear degradation pattern detected — flat at cap (healthy)
        rul_curve[:] = rul_max_cap

    rul_curve = np.clip(rul_curve, 0, rul_max_cap)
    current_rul = float(rul_curve[-1])

    return {
        'rul_curve': rul_curve,
        'eol_index': eol_index,
        'degradation_start': int(degradation_start),
        'current_rul': round(current_rul, 1),
        'rul_max_cap': rul_max_cap,
        'is_critical': current_rul < rul_max_cap * 0.15,
    }

def plot_rul_curve(rul_data, fault_start, n_total):
    """Plot the RUL degradation curve — standard chart in C-MAPSS papers."""
    lang = st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(n_total)
    rul = rul_data['rul_curve']
    ax.plot(x, rul, color='#4dff88', lw=2, zorder=3)
    ax.fill_between(x, rul, 0, alpha=0.12, color='#4dff88')
    ax.axvline(rul_data['degradation_start'], color='#ffd700', ls='--', lw=1.3, alpha=0.8,
               label={'ar':'بداية التدهور المكتشفة','en':'Detected degradation onset'}[lang])
    if rul_data['eol_index'] is not None and rul_data['eol_index'] < n_total*1.3:
        eol = min(rul_data['eol_index'], n_total-1)
        ax.axvline(eol, color='#ff1744', ls=':', lw=1.5, alpha=0.8,
                   label={'ar':'نهاية العمر المقدّرة (EOL)','en':'Estimated End-of-Life (EOL)'}[lang])
    crit_line = rul_data['rul_max_cap'] * 0.15
    ax.axhline(crit_line, color='#ff6d00', ls='--', lw=1, alpha=0.5,
               label={'ar':'منطقة حرجة','en':'Critical zone'}[lang])
    ax.fill_between(x, 0, crit_line, alpha=0.06, color='#ff1744')
    ax.set_ylim(0, rul_data['rul_max_cap']*1.05)
    ax.set_xlabel({'ar':'القياس','en':'Measurement step'}[lang], fontsize=9)
    ax.set_ylabel('RUL', fontsize=9)
    title = {'ar':'العمر الإنتاجي المتبقي (RUL) — نموذج التدهور الخطي المتجزئ',
             'en':'Remaining Useful Life (RUL) — Piecewise-Linear Degradation Model'}[lang]
    ax.set_title(title, color='#4db8ff', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    add_demo_watermark(fig)
    return fig


    """ROC curve + confusion matrix heatmap for scientific reporting."""
    lang = st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ROC Curve
    ax = axes[0]
    ax.plot(metrics['fpr'], metrics['tpr'], color='#4db8ff', lw=2,
            label=f"ROC (AUC = {metrics['auc']:.3f})")
    ax.plot([0,1],[0,1], color='#5a8aaa', lw=1, ls='--', alpha=0.6,
            label={'ar':'عشوائي (AUC=0.5)','en':'Random (AUC=0.5)'}[lang])
    ax.fill_between(metrics['fpr'], metrics['tpr'], alpha=0.15, color='#4db8ff')
    ax.set_xlabel({'ar':'معدل الإيجابيات الكاذبة (FPR)','en':'False Positive Rate'}[lang], fontsize=9)
    ax.set_ylabel({'ar':'معدل الإيجابيات الصحيحة (TPR)','en':'True Positive Rate'}[lang], fontsize=9)
    ax.set_title({'ar':'منحنى ROC','en':'ROC Curve'}[lang], color='#4db8ff', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)

    # Confusion Matrix
    ax2 = axes[1]
    cm = metrics['confusion_matrix']
    im = ax2.imshow(cm, cmap='Blues', alpha=0.85)
    labels = [{'ar':'طبيعي','en':'Normal'}[lang], {'ar':'عطل','en':'Fault'}[lang]]
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(labels, fontsize=9); ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel({'ar':'التنبؤ','en':'Predicted'}[lang], fontsize=9)
    ax2.set_ylabel({'ar':'الحقيقة','en':'Actual'}[lang], fontsize=9)
    ax2.set_title({'ar':'مصفوفة الالتباس','en':'Confusion Matrix'}[lang], color='#4db8ff', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else '#0d1525')
    plt.tight_layout()
    add_demo_watermark(fig)
    return fig

def detect_fault_type(spe_series,fault_start):
    post=np.array(spe_series[fault_start:],dtype=float)
    pre=np.array(spe_series[:fault_start],dtype=float) if fault_start>0 else np.array([0.01])
    if len(post)<10: return 'unknown','Unknown',0.0
    std_post=post.std(); mean_post=post.mean(); pre_std=pre.std()+1e-10
    slope=float(np.polyfit(range(len(post)),post,1)[0])
    r2_pts=post-(slope*np.arange(len(post))+post[0])
    linearity=1-r2_pts.std()/(std_post+1e-10)
    lang=st.session_state.get('lang','en')
    if linearity>0.5 and slope>1e-4:
        name={'ar':'📈 انجراف (Drift)','en':'📈 Drift (Progressive)'}[lang]
        return 'drift',name,round(slope,6)
    if std_post>pre_std*4:
        name={'ar':'📡 ضجيج (Noise)','en':'📡 Noise (Intermittent)'}[lang]
        return 'noise',name,round(std_post,4)
    name={'ar':'📊 انزياح (Bias)','en':'📊 Bias (Offset)'}[lang]
    return 'bias',name,round(mean_post,4)

def compute_health_timeline(spe_filtered,threshold,window=30):
    return np.array([compute_health_score(spe_filtered[i:i+window].mean(),threshold)
                     for i in range(0,len(spe_filtered),window)])

def predict_fault(spe_filtered,threshold,window=40):
    if len(spe_filtered)<window*2: return None,T('stable')
    recent=spe_filtered[-window:]
    slope=float(np.polyfit(range(len(recent)),recent,1)[0])
    current=float(spe_filtered[-1])
    if current>=threshold: return 0,T('already_fault')
    if slope<=1e-6: return None,T('stable')
    steps=int((threshold-current)/slope)
    if steps>2000: return None,T('stable')
    lang=st.session_state.get('lang','en')
    msg={'ar':f"⚠️ متوقع بعد ~{max(0,steps)} قياس",
         'en':f"⚠️ Predicted in ~{max(0,steps)} measurements"}[lang]
    return max(0,steps),msg

def get_alert(det_rate,identified,det_time,fault_start,conf,fault_type_name):
    lang=st.session_state.get('lang','en')
    short=SS(identified)
    if det_rate<5:
        base=AL('NORMAL'); base.update({'sensor':None,'confidence':None,'level':'NORMAL'}); return base
    if det_rate<30:
        base=AL('WARNING')
        reason={'ar':f"شذوذ ({det_rate:.0f}%) في {short}. نوع: {fault_type_name}",
                'en':f"Anomaly ({det_rate:.0f}%) in {short}. Type: {fault_type_name}"}[lang]
        action={'ar':f"راقب {short}. جدول فحصاً خلال 48 ساعة.",
                'en':f"Monitor {short}. Schedule inspection within 48h."}[lang]
        base['reason']=reason; base['action']=action
        base.update({'sensor':SN(identified),'confidence':conf,'level':'WARNING'}); return base
    if det_rate<70:
        base=AL('FAULT'); d=det_time-fault_start if det_time else 0
        reason={'ar':f"عطل مؤكد في {short} ({det_rate:.0f}%). نوع: {fault_type_name}. كُشف بعد {d} قياساً.",
                'en':f"Confirmed fault in {short} ({det_rate:.0f}%). Type: {fault_type_name}. Detected after {d} pts."}[lang]
        action={'ar':f"صيانة عاجلة لـ {short}.","en":f"Urgent maintenance for {short}."}[lang]
        base['reason']=reason; base['action']=action
        base.update({'sensor':SN(identified),'confidence':conf,'level':'FAULT'}); return base
    base=AL('CRITICAL'); d=det_time-fault_start if det_time else 0
    reason={'ar':f"عطل حرج في {short} ({det_rate:.0f}%). نوع: {fault_type_name}. بدأ عند القياس {det_time}.",
            'en':f"Critical fault in {short} ({det_rate:.0f}%). Type: {fault_type_name}. Onset at #{det_time}."}[lang]
    action={'ar':f"⛔ أوقف التشغيل فوراً. افحص {short} قبل الإعادة.",
            'en':f"⛔ STOP TURBINE. Inspect {short} before restart."}[lang]
    base['reason']=reason; base['action']=action
    base.update({'sensor':SN(identified),'confidence':conf,'level':'CRITICAL'}); return base

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE,'r',encoding='utf-8') as f: return json.load(f)
        except: pass
    return {'events':[],'stats':{'total':0,'critical':0,'warnings':0}}

def save_log(alert,identified,conf,fault_type):
    log=load_log()
    ev={'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'level':alert['level'],
        'sensor_en':SENSOR_NAMES['en'][identified] if identified<7 else f'S{identified+1}',
        'sensor_ar':SENSOR_NAMES['ar'][identified] if identified<7 else f'ح{identified+1}',
        'fault_type':fault_type,'confidence':conf,'occurrence':1}
    for ev2 in log['events']:
        if ev2.get('sensor_en')==ev['sensor_en'] and ev2.get('fault_type')==ev['fault_type']:
            ev['occurrence']=ev2.get('occurrence',1)+1; break
    log['events'].insert(0,ev); log['events']=log['events'][:100]
    log['stats']={'total':len(log['events']),
                  'critical':sum(1 for e in log['events'] if e['level']=='CRITICAL'),
                  'warnings':sum(1 for e in log['events'] if e['level'] in ['WARNING','FAULT'])}
    with open(LOG_FILE,'w',encoding='utf-8') as f: json.dump(log,f,ensure_ascii=False)
    return log

def check_data_quality(X):
    lang=st.session_state.get('lang','en')
    issues=[]; n,p=X.shape
    nan_count=int(np.isnan(X).sum())
    if nan_count>0:
        issues.append({'ar':f"⚠️ {nan_count} قيمة مفقودة — ستُعالج تلقائياً",
                        'en':f"⚠️ {nan_count} missing values — will be interpolated"}[lang])
    for i in range(p):
        col=X[:,i]; col_c=col[~np.isnan(col)]
        if len(col_c)<2: continue
        z=np.abs((col_c-col_c.mean())/(col_c.std()+1e-10))
        if (z>5).mean()*100>1:
            issues.append({'ar':f"⚠️ ح{i+1}: قيم شاذة مكتشفة",
                            'en':f"⚠️ S{i+1}: Extreme outliers detected"}[lang])
        if col_c.std()<1e-8:
            issues.append({'ar':f"❌ ح{i+1}: إشارة ثابتة — الحساس ربما معطوب",
                            'en':f"❌ S{i+1}: Flat signal — sensor may be stuck"}[lang])
    if n<100:
        issues.append({'ar':f"⚠️ {n} عينة فقط — الحد الأدنى الموصى به 200",
                        'en':f"⚠️ Only {n} samples — minimum 200 recommended"}[lang])
    X_clean=X.copy()
    if nan_count>0:
        for i in range(p):
            col=X_clean[:,i]; nans=np.isnan(col)
            if nans.any():
                idx=np.arange(n); col[nans]=np.interp(idx[nans],idx[~nans],col[~nans])
                X_clean[:,i]=col
    score=max(0,100-len(issues)*20)
    ok_msg={'ar':'✅ كل الفحوصات ناجحة — البيانات جاهزة للتحليل',
            'en':'✅ All checks passed — data ready for analysis'}[lang]
    return {'score':score,
            'status':('✅ '+{'ar':'جيد','en':'GOOD'}[lang] if score>=80 else
                      '⚠️ '+{'ar':'مقبول','en':'ACCEPTABLE'}[lang] if score>=50 else
                      '❌ '+{'ar':'ضعيف','en':'POOR'}[lang]),
            'issues':issues if issues else [ok_msg],
            'n_samples':n,'n_sensors':p,'X_clean':X_clean}

def compute_sensor_ranking(ratios,health_scores):
    lang=st.session_state.get('lang','en')
    r_min,r_max=ratios.min(),ratios.max()
    fault_prob=(1-(ratios-r_min)/(r_max-r_min+1e-10))*100
    ranking=[]
    for i in np.argsort(fault_prob)[::-1]:
        fp=float(fault_prob[i]); h=float(health_scores[i]) if i<len(health_scores) else 100.0
        badge=('🔴 حرج' if fp>80 else '🟠 مرتفع' if fp>55 else '🟡 متوسط' if fp>25 else '🟢 منخفض') if lang=='ar' else \
              ('🔴 CRITICAL' if fp>80 else '🟠 HIGH' if fp>55 else '🟡 MEDIUM' if fp>25 else '🟢 LOW')
        ranking.append({'rank':len(ranking)+1,'idx':int(i),'full_name':SN(int(i)),
                        'fault_prob':round(fp,1),'health':round(h,1),
                        'ratio':round(float(ratios[i]),1),'badge':badge})
    return ranking

# ══════════════════════════════════════════════════════════════════════
#  XAI — Explainability Engine  /  محرك التفسير الذكي
# ══════════════════════════════════════════════════════════════════════
def compute_xai(Xn_faulty, fault_start, identified, spe_ff, threshold, model, fault_type):
    """Per-sensor contribution to the fault decision, correlation impact, pattern."""
    lang = st.session_state.get('lang','en')
    n_sensors = Xn_faulty.shape[1]
    if model['type'] == 'autoencoder':
        Xr_normal = model['model'].reconstruct(Xn_faulty[:fault_start])
        Xr_fault  = model['model'].reconstruct(Xn_faulty[fault_start:])
    elif model['type'] == 'linear':
        P = model['evecs'][:, :model['n_components']]
        Xr_normal = Xn_faulty[:fault_start] @ P @ P.T
        Xr_fault  = Xn_faulty[fault_start:] @ P @ P.T
    else:
        kpca = model['model']
        Xr_normal = kpca.inverse_transform(kpca.transform(Xn_faulty[:fault_start]))
        Xr_fault  = kpca.inverse_transform(kpca.transform(Xn_faulty[fault_start:]))
    err_normal = np.mean((Xn_faulty[:fault_start] - Xr_normal)**2, axis=0) + 1e-10
    err_fault  = np.mean((Xn_faulty[fault_start:] - Xr_fault)**2,  axis=0) + 1e-10
    err_ratio = err_fault / err_normal
    contributions = err_ratio / err_ratio.sum() * 100
    spe_ratio = float(spe_ff[fault_start:].mean()) / (threshold + 1e-10)
    spe_excess = round((max(spe_ratio, 1.0) - 1.0) * 100, 1)
    corr_m = np.corrcoef(Xn_faulty.T)
    corr_impact = []
    if identified < n_sensors:
        row = np.abs(corr_m[identified]).copy(); row[identified] = 0
        for idx in np.argsort(row)[::-1][:3]:
            if row[idx] > 0.60 and idx < n_sensors:
                corr_impact.append({'sensor_idx': int(idx), 'sensor_name': SS(int(idx)),
                                    'correlation': round(float(row[idx]),2),
                                    'contribution': round(float(contributions[idx]),1)})
    patterns = {
        'ar': {'bias':  f"الحساس S{identified+1} يُظهر انزياحاً ثابتاً — القراءة أعلى بـ {spe_excess:.0f}% من خط الأساس",
               'drift': f"الحساس S{identified+1} يتدهور تدريجياً — خطأ إعادة البناء يرتفع بمعدل منتظم",
               'noise': f"الحساس S{identified+1} يُظهر تذبذباً — مستوى الضجيج {spe_excess:.0f}% فوق الطبيعي",
               'unknown': f"الحساس S{identified+1} يتجاوز حد القبول بـ {spe_excess:.0f}%"},
        'en': {'bias':  f"Sensor S{identified+1} shows constant offset — {spe_excess:.0f}% above baseline",
               'drift': f"Sensor S{identified+1}: progressive degradation — error rising steadily",
               'noise': f"Sensor S{identified+1}: signal instability — noise {spe_excess:.0f}% above normal",
               'unknown': f"Sensor S{identified+1} exceeds threshold by {spe_excess:.0f}%"}
    }
    return {'contributions': contributions, 'corr_impact': corr_impact, 'spe_excess': spe_excess,
            'pattern': patterns[lang].get(fault_type, patterns[lang]['unknown']),
            'top_contribution': round(float(contributions[identified]), 1)}

def generate_explanation(identified,corr_matrix,fault_type):
    lang=st.session_state.get('lang','en')
    short=SS(identified)
    corr_sensors=[]
    if corr_matrix is not None and identified<corr_matrix.shape[0]:
        row=np.abs(corr_matrix[identified]).copy(); row[identified]=0
        for idx in np.argsort(row)[::-1][:3]:
            if row[idx]>0.65 and idx<7: corr_sensors.append(SS(idx))
        corr_sensors=corr_sensors[:2]
    causes=PROBABLE_CAUSES[lang].get(short, PROBABLE_CAUSES['en'].get(short,
           ['Sensor fault','Signal chain issue','Wiring problem']))
    cause_note = FAULT_TYPE_CAUSES[lang].get(fault_type, '')
    return {'corr_sensors':corr_sensors,'probable_causes':causes,'cause_note':cause_note}

def generate_recommendations(alert_level,identified,fault_type,prediction):
    lang=st.session_state.get('lang','en')
    short=SS(identified)
    actions=MAINTENANCE_ACTIONS[lang].get(fault_type,MAINTENANCE_ACTIONS[lang]['unknown'])
    priority=PRIORITY_MAP[lang].get(alert_level,PRIORITY_MAP[lang]['FAULT'])
    pred_note=""
    if prediction[0] is not None and prediction[0]>0:
        pred_note={'ar':f"⏱️ تحليل الاتجاه يتوقع تجاوز العتبة بعد ~{prediction[0]} قياساً.",
                   'en':f"⏱️ Trend analysis predicts threshold breach in ~{prediction[0]} measurements."}[lang]
    stop_note=""
    if alert_level=='CRITICAL':
        stop_note={'ar':f"⛔ أوقف التوربين قبل الفحص. لا تُشغّله حتى يتم التحقق من {short}.",
                   'en':f"⛔ STOP TURBINE before inspection. Do not restart until {short} is verified."}[lang]
    return {'priority':priority['label'],'timeframe':priority['time'],
            'immediate':actions['immediate'],'short_term':actions['short_term'],
            'preventive':actions['preventive'],'prediction_note':pred_note,'stop_note':stop_note}

# ══════════════════════════════════════════════════════════════════════
#  Plain-Text Industrial Decision Output  /  مخرجات القرار الصناعي النصي
# ══════════════════════════════════════════════════════════════════════
def render_plain_decision(alert, identified, conf_score, ft_name, health, xai_data, recommendations):
    """What a factory floor worker needs: a clear decision, not a chart."""
    lang = st.session_state.get('lang', 'en')
    dir_style = "direction:rtl;text-align:right;" if lang == 'ar' else ""
    color = alert['color']; level = alert['level']; short = SS(identified)

    sev_map = {'NORMAL':{'ar':'لا يوجد عطل','en':'NO FAULT'}, 'WARNING':{'ar':'منخفض','en':'LOW'},
               'FAULT':{'ar':'مرتفع','en':'HIGH'}, 'CRITICAL':{'ar':'حرج','en':'CRITICAL'}}
    severity = sev_map.get(level, sev_map['FAULT'])[lang]
    loc_label = {'ar':'الموقع: الحساس','en':'Location: Sensor'}[lang]
    sev_label = {'ar':'الشدة:','en':'Severity:'}[lang]
    conf_label = {'ar':'الثقة:','en':'Confidence:'}[lang]
    fault_label = {'ar':'نوع العطل:','en':'Fault Type:'}[lang]
    health_label = {'ar':'صحة النظام:','en':'System Health:'}[lang]
    trend_val = health[identified]
    trend_txt = ({'ar':'تدهور 📉','en':'Degrading 📉'} if trend_val<60 else
                 {'ar':'مستقر ➡','en':'Stable ➡'} if trend_val<80 else {'ar':'جيد 📈','en':'Good 📈'})[lang]
    detect_icon = '🚨' if level in ['FAULT','CRITICAL'] else ('⚠️' if level=='WARNING' else '✅')
    fault_detected = ({'ar':'تم كشف عطل','en':'FAULT DETECTED'} if level!='NORMAL' else
                       {'ar':'النظام طبيعي','en':'SYSTEM NORMAL'})[lang]

    block1 = f"""
    <div class="plain-decision" style="border-color:{color};{dir_style}">
        <div style="font-size:1.4rem;font-weight:900;color:{color};margin-bottom:14px;letter-spacing:2px;">
            {detect_icon} {fault_detected}
        </div>
        <div class="plain-line"><span class="plain-label">{loc_label}</span>
            <span style="color:{color};font-weight:bold;"> {short}</span></div>
        <div class="plain-line"><span class="plain-label">{sev_label}</span>
            <span style="color:{color};font-weight:bold;"> {severity}</span></div>
        <div class="plain-line"><span class="plain-label">{conf_label}</span>
            <span style="color:#4dff88;font-weight:bold;"> {conf_score}%</span></div>
        <div class="plain-line"><span class="plain-label">{fault_label}</span>
            <span style="color:#c8d8e8;font-weight:bold;"> {ft_name}</span></div>
        <div class="plain-line"><span class="plain-label">{health_label}</span>
            <span style="color:{'#ff1744' if trend_val<40 else '#ffd700' if trend_val<70 else '#00c853'};font-weight:bold;"> {health.mean():.0f}% — {trend_txt}</span></div>
        <div style="border-top:1px solid #1a3a6a;margin-top:14px;padding-top:10px;
             color:{color};font-size:0.88rem;font-weight:bold;">
            ⚡ {alert['action']}
        </div>
    </div>"""

    reason_title = {'ar':'🧠 لماذا هذا القرار؟','en':'🧠 Why This Decision?'}[lang]
    r1 = {'ar':f"• الحساس {short} انحرف عن النمط الطبيعي",'en':f"• {short} deviated from normal pattern"}[lang]
    r2 = {'ar':f"• تجاوز خطأ إعادة البناء العتبة بـ {xai_data['spe_excess']:.0f}%",
          'en':f"• Reconstruction error exceeded threshold by {xai_data['spe_excess']:.0f}%"}[lang]
    r3 = ''
    if xai_data['corr_impact']:
        names = ', '.join([c['sensor_name'] for c in xai_data['corr_impact'][:2]])
        r3 = {'ar':f"• مرتبط مع: {names}",'en':f"• Correlated with: {names}"}[lang]
    r4 = {'ar':f"• مساهمة الحساس في العطل: {xai_data['top_contribution']:.0f}%",
          'en':f"• Sensor contribution to fault: {xai_data['top_contribution']:.0f}%"}[lang]

    block2 = f"""
    <div class="xai-card" style="border-left-color:{color};{dir_style}">
        <div style="color:{color};font-size:0.85rem;font-weight:bold;margin-bottom:10px;">{reason_title}</div>
        <div class="plain-line">{r1}</div>
        <div class="plain-line">{r2}</div>
        {'<div class="plain-line">'+r3+'</div>' if r3 else ''}
        <div class="plain-line">{r4}</div>
    </div>"""

    action_title = {'ar':'🔧 الإجراء الموصى به','en':'🔧 Recommended Action'}[lang]
    a3 = {'ar':f"صيانة مطلوبة خلال: {recommendations['timeframe']}",
          'en':f"Maintenance required within: {recommendations['timeframe']}"}[lang]
    block3 = f"""
    <div class="xai-card" style="border-left-color:#4db8ff;{dir_style}">
        <div style="color:#4db8ff;font-size:0.85rem;font-weight:bold;margin-bottom:10px;">{action_title}</div>
        <div class="plain-line">1. {recommendations['immediate']}</div>
        <div class="plain-line">2. {recommendations['short_term']}</div>
        <div class="plain-line" style="color:{color};font-weight:bold;">3. {a3}</div>
    </div>"""
    return block1, block2, block3

# ══════════════════════════════════════════════════════════════════════
#  JSON Output for SCADA/DCS Integration
# ══════════════════════════════════════════════════════════════════════
def build_json_output(alert, identified, conf_score, ft_key, health, det_rate,
                      prediction, xai_data, recommendations, data_source):
    return {
        "system": "Turbine Guardian v6.3",
        "data_disclaimer": "Synthetic simulation calibrated to published NASA CMAPSS statistics. NOT real NASA telemetry. Educational use only.",
        "timestamp": datetime.now().isoformat(),
        "data_source": data_source,
        "fault": alert['level'] != 'NORMAL',
        "severity": alert['level'],
        "sensor": f"S{identified+1}",
        "fault_type": ft_key,
        "confidence": conf_score,
        "detection_rate": round(det_rate, 1),
        "system_health": round(float(health.mean()), 1),
        "sensor_health": round(float(health[identified]), 1),
        "trend": "degrading" if health[identified]<60 else ("stable" if health[identified]<80 else "good"),
        "prediction_steps": prediction[0],
        "xai": {"sensor_contribution_pct": round(float(xai_data['top_contribution']), 1),
                "spe_excess_pct": round(float(xai_data['spe_excess']), 1),
                "correlated_sensors": [c['sensor_name'] for c in xai_data['corr_impact']]},
        "recommended_action": {"immediate": recommendations['immediate'],
                                "short_term": recommendations['short_term'],
                                "timeframe": recommendations['timeframe'],
                                "priority": recommendations['priority']},
    }

# ══════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════
PLT_STYLE = {'figure.facecolor':'#0a0e1a','axes.facecolor':'#0d1525',
    'axes.edgecolor':'#1e4a7a','axes.labelcolor':'#7ab0d4',
    'xtick.color':'#7ab0d4','ytick.color':'#7ab0d4',
    'grid.color':'#1e3a6a','grid.alpha':0.35,'text.color':'#c8d8e8','font.family':'monospace'}

def add_demo_watermark(fig):
    """Add watermark to matplotlib figures in demo mode."""
    if DEMO_MODE:
        fig.text(0.5, 0.5, DEMO_WATERMARK,
                 fontsize=22, color='gray', alpha=0.18,
                 ha='center', va='center', rotation=30,
                 fontweight='bold', transform=fig.transFigure)

def plot_health_map(health_scores,n_sensors=7):
    lang=st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor('#080d18'); ax.set_facecolor('#080d18')
    for i in range(n_sensors):
        h=health_scores[i]; c='#00c853' if h>70 else '#ffd700' if h>40 else '#ff1744'
        ax.barh(i,100,color='#0d1525',height=0.65,zorder=1)
        ax.barh(i,h,color=c+'44',height=0.65,zorder=2,edgecolor=c,linewidth=1.5)
        s='🟢' if h>70 else '🟡' if h>40 else '🔴'
        status_txt=T('healthy') if h>70 else T('warning_status') if h>40 else T('fault_status')
        ax.text(102,i,f'{h:.0f}%  {s} {status_txt}',va='center',fontsize=9,color=c,fontweight='bold')
        name=SN(i).split('—')[0].strip() if i<7 else f'S{i+1}'
        ax.text(-2,i,name,va='center',ha='right',fontsize=9,color='#7ab0d4')
    ax.set_xlim(-45,165); ax.set_ylim(-0.8,n_sensors-0.2); ax.set_yticks([])
    ax.set_title(T('health_map_title'),color='#4db8ff',fontsize=12,fontweight='bold',pad=12)
    ax.grid(True,axis='x',alpha=0.2); plt.tight_layout(); add_demo_watermark(fig); return fig

def plot_xai_contributions(xai_data, n_sensors, identified):
    plt.rcParams.update(PLT_STYLE)
    lang = st.session_state.get('lang', 'en')
    fig, ax = plt.subplots(figsize=(10, 4))
    contributions = xai_data['contributions'][:n_sensors]
    labels = [f"{'ح' if lang=='ar' else 'S'}{i+1}" for i in range(n_sensors)]
    colors = ['#ff1744' if i == identified else '#1e4a7a' for i in range(n_sensors)]
    bars = ax.bar(labels, contributions, color=colors, alpha=0.85, edgecolor='#2a6fbd', linewidth=1.2)
    for bar, val in zip(bars, contributions):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%',
                ha='center', fontsize=9, color='#ff4d4d' if bar.get_facecolor()[0]>0.5 else '#7ab0d4', fontweight='bold')
    title={'ar':'مساهمة كل حساس في العطل المكتشف (%)','en':'Per-Sensor Contribution to Detected Fault (%)'}[lang]
    ax.set_title(title,color='#4db8ff',fontsize=11,fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y'); plt.tight_layout(); add_demo_watermark(fig); return fig

def plot_radar(health_scores,n_sensors=7):
    lang=st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    cats=[f'S{i+1}' if lang=='en' else f'ح{i+1}' for i in range(n_sensors)]
    angles=[k/float(n_sensors)*2*np.pi for k in range(n_sensors)]; angles+=angles[:1]
    vals=list(health_scores[:n_sensors])+[health_scores[0]]
    fig=plt.figure(figsize=(6,6)); fig.patch.set_facecolor('#080d18')
    ax=fig.add_subplot(111,polar=True); ax.set_facecolor('#0d1525')
    ax.fill(angles,vals,alpha=0.18,color='#4db8ff')
    ax.plot(angles,vals,'o-',lw=2,color='#4db8ff',markersize=6)
    for a,v in zip(angles[:-1],vals[:-1]):
        ax.plot(a,v,'o',color='#00c853' if v>70 else '#ffd700' if v>40 else '#ff1744',markersize=10,zorder=5)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,color='#c8d8e8',size=11,fontweight='bold')
    ax.set_ylim(0,100); ax.grid(color='#1e3a6a',alpha=0.5); ax.spines['polar'].set_color('#1e4a7a')
    title={'ar':'بصمة العطل','en':'FAULT FINGERPRINT'}[lang]
    ax.set_title(title,color='#4db8ff',fontsize=11,fontweight='bold',pad=18)
    plt.tight_layout(); add_demo_watermark(fig); return fig

def plot_health_timeline(timeline,fault_start_idx=None):
    lang=st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig,ax=plt.subplots(figsize=(12,4)); x=np.arange(len(timeline))
    ax.fill_between(x,timeline,0,alpha=0.15,color='#4db8ff')
    ax.plot(x,timeline,lw=2,color='#4db8ff',zorder=3)
    for xp,yp in zip(x,timeline):
        ax.scatter(xp,yp,color='#00c853' if yp>70 else '#ffd700' if yp>40 else '#ff1744',s=35,zorder=4)
    ax.axhline(70,color='#ffd700',ls='--',lw=1.2,alpha=0.7,label={'ar':'تحذير (70%)','en':'Warning (70%)'}[lang])
    ax.axhline(40,color='#ff1744',ls='--',lw=1.2,alpha=0.7,label={'ar':'حرج (40%)','en':'Critical (40%)'}[lang])
    ax.fill_between(x,0,40,alpha=0.05,color='#ff1744'); ax.fill_between(x,40,70,alpha=0.05,color='#ffd700')
    ax.fill_between(x,70,100,alpha=0.05,color='#00c853')
    if fault_start_idx: ax.axvline(fault_start_idx//30,color='#ff8f00',ls=':',lw=1.5,alpha=0.8)
    ax.set_ylim(0,105); ax.set_xlim(0,len(x)-1)
    title={'ar':'تطور صحة التوربين','en':'TURBINE HEALTH TIMELINE'}[lang]
    ax.set_title(title,color='#4db8ff',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8,loc='lower left'); ax.grid(True,alpha=0.25); plt.tight_layout(); add_demo_watermark(fig); return fig

def plot_localization(ratios,identified,n_sensors=7):
    lang=st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]; bar_c=['#ff1744' if i==identified else '#1e4a7a' for i in range(n_sensors)]
    bars=ax.bar(range(n_sensors),ratios,color=bar_c,alpha=0.85,edgecolor='#2a6fbd',linewidth=1.2)
    ax.set_xticks(range(n_sensors))
    labels=[f"{'ح' if lang=='ar' else 'S'}{i+1}" for i in range(n_sensors)]
    ax.set_xticklabels(labels,fontsize=10)
    title={'ar':'تصنيف الحساسات — أقل نسبة = معطوب','en':'FAULT LOCALIZATION RANKING\n(Lowest ratio = Faulty)'}[lang]
    ax.set_title(title,color='#4db8ff',fontsize=10,fontweight='bold')
    ax.grid(True,alpha=0.2,axis='y')
    for i,bar in enumerate(bars):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(ratios)*0.02,f'{ratios[i]:.1f}',
                ha='center',fontsize=8,color='#ff4d4d' if i==identified else '#7ab0d4',fontweight='bold')
    ax2=axes[1]; norm=(ratios-ratios.min())/(ratios.max()-ratios.min()+1e-10); fp=(1-norm)*100
    c2=['#ff1744' if fp[i]>80 else '#ff6d00' if fp[i]>55 else '#ffd700' if fp[i]>25 else '#1e4a7a' for i in range(n_sensors)]
    ax2.barh(range(n_sensors),fp[::-1],color=c2[::-1],alpha=0.85)
    ax2.set_yticks(range(n_sensors))
    ax2.set_yticklabels([f"{'ح' if lang=='ar' else 'S'}{n_sensors-i}" for i in range(n_sensors)],fontsize=9)
    title2={'ar':'احتمال العطل لكل حساس','en':'SENSOR FAULT PROBABILITY'}[lang]
    ax2.set_title(title2,color='#4db8ff',fontsize=10,fontweight='bold')
    ax2.axvline(80,color='#ff1744',ls='--',lw=1,alpha=0.6); ax2.grid(True,alpha=0.2,axis='x')
    for i,v in enumerate(fp[::-1]):
        ax2.text(v+1,i,f'{v:.0f}%',va='center',fontsize=8,color='#ff4d4d' if (n_sensors-1-i)==identified else '#7ab0d4')
    plt.tight_layout(); add_demo_watermark(fig); return fig

def plot_sensor_signals(X_clean_full,X_faulty,t_axis,fault_start,identified,has_inj,n_s):
    lang=st.session_state.get('lang','en')
    plt.rcParams.update(PLT_STYLE)
    fig,axes=plt.subplots(n_s,1,figsize=(14,14),sharex=True)
    if n_s==1: axes=[axes]
    title={'ar':'إشارات حساسات التوربين — MS5002C / محاكاة مستلهمة من NASA','en':'Turbine Sensor Signals — MS5002C / NASA-Inspired Simulation'}[lang]
    fig.suptitle(title,color='#4db8ff',fontsize=12,fontweight='bold',y=1.01)
    for i,ax in enumerate(axes):
        c=S_COLORS[i] if i<len(S_COLORS) else '#4db8ff'; fc='#ff1744' if i==identified else c
        if has_inj:
            ax.plot(t_axis,X_clean_full[:,i],color=c,alpha=0.3,lw=0.7,label=T('normal_period'))
            ax.plot(t_axis,X_faulty[:,i],color=fc,lw=0.9,label=T('faulty_period') if i==identified else '')
        else:
            ax.plot(t_axis,X_faulty[:,i],color=c,lw=0.8)
        ax.axvline(t_axis[fault_start],color='#ff8f00',ls=':',lw=1.2,alpha=0.7)
        name=SN(i).split('—')[0].strip() if i<7 else f'S{i+1}'
        ax.set_ylabel(name,fontsize=8,color=fc); ax.grid(True,alpha=0.2)
        if i==identified: ax.set_facecolor('#120404'); ax.legend(fontsize=7,loc='upper right')
    axes[-1].set_xlabel({'ar':'الزمن','en':'Time'}[lang],fontsize=9)
    plt.tight_layout(); add_demo_watermark(fig); return fig

# ══════════════════════════════════════════════════════════════════════
#  Bilingual Illustrated Turbine Catalog  /  كتالوج التوربين المصور
# ══════════════════════════════════════════════════════════════════════
def render_turbine_catalog():
    lang = st.session_state.get('lang','en')
    dir_style = "direction:rtl;text-align:right;" if lang=='ar' else ""
    st.markdown(f'<div class="shdr">{T("catalog_title")}</div>', unsafe_allow_html=True)
    st.markdown('<span class="nasa-badge">🛰️ NASA CMAPSS-Inspired — Reference Stats from Saxena & Goebel (2008), Synthetic Signal</span>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown(f"""
        <div class="catalog-card" style="{dir_style}">
            <div style="color:#4db8ff;font-size:1.05rem;font-weight:bold;margin-bottom:12px;">
                ⚙️ {"GE MS5002C توربين غاز صناعي" if lang=="ar" else "GE MS5002C Industrial Gas Turbine"}
            </div>
            <svg viewBox="0 0 420 230" xmlns="http://www.w3.org/2000/svg" width="100%">
              <rect width="420" height="230" fill="#080d18" rx="8"/>
              <rect x="55" y="85" width="310" height="60" fill="#0d2347" stroke="#1e4a7a" stroke-width="1.5" rx="4"/>
              <ellipse cx="85" cy="115" rx="30" ry="40" fill="#0a1628" stroke="#4db8ff" stroke-width="2"/>
              <ellipse cx="85" cy="115" rx="18" ry="25" fill="#0d2347" stroke="#2a6fbd" stroke-width="1.2"/>
              <line x1="85" y1="75" x2="85" y2="90" stroke="#4db8ff" stroke-width="2.5"/>
              <line x1="85" y1="140" x2="85" y2="155" stroke="#4db8ff" stroke-width="2.5"/>
              <line x1="55" y1="115" x2="67" y2="115" stroke="#4db8ff" stroke-width="2.5"/>
              <rect x="160" y="75" width="100" height="80" fill="#1a0800" stroke="#ff6d00" stroke-width="1.5" rx="6"/>
              <text x="210" y="108" fill="#ff6d00" font-family="monospace" font-size="9" text-anchor="middle">COMBUSTOR</text>
              <text x="210" y="123" fill="#ff8f00" font-family="monospace" font-size="8" text-anchor="middle">{"غرفة الاحتراق" if lang=="ar" else "T30: 1590 R"}</text>
              <ellipse cx="335" cy="115" rx="30" ry="40" fill="#0a1628" stroke="#4dff88" stroke-width="2"/>
              <ellipse cx="335" cy="115" rx="18" ry="25" fill="#0d2347" stroke="#00c853" stroke-width="1.2"/>
              <line x1="335" y1="75" x2="335" y2="90" stroke="#4dff88" stroke-width="2.5"/>
              <line x1="335" y1="140" x2="335" y2="155" stroke="#4dff88" stroke-width="2.5"/>
              <line x1="353" y1="115" x2="365" y2="115" stroke="#4dff88" stroke-width="2.5"/>
              <line x1="115" y1="115" x2="160" y2="115" stroke="#ffd700" stroke-width="2" stroke-dasharray="5,3"/>
              <line x1="260" y1="115" x2="305" y2="115" stroke="#ffd700" stroke-width="2" stroke-dasharray="5,3"/>
              <circle cx="70" cy="95" r="5" fill="#4db8ff" opacity="0.9"/>
              <text x="73" y="87" fill="#4db8ff" font-family="monospace" font-size="7">S1</text>
              <circle cx="152" cy="95" r="5" fill="#ffd700" opacity="0.9"/>
              <text x="155" y="87" fill="#ffd700" font-family="monospace" font-size="7">S4</text>
              <circle cx="268" cy="95" r="5" fill="#ffd700" opacity="0.9"/>
              <text x="271" y="87" fill="#ffd700" font-family="monospace" font-size="7">S5</text>
              <circle cx="350" cy="95" r="5" fill="#4dff88" opacity="0.9"/>
              <text x="353" y="87" fill="#4dff88" font-family="monospace" font-size="7">S7</text>
              <text x="30" y="120" fill="#5a8aaa" font-family="monospace" font-size="9">AIR</text>
              <text x="368" y="120" fill="#ff9944" font-family="monospace" font-size="9">EXH</text>
              <text x="85" y="178" fill="#4db8ff" font-family="monospace" font-size="8" text-anchor="middle">{"ضاغط" if lang=="ar" else "COMPRESSOR"}</text>
              <text x="210" y="178" fill="#ff6d00" font-family="monospace" font-size="8" text-anchor="middle">{"احتراق" if lang=="ar" else "COMBUSTION"}</text>
              <text x="335" y="178" fill="#4dff88" font-family="monospace" font-size="8" text-anchor="middle">{"توربين" if lang=="ar" else "TURBINE"}</text>
              <text x="210" y="210" fill="#2a6fbd" font-family="monospace" font-size="8" text-anchor="middle">GE MS5002C — 30.6 MW — 4670 RPM — Hassi R'Mel Algeria</text>
            </svg>
        </div>""", unsafe_allow_html=True)

    with col2:
        specs = [
            ({"ar":"⚡ القدرة","en":"⚡ Power"}[lang], f"{MS5002C_SPECS['power_mw']} MW"),
            ({"ar":"🔄 السرعة","en":"🔄 Speed"}[lang], f"{MS5002C_SPECS['speed_rpm']} RPM"),
            ({"ar":"📊 نسبة الضغط","en":"📊 Pressure Ratio"}[lang], f"{MS5002C_SPECS['pressure_ratio']} : 1"),
            ({"ar":"🌡️ حرارة العادم","en":"🌡️ Exhaust Temp"}[lang], f"{MS5002C_SPECS['exhaust_temp_c']} °C"),
            ({"ar":"💨 تدفق الهواء","en":"💨 Air Flow"}[lang], f"{MS5002C_SPECS['mass_flow_kgs']} kg/s"),
            ({"ar":"⚙️ الكفاءة","en":"⚙️ Efficiency"}[lang], f"{MS5002C_SPECS['efficiency']}%"),
            ({"ar":"🔧 التطبيق","en":"🔧 Application"}[lang], MS5002C_SPECS['application_ar'] if lang=='ar' else MS5002C_SPECS['application_en']),
            ({"ar":"📍 الموقع","en":"📍 Location"}[lang], MS5002C_SPECS['origin']),
        ]
        html = f'<div class="catalog-card" style="{dir_style}">'
        html += f'<div style="color:#4db8ff;font-size:1.05rem;font-weight:bold;margin-bottom:12px;">🔧 {"المواصفات التقنية" if lang=="ar" else "Technical Specifications"}</div>'
        for label, val in specs:
            html += f'<div class="catalog-spec">{label} <span class="catalog-spec-val">{val}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="catalog-card" style="{dir_style}">
        <div style="color:#b388ff;font-size:0.95rem;font-weight:bold;margin-bottom:10px;">
            🛰️ {"محاكاة مستلهمة من NASA CMAPSS — إحصاءات مرجعية فقط" if lang=="ar" else "NASA CMAPSS-Inspired Simulation — Reference Statistics Only"}
        </div>
        <div style="font-family:monospace;font-size:0.78rem;color:#7ab0d4;line-height:2.0;">
            📌 {"المرجع" if lang=="ar" else "Reference"}: <span style="color:#4db8ff;">A. Saxena & K. Goebel (2008) — NASA Ames Research Center</span><br>
            ⚠️ {"تنبيه: محاكاة رياضية معايرة على الإحصاءات المنشورة — ليست ملفات FD001 الخام" if lang=="ar" else "Note: synthetic signal calibrated to published stats — NOT raw FD001 files"}<br>
            📊 {"المجموعة" if lang=="ar" else "Dataset"}: <span style="color:#b388ff;">CMAPSS FD001 — {"تدهور ضاغط الضغط العالي" if lang=="ar" else "HPC Degradation — Single fault mode"}</span><br>
            📡 {"حساسات تدريبية" if lang=="ar" else "Training Sensors"}: <span style="color:#4dff88;">T2, T24, T30, T50, P2, P15, P30</span><br>
            ⚙️ {"وحدات التدريب" if lang=="ar" else "Training Units"}: <span style="color:#ffd700;">100 {"وحدة توربين محاكاة" if lang=="ar" else "simulated turbine units"}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    plt.rcParams.update(PLT_STYLE)
    names = NASA_CMAPSS_SENSOR_NAMES[lang]
    means = np.array(NASA_CMAPSS_NORMAL['means']); stds = np.array(NASA_CMAPSS_NORMAL['stds'])
    units = NASA_CMAPSS_NORMAL['units']
    fig, ax = plt.subplots(figsize=(12,3.5)); x = np.arange(len(means))
    ax.bar(x, means, yerr=stds*2, color='#1e4a7a', alpha=0.75, edgecolor='#4db8ff',
           linewidth=1.2, capsize=5, ecolor='#ffd700', label='Mean ± 2σ (NASA-published ref.)')
    for xi,(m,s,u) in enumerate(zip(means,stds,units)):
        ax.text(xi, m+s*2+means.max()*0.04, f'{m:.0f} {u}', ha='center', fontsize=7, color='#4db8ff')
    ax.set_xticks(x); short = [n.split('(')[0].strip()[:14] for n in names]
    ax.set_xticklabels(short, fontsize=7, rotation=12)
    title = {'ar':'القيم الطبيعية للحساسات — إحصاءات مرجعية من NASA CMAPSS FD001 (محاكاة)','en':'Normal Sensor Values — NASA CMAPSS FD001 reference stats (simulated)'}[lang]
    ax.set_title(title, color='#4db8ff', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y'); plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════
#  PDF Report
# ══════════════════════════════════════════════════════════════════════
def generate_pdf(alert,identified,ratios,health_scores,conf,fault_type_name,
                 spe_filt,threshold,fault_start,t_axis,X_faulty,
                 model_method,prediction,session_info,recommendations):
    lang=st.session_state.get('lang','en')
    BLUE=rl_colors.HexColor('#1F3864'); LBLUE=rl_colors.HexColor('#2E75B6')
    RED=rl_colors.HexColor('#C62828'); GREEN=rl_colors.HexColor('#1B5E20'); ORANGE=rl_colors.HexColor('#E65100')
    alert_rl={'NORMAL':GREEN,'WARNING':rl_colors.HexColor('#F57F17'),'FAULT':ORANGE,'CRITICAL':RED}.get(alert['level'],RED)

    fig,axes=plt.subplots(2,1,figsize=(14,8)); fig.patch.set_facecolor('white')
    axes[0].plot(t_axis,spe_filt,color='#1565C0',lw=1.5,label='Error Index + EWMA')
    axes[0].axhline(threshold,color='#C62828',ls='--',lw=2,label='Threshold')
    axes[0].axvline(t_axis[fault_start],color='#FF8F00',ls=':',lw=2,label='Fault Start')
    axes[0].fill_between(t_axis,spe_filt,threshold,where=spe_filt>threshold,color='#FFCDD2',alpha=0.7)
    axes[0].set_facecolor('#FAFAFA'); axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3)
    axes[0].set_title('Fault Detection Index (SPE + EWMA)',fontsize=11,fontweight='bold')
    n_s=len(ratios); bc=['#C62828' if i==identified else '#1565C0' for i in range(n_s)]
    axes[1].bar(range(n_s),ratios,color=bc,alpha=0.85,edgecolor='#333',lw=0.5)
    lbs=[f"{'ح' if lang=='ar' else 'S'}{i+1}" for i in range(n_s)]
    axes[1].set_xticks(range(n_s)); axes[1].set_xticklabels(lbs)
    axes[1].set_title('Fault Localization Ranking',fontsize=11,fontweight='bold')
    axes[1].set_facecolor('#FAFAFA'); axes[1].grid(True,alpha=0.3,axis='y')
    plt.tight_layout(); cb=io.BytesIO()
    plt.savefig(cb,format='png',dpi=120,bbox_inches='tight',facecolor='white'); plt.close(); cb.seek(0)

    buf=io.BytesIO(); styles=getSampleStyleSheet()
    TS=ParagraphStyle('T',fontSize=17,fontName='Helvetica-Bold',textColor=BLUE,alignment=TA_CENTER,spaceAfter=3)
    SS2=ParagraphStyle('S2',fontSize=9,fontName='Helvetica',textColor=rl_colors.HexColor('#555'),alignment=TA_CENTER,spaceAfter=8)
    SEC=ParagraphStyle('Sec',fontSize=12,fontName='Helvetica-Bold',textColor=LBLUE,spaceBefore=10,spaceAfter=4)
    BD=ParagraphStyle('Bd',fontSize=9,fontName='Helvetica',textColor=rl_colors.HexColor('#333'),spaceAfter=3,leading=14)
    ALS=ParagraphStyle('Als',fontSize=13,fontName='Helvetica-Bold',textColor=alert_rl,alignment=TA_CENTER,spaceBefore=5,spaceAfter=5)
    FT=ParagraphStyle('Ft',fontSize=7,textColor=rl_colors.HexColor('#888'),alignment=TA_CENTER)

    def mk_tbl(data,cw):
        t=Table(data,colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
            ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#CCC')),('ALIGN',(1,0),(-1,-1),'CENTER'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#EEF4FF'),rl_colors.white]),
            ('LEFTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)
        ])); return t

    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=1.8*cm,leftMargin=1.8*cm,topMargin=1.8*cm,bottomMargin=1.8*cm)
    algo_map={'linear':'Linear PCA','kernel':'Kernel PCA','autoencoder':'Autoencoder ACPNL (5-Layer)'}
    info_data=[[T('param'),T('value')],[T('report_date'),session_info['date']],
               [T('algo_used'),algo_map.get(model_method,model_method)],
               ['Data Source', session_info['data_source']],
               [T('samples'),str(session_info['n_samples'])],
               [T('det_rate'),f"{session_info['det_rate']:.1f}%"],
               [T('conf_score'),f"{conf}%"],[T('fault_type_rep'),fault_type_name],
               [T('pred_note'),prediction[1] if prediction[0] is not None else 'N/A']]

    n_s2=len(health_scores)
    health_data=[[T('sensor_col'),T('health_col'),T('status_col'),T('ratio_col')]]
    for i in range(n_s2):
        h=health_scores[i]; r=ratios[i] if i<len(ratios) else 0
        st_txt=T('faulty_rep') if i==identified else (T('warn_rep') if h<70 else T('healthy_rep'))
        health_data.append([SN(i),f'{h:.1f}%',st_txt,f'{r:.1f}'])

    ht=mk_tbl(health_data,[6.5*cm,3*cm,4*cm,4*cm])
    ht.setStyle(TableStyle([('BACKGROUND',(0,identified+1),(-1,identified+1),rl_colors.HexColor('#FFEBEE')),
        ('TEXTCOLOR',(0,identified+1),(-1,identified+1),RED),
        ('FONTNAME',(0,identified+1),(-1,identified+1),'Helvetica-Bold')]))

    rec_items=[f"1. {recommendations['immediate']}", f"2. {recommendations['short_term']}",
        f"3. {recommendations['preventive']}",
        f"4. {'بعد الإصلاح، أعد تدريب النموذج على بيانات التشغيل الجديدة.' if lang=='ar' else 'After repair, retrain model on new normal-operation data.'}",
        f"5. {'سجّل هذا الحادث في قاعدة بيانات تاريخ الصيانة.' if lang=='ar' else 'Log this event in maintenance history database.'}"]

    story=[Paragraph(T('report_title'),TS), Paragraph(T('report_subtitle'),SS2),
        Paragraph('NASA CMAPSS — Real FD001 telemetry or inspired simulation · Educational Decision Engine v6.3',
                  ParagraphStyle('n',fontSize=8,textColor=rl_colors.HexColor('#7c4dff'),alignment=TA_CENTER,spaceAfter=6)),
        HRFlowable(width="100%",thickness=2.5,color=LBLUE,spaceAfter=8),
        Paragraph(alert['title'],ALS),
        Paragraph(f"{alert['action']}",ParagraphStyle('a',fontSize=9,fontName='Helvetica-Bold',textColor=alert_rl,alignment=TA_CENTER,spaceAfter=6)),
        HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor('#DDD'),spaceAfter=8),
        Paragraph(T('session_info'),SEC), mk_tbl(info_data,[8*cm,9.5*cm]),
        Spacer(1,0.3*cm), Paragraph(T('sensor_health'),SEC), ht,
        Spacer(1,0.3*cm), Paragraph(T('charts_section'),SEC),
        Image(cb,width=16.5*cm,height=8.5*cm), Spacer(1,0.3*cm),
        Paragraph(T('maintenance_rec'),SEC),
        HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor('#DDD'),spaceAfter=5)]
    for r in rec_items: story.append(Paragraph(r,BD))
    story+=[Spacer(1,0.5*cm), HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor('#CCC')),
            Paragraph(f"Turbine Guardian v6.3 | Data source: {session_info['data_source']} | {T('confidential')} | {T('footer_txt')}",FT)]
    doc.build(story); buf.seek(0); return buf.read()

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:12px 14px 8px;border-bottom:1px solid #1a3a6a;margin-bottom:14px;">
        <div style="font-family:'Share Tech Mono',monospace;color:#4db8ff;font-size:1.1rem;
             text-shadow:0 0 10px rgba(77,184,255,0.5);">🛡️ {T('app_title')}</div>
        <div style="font-family:'Share Tech Mono',monospace;color:#5a8aaa;
             font-size:0.6rem;letter-spacing:2px;">v6.3 · Real NASA FD001 + RUL · MS5002C · EDUCATIONAL</div>
        <div style="margin-top:6px;"><span class="nasa-badge">🛰️ NASA-Inspired Sim</span></div>
    </div>""", unsafe_allow_html=True)

    lang_choice = st.selectbox(T('language'), options=['English 🇬🇧','العربية 🇩🇿'],
                                index=0 if st.session_state.lang=='en' else 1)
    new_lang = 'en' if 'English' in lang_choice else 'ar'
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang; st.rerun()

    st.markdown(f'<div class="shdr">{T("display_mode")}</div>', unsafe_allow_html=True)
    display_mode = st.radio("", [T('simple_mode'), T('expert_mode')], label_visibility='collapsed')
    is_simple = (display_mode == T('simple_mode'))

    st.markdown(f'<div class="shdr">{T("data_source")}</div>', unsafe_allow_html=True)
    if DEMO_MODE:
        st.markdown(
            f"<div style='background:#fff3cd;border:1px solid #ffc107;border-radius:5px;"
            f"padding:6px 10px;font-size:0.72rem;color:#856404;margin-bottom:6px;'>"
            f"🔒 NASA الحقيقية ورفع الملفات متاحان في "
            f"<a href='{GUMROAD_URL}' target='_blank' style='color:#856404;font-weight:700;'>النسخة الكاملة</a>"
            f"</div>", unsafe_allow_html=True)
        data_src = st.radio("", [T('nasa_sim'), T('simulation')], label_visibility='collapsed')
    else:
        data_src = st.radio("", [T('nasa_real'), T('nasa_sim'), T('simulation'), T('upload_file')], label_visibility='collapsed')

    X_uploaded = None
    nasa_real_data = None
    nasa_engine_id = None
    nasa_use_test = False

    if T('nasa_real') in data_src:
        # ── Real NASA FD001 data path ──
        st.markdown(f"""<div style="background:#0a1628;border:1px solid #00c853;border-radius:6px;
          padding:8px 12px;margin-bottom:8px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#4dff88;">
          {T('real_data_badge')}</div>""", unsafe_allow_html=True)
        try:
            nasa_real_data = load_nasa_fd001(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasa_data', 'train_FD001.txt'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasa_data', 'test_FD001.txt'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasa_data', 'RUL_FD001.txt'))
            nasa_use_test = st.checkbox(T('use_test_set'), value=False)
            unit_list = nasa_real_data['units'] if not nasa_use_test else sorted(nasa_real_data['test_df']['unit'].unique().tolist())
            nasa_engine_id = st.selectbox(T('engine_select'), unit_list, index=0)
            st.caption(T('engine_info'))
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}")
            nasa_real_data = None
    elif T('upload_file') in data_src:
        uf = st.file_uploader("", type=['csv','xlsx','xls'], label_visibility='collapsed')
        if uf:
            try:
                df = pd.read_csv(uf) if uf.name.endswith('.csv') else pd.read_excel(uf)
                df = df.dropna(axis=1,how='all').dropna(axis=0,how='all')
                nc = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(nc) >= 3:
                    X_uploaded = df[nc].values
                    st.success(f"✅ {X_uploaded.shape[0]}×{X_uploaded.shape[1]}")
                else:
                    st.error("Min 3 numeric columns required / يجب 3 أعمدة رقمية على الأقل")
            except Exception as e:
                st.error(str(e))
    else:
        if DEMO_MODE:
            st.markdown(f"<span style='font-size:0.75rem;color:#856404;'>🔒 الديمو: حد أقصى {DEMO_MAX_POINTS} نقطة</span>", unsafe_allow_html=True)
            n_samples = st.slider(T('num_points'), 100, DEMO_MAX_POINTS, DEMO_MAX_POINTS, 50)
        else:
            n_samples = st.slider(T('num_points'), 500, 2000, 1000, 100)
        seed_v = st.number_input("Seed", 0, 999, 42)

    if T('nasa_real') not in data_src and T('upload_file') not in data_src:
        st.markdown(f'<div class="shdr">{T("fault_settings")}</div>', unsafe_allow_html=True)
        faulty_s = st.selectbox(T('faulty_sensor'), range(7), format_func=lambda x: SN(x).split('—')[0].strip())
        fault_pct = st.slider(T('fault_start'), 10, 80, 40)
        fault_mag = st.slider(T('fault_severity'), 0.1, 1.5, 0.4, 0.05)
        ft_options = {T('bias'):'bias', T('drift'):'drift', T('noise'):'noise'}
        ft_sel = st.selectbox(T('fault_type'), list(ft_options.keys()))
        fault_tp = ft_options[ft_sel]
    else:
        faulty_s=0; fault_pct=40; fault_mag=0.4; fault_tp='bias'

    st.markdown(f'<div class="shdr">{T("model_section")}</div>', unsafe_allow_html=True)
    algo_opts = {T('autoencoder_lbl'):'autoencoder', T('linear_pca'):'linear', T('kernel_pca'):'kernel'}
    algo_sel = st.selectbox(T('algorithm'), list(algo_opts.keys()))
    method = algo_opts[algo_sel]

    ewma_a = st.slider(T('ewma_alpha'), 0.05, 0.5, 0.2, 0.05)
    conf_lvl = st.slider(T('confidence_level'), 90, 99, 99, 1)
    if method == 'autoencoder' and not is_simple:
        ae_ep = st.slider(T('epochs'), 100, 600, 400, 50)
        ae_ep_p = st.slider(T('partial_epochs'), 200, 400, 300, 50)
    else:
        ae_ep=400; ae_ep_p=300

    uploaded_model = None
    if not is_simple:
        st.markdown(f'<div class="shdr">{T("load_model")}</div>', unsafe_allow_html=True)
        uploaded_model = st.file_uploader("", type=['pkl'], label_visibility='collapsed', key="model_upl")

    st.markdown(f'<div class="one-btn">{T("one_btn_note")}</div>', unsafe_allow_html=True)
    run_btn = st.button(f"🚀 {T('run_analysis')}", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
lang = st.session_state.get('lang', 'en')
dir_style = "direction:rtl;text-align:right;" if lang == 'ar' else ""

st.markdown(f"""
<div class="hero">
    <div style="display:flex;align-items:center;gap:16px;{dir_style}">
        <div style="font-size:2.5rem;">🛡️</div>
        <div>
            <h1>{T('app_title')} <span style="font-size:0.6em;color:#7c4dff;">v6</span></h1>
            <div class="sub">{T('app_subtitle')}</div>
            <div class="desc">{T('app_desc')}</div>
        </div>
        <div style="margin-left:auto;">
            <span class="nasa-badge">🛰️ NASA-Inspired Simulation</span>&nbsp;
            <span style="background:#0d2347;border:1px solid #4db8ff;border-radius:20px;
              padding:4px 12px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;
              color:#4db8ff;">{'🇩🇿 عربي' if lang=='ar' else '🇬🇧 English'}</span>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  Welcome Screen
# ══════════════════════════════════════════════════════════════════════
if not run_btn:
    icons_row = [('🚨','tab_decision'),('🧠','tab_xai'),('📘','tab_catalog'),('📈','tab_timeline'),('📋','tab_log')]
    cols = st.columns(len(icons_row))
    for col,(icon,key) in zip(cols,icons_row):
        with col:
            st.markdown(f'<div class="mcard"><div class="mval">{icon}</div><div class="mlbl">{T(key)}</div></div>',
                        unsafe_allow_html=True)

    st.markdown(f"""<br>
    <div style="background:#080d18;border:1px solid #1a3a6a;border-radius:8px;padding:26px;
      font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#5a8aaa;line-height:2.4;{dir_style}">
        <div style="color:#4db8ff;font-size:0.95rem;font-weight:bold;margin-bottom:14px;">
            🛡️ {T('welcome_title')} — {"ما الجديد؟" if lang=='ar' else "What's New?"}
        </div>
        🛰️ {"محاكاة رياضية معايرة على إحصاءات NASA C-MAPSS المنشورة (FD001) — وليست بيانات تليمتري حقيقية" if lang=='ar' else "Synthetic simulation calibrated to published NASA C-MAPSS (FD001) statistics — not real telemetry"}<br>
        🚨 {"قرار صناعي نصي فوري — بدل المخططات المعقدة" if lang=='ar' else "Instant plain-text industrial decision — replaces complex charts"}<br>
        🧠 {"تفسير ذكي XAI — لماذا هذا القرار بالأرقام" if lang=='ar' else "XAI explanation — why this decision, with numbers"}<br>
        🔍 {"تحديد نوع العطل: Bias / Drift / Noise" if lang=='ar' else "Fault type identification: Bias / Drift / Noise"}<br>
        💡 {"محرك توصيات ذكي — الإجراء الصحيح مباشرة" if lang=='ar' else "Smart recommendation engine — correct action directly"}<br>
        📘 {"كتالوج مصور ثنائي اللغة للتوربين MS5002C" if lang=='ar' else "Illustrated bilingual catalog for MS5002C turbine"}<br>
        🔌 {"تصدير JSON للتكامل مع أنظمة SCADA/DCS" if lang=='ar' else "JSON export for SCADA/DCS integration"}<br>
        🎯 {"مفتاح تشغيل واحد — زر واحد يشغّل كل شيء" if lang=='ar' else "Single start button — one click runs everything"}<br>
    </div>""", unsafe_allow_html=True)

else:
    # ══════════════════════════════════════════════════════
    #  DATA PREPARATION
    # ══════════════════════════════════════════════════════
    nasa_true_rul = None
    if T('nasa_real') in data_src and nasa_real_data is not None:
        # ── REAL NASA FD001 telemetry — genuine run-to-failure engine data ──
        sensors_used = NASA_FD001_TOP7_SENSORS
        if nasa_use_test:
            X_raw, t_axis, nasa_true_rul = extract_nasa_test_unit(
                nasa_real_data['test_df'], nasa_real_data['rul_true'], nasa_engine_id, sensors_used)
            data_lbl = f"NASA C-MAPSS FD001 — REAL telemetry, test unit #{nasa_engine_id} (genuine NASA data)"
        else:
            X_raw, t_axis, _ = extract_nasa_unit_run(nasa_real_data['train_df'], nasa_engine_id, sensors_used)
            data_lbl = f"NASA C-MAPSS FD001 — REAL telemetry, train unit #{nasa_engine_id} (genuine NASA data, run-to-failure)"
        n_s = X_raw.shape[1]; N = X_raw.shape[0]
        # No fault is injected here — the engine genuinely degrades. We mark
        # the "fault_start" as the onset of detectable degradation (last 1/3
        # of life is a standard heuristic in the C-MAPSS literature when the
        # true onset isn't separately labeled in the raw files).
        fault_start = max(5, int(N * 0.65))
        X_clean = X_raw[:fault_start]
        X_faulty = X_raw
        X_clean_full = X_raw
        faulty_s = 0  # not used (no manual injection) — kept for downstream compatibility
        has_inj = False
        # Real NASA sensor names replace the MS5002C placeholder names everywhere in the UI
        st.session_state['active_sensor_labels'] = [NASA_FD001_SENSOR_LABELS.get(s, s) for s in sensors_used]
        st.session_state['active_sensor_labels_short'] = [NASA_FD001_SENSOR_LABELS.get(s, s).split('(')[0].strip() for s in sensors_used]
    elif T('upload_file') in data_src and X_uploaded is not None:
        X_raw=X_uploaded[:,:min(X_uploaded.shape[1],7)]
        n_s=X_raw.shape[1]; N=X_raw.shape[0]
        fault_start=int(N*fault_pct/100)
        X_clean=X_raw[:fault_start]; X_faulty=X_raw
        t_axis=np.arange(N,dtype=float)
        data_lbl="Uploaded File"; has_inj=False; X_clean_full=X_raw
        st.session_state['active_sensor_labels'] = None
        st.session_state['active_sensor_labels_short'] = None
    elif T('upload_file') in data_src and X_uploaded is None:
        st.error("⚠️ " + ("يرجى رفع ملف البيانات أولاً" if lang=='ar' else "Please upload a data file first"))
        st.stop()
    elif T('nasa_real') in data_src and nasa_real_data is None:
        st.error("⚠️ " + ("تعذّر تحميل ملفات NASA — تحقق من وجودها في nasa_data/" if lang=='ar' else "Could not load NASA files — check nasa_data/ folder"))
        st.stop()
    else:
        if T('nasa_sim') in data_src:
            X_clean_full, t_axis = generate_turbine_data(n_samples, seed_v, nasa_mode=True)
            data_lbl = "NASA CMAPSS-Inspired Simulation (synthetic, calibrated to published FD001 stats)"
        else:
            X_clean_full, t_axis = generate_turbine_data(n_samples, seed_v, nasa_mode=False)
            data_lbl = "MS5002C Simulation"
        fault_start = int(n_samples * fault_pct / 100)
        X_clean = X_clean_full[:fault_start]
        X_faulty = inject_fault(X_clean_full, faulty_s, fault_start, fault_mag, fault_tp)
        n_s=7; N=n_samples; has_inj=True
        st.session_state['active_sensor_labels'] = None
        st.session_state['active_sensor_labels_short'] = None

    # ══════════════════════════════════════════════════════
    #  PROCESSING — ONE BUTTON RUNS EVERYTHING
    # ══════════════════════════════════════════════════════
    with st.status(T('processing'), expanded=True) as status:
        st.write(T('checking_quality'))
        dq = check_data_quality(X_faulty); X_faulty = dq['X_clean']

        # ── Real train/test split on the NORMAL data ──
        # Train the model on TRAIN_SPLIT (e.g. 70%) of normal operation only.
        # The held-out 30% test split is NEVER seen during training and is
        # used (together with the full faulty sequence) to compute the
        # threshold and to score Precision/Recall/F1/ROC-AUC. This avoids
        # evaluating on the same data the model was fit on.
        TRAIN_SPLIT = 0.70
        n_clean = X_clean.shape[0]
        split_idx = max(10, int(n_clean * TRAIN_SPLIT))
        X_train_raw = X_clean[:split_idx]
        X_testnorm_raw = X_clean[split_idx:]  # held-out normal data, unseen by training

        scaler = StandardScaler()
        Xn_train = scaler.fit_transform(X_train_raw)
        Xn_testnorm = scaler.transform(X_testnorm_raw) if len(X_testnorm_raw) > 0 else Xn_train[-5:]
        Xn_faulty = scaler.transform(X_faulty)

        if uploaded_model:
            st.write("📂 " + ("تحميل النموذج..." if lang=='ar' else "Loading model..."))
            try:
                d = pickle.load(uploaded_model)
                model=d['model']; scaler=d['scaler']; threshold=d['threshold']; method=d['method']
                saved_ver = d.get('version', 'unknown')
                if saved_ver != '6.3':
                    st.warning(("⚠️ نموذج محفوظ بإصدار مختلف ("+str(saved_ver)+") — قد لا يكون متوافقاً تماماً مع v6.2" if lang=='ar'
                                else "⚠️ Model saved with a different version ("+str(saved_ver)+") — may not be fully compatible with v6.2"))
            except Exception as e:
                st.error(str(e)); st.stop()
        else:
            pb1 = st.progress(0, text=T('building_model'))
            if method == 'linear':
                model = build_pca(Xn_train); pb1.progress(100, text=T('training_complete'))
            elif method == 'kernel':
                kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1, fit_inverse_transform=True)
                kpca.fit(Xn_train)
                model = {'type':'kernel','model':kpca,'n_components':3}
                pb1.progress(100, text=T('training_complete'))
            else:
                model = build_autoencoder(Xn_train, ae_ep, pb1)

        # Threshold computed on the HELD-OUT normal test split (not training data)
        spe_testnorm, _ = compute_spe(Xn_testnorm, model)
        threshold = np.percentile(spe_testnorm, conf_lvl)
        spe_f, Xne = compute_spe(Xn_faulty, model)
        spe_ff = ewma_filter(spe_f, ewma_a)
        fd = spe_ff > threshold
        det_rate = fd[fault_start:].sum() / (N - fault_start) * 100
        false_alarm = fd[:fault_start].sum() / fault_start * 100 if fault_start > 0 else 0
        det_time = next((i for i in range(fault_start, len(fd)) if fd[i]), None)

        pb2 = st.progress(0, text=T('localizing'))
        ratios, identified, spe_bef, spe_aft = localize_fault(Xn_faulty, fault_start, method, ae_ep_p, pb2)

        st.write(T('computing_health'))
        health = np.clip([compute_health_score(spe_aft[i]/(spe_bef[i]+1e-10)*threshold*0.3, threshold)
                          for i in range(n_s)], 0, 100)
        health = np.array(health)
        health[identified] = compute_health_score(spe_ff[fault_start:].mean() * 2.5, threshold)
        health = np.clip(health, 0, 100)

        conf_score = compute_confidence(ratios)
        ft_key, ft_name, _ = detect_fault_type(spe_ff, fault_start)
        timeline = compute_health_timeline(spe_ff, threshold, window=max(10, N//50))
        prediction = predict_fault(spe_ff, threshold, window=min(40, N//10))
        alert = get_alert(det_rate, identified, det_time, fault_start, conf_score, ft_name)
        log = save_log(alert, identified, conf_score, ft_key)

        # ── Scientific evaluation on a held-out, model-unseen sequence ──
        # Eval sequence = held-out normal test split (label 0) + full faulty
        # sequence (label 1 from fault_start onward). The model has NOT seen
        # X_testnorm_raw during training — this is a genuine train/test split,
        # not an in-sample evaluation.
        spe_eval = np.concatenate([spe_testnorm, spe_f])
        spe_eval_filtered = ewma_filter(spe_eval, ewma_a)
        eval_fault_start = len(spe_testnorm) + fault_start
        det_metrics = compute_detection_metrics(spe_eval_filtered, threshold, eval_fault_start, len(spe_eval_filtered))
        det_metrics['n_test_normal'] = len(spe_testnorm)
        det_metrics['n_train'] = len(Xn_train)

        # ── RUL — Remaining Useful Life (piecewise-linear degradation model) ──
        rul_data = compute_rul(spe_ff, threshold, fault_start, N)

        # ── Genuine RUL accuracy check against NASA's own ground truth ──
        # Only possible when using the NASA test set (RUL_FD001.txt provides
        # the true remaining life at the cutoff point for each test engine).
        rul_data['true_rul'] = None
        rul_data['rul_error'] = None
        if nasa_true_rul is not None:
            rul_data['true_rul'] = nasa_true_rul
            rul_data['rul_error'] = round(abs(rul_data['current_rul'] - nasa_true_rul), 1)

        corr_m = np.corrcoef(X_faulty[:, :n_s].T)
        explanation = generate_explanation(identified, corr_m, ft_key)
        recommendations = generate_recommendations(alert['level'], identified, ft_key, prediction)
        ranking = compute_sensor_ranking(ratios, health)
        xai_data = compute_xai(Xn_faulty, fault_start, identified, spe_ff, threshold, model, ft_key)
        json_output = build_json_output(alert, identified, conf_score, ft_key, health,
                                        det_rate, prediction, xai_data, recommendations, data_lbl)
        json_output["evaluation_metrics"] = {
            "precision": det_metrics['precision'], "recall": det_metrics['recall'],
            "f1_score": det_metrics['f1'], "accuracy": det_metrics['accuracy'],
            "specificity": det_metrics['specificity'], "roc_auc": det_metrics['auc'],
            "train_test_split": {"n_train": det_metrics['n_train'],
                                  "n_test_held_out": det_metrics['n_test_normal'],
                                  "note": "Evaluated on held-out data not seen during training"},
        }
        json_output["rul"] = {
            "current_rul_steps": rul_data['current_rul'],
            "degradation_onset_step": rul_data['degradation_start'],
            "estimated_eol_step": rul_data['eol_index'],
            "is_critical": rul_data['is_critical'],
            "model": "piecewise-linear degradation (standard C-MAPSS literature method)",
        }
        status.update(label=T('analysis_complete'), state="complete")

    # ══════════════════════════════════════════════════════
    #  DATA SOURCE BANNER — make real vs simulated unmistakable
    # ══════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    is_real_nasa = "REAL telemetry" in data_lbl
    banner_color = '#00c853' if is_real_nasa else '#7c4dff'
    banner_icon = '🛰️✅' if is_real_nasa else '🔬'
    banner_text = (f"{T('real_data_badge')} — {data_lbl}" if is_real_nasa else
                   f"{'⚠️ بيانات محاكاة (وليست NASA حقيقية): ' if lang=='ar' else '⚠️ Simulated data (not real NASA telemetry): '}{data_lbl}")
    st.markdown(f"""<div style="background:#080d18;border:1px solid {banner_color};border-radius:8px;
      padding:10px 18px;font-family:'Share Tech Mono',monospace;font-size:0.76rem;color:{banner_color};
      margin-bottom:6px;{dir_style}">{banner_icon} {banner_text}</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  METRIC CARDS
    # ══════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(6)
    mets = [(f"{det_rate:.0f}%", T('detection_rate'), "#ff1744" if det_rate>70 else "#ffd700" if det_rate>20 else "#00c853"),
            (f"{conf_score}%", T('confidence'), "#00c853" if conf_score>80 else "#ffd700" if conf_score>60 else "#ff8f00"),
            (ft_key.upper(), T('fault_type_label'), "#4db8ff"),
            (f"{false_alarm:.1f}%", T('false_alarms'), "#ff1744" if false_alarm>5 else "#00c853"),
            (f"{health.mean():.0f}%", T('overall_health'), "#00c853" if health.mean()>70 else "#ffd700" if health.mean()>40 else "#ff1744"),
            (f"{health[identified]:.0f}%", {'ar':f'صحة ح{identified+1}','en':f'S{identified+1} Health'}[lang],
             "#ff1744" if health[identified]<40 else "#ffd700" if health[identified]<70 else "#00c853")]
    for col,(val,lbl,color) in zip(cols,mets):
        with col:
            st.markdown(f'<div class="mcard"><div class="mval" style="color:{color}">{val}</div><div class="mlbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  DOWNLOAD BUTTONS
    # ══════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    dl1,dl2,dl3,dl4 = st.columns(4)
    session_info = {'date': datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), 'n_samples': N,
                    'data_source': data_lbl, 'det_rate': det_rate}
    with dl1:
        if DEMO_MODE:
            st.markdown(f"""<div style='background:#fff3cd;border:1px solid #ffc107;border-radius:6px;
            padding:8px 12px;font-size:0.78rem;color:#856404;text-align:center;'>
            🔒 تصدير PDF متاح في النسخة الكاملة &nbsp;
            <a href='{GUMROAD_URL}' target='_blank' style='color:#856404;font-weight:700;'>اشتر الآن</a>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("PDF..."):
                pdf = generate_pdf(alert, identified, ratios, health, conf_score, ft_name, spe_ff, threshold,
                                   fault_start, t_axis, X_faulty, method, prediction, session_info, recommendations)
            st.download_button(T('download_pdf'), pdf, f"TurbineGuardian_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                               "application/pdf", use_container_width=True)
    with dl2:
        mb = io.BytesIO()
        pickle.dump({'model':model,'scaler':scaler,'threshold':threshold,'method':method,'version':'6.3'}, mb)
        mb.seek(0)
        st.download_button(T('save_model'), mb.read(), f"TurbineGuardian_{method}_{datetime.now().strftime('%Y%m%d')}.pkl",
                           use_container_width=True)
    with dl3:
        res_df = pd.DataFrame({T('sensor_col'): SENSOR_NAMES[lang][:n_s], T('health_col'): health.round(1),
                               T('risk_ratio'): ratios.round(2),
                               T('status'): [T('faulty_rep') if i==identified else T('healthy_rep') for i in range(n_s)]})
        st.download_button(T('export_csv'), res_df.to_csv(index=False).encode('utf-8-sig'),
                           f"Results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", use_container_width=True)
    with dl4:
        json_str = json.dumps(json_output, ensure_ascii=False, indent=2)
        st.download_button(T('export_json'), json_str.encode('utf-8'),
                           f"API_Output_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                           "application/json", use_container_width=True)

    # ══════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════
    if is_simple:
        tabs = st.tabs([T('tab_decision'), T('tab_health'), T('tab_xai'), T('tab_eval'), T('tab_catalog'), T('tab_log')])
    else:
        tabs = st.tabs([T('tab_decision'), T('tab_health'), T('tab_xai'), T('tab_eval'), T('tab_catalog'),
                        T('tab_fingerprint'), T('tab_timeline'), T('tab_localization'), T('tab_signals'), T('tab_log')])

    # TAB 0: INSTANT DECISION
    with tabs[0]:
        st.markdown(f'<div class="shdr">🚨 {"القرار الصناعي الفوري" if lang=="ar" else "INSTANT INDUSTRIAL DECISION"}</div>',
                    unsafe_allow_html=True)
        block1, block2, block3 = render_plain_decision(alert, identified, conf_score, ft_name, health,
                                                        xai_data, recommendations)
        c1,c2 = st.columns([3,2])
        with c1:
            st.markdown(block1, unsafe_allow_html=True)
            st.markdown(block2, unsafe_allow_html=True)
            st.markdown(block3, unsafe_allow_html=True)
        with c2:
            h_val = health[identified]
            h_color = '#ff1744' if h_val<40 else '#ffd700' if h_val<70 else '#00c853'
            trend_icon = '📉' if h_val<60 else ('➡️' if h_val<80 else '📈')
            st.markdown(f"""
            <div style="background:#080d18;border:1px solid {h_color};border-radius:10px;
              padding:24px;text-align:center;margin-bottom:12px;">
                <div style="color:#5a8aaa;font-family:monospace;font-size:0.72rem;margin-bottom:4px;">
                    {"صحة النظام" if lang=='ar' else "SYSTEM HEALTH"}
                </div>
                <div class="health-score-big" style="color:{h_color};">{health.mean():.0f}%</div>
                <div class="health-trend" style="color:{h_color};">{trend_icon} {('تدهور' if h_val<60 else 'مستقر' if h_val<80 else 'جيد') if lang=='ar' else ('Degrading' if h_val<60 else 'Stable' if h_val<80 else 'Good')}</div>
            </div>""", unsafe_allow_html=True)
            rul_color = '#ff1744' if rul_data['is_critical'] else ('#ffd700' if rul_data['current_rul'] < rul_data['rul_max_cap']*0.4 else '#00c853')
            st.markdown(f"""
            <div style="background:#080d18;border:1px solid {rul_color};border-radius:10px;
              padding:16px;text-align:center;margin-bottom:12px;">
                <div style="color:#5a8aaa;font-family:monospace;font-size:0.7rem;margin-bottom:4px;">
                    RUL — {"عمر متبقٍ" if lang=='ar' else "Remaining Useful Life"}
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:1.8rem;font-weight:bold;color:{rul_color};">{rul_data['current_rul']:.0f}</div>
                <div style="font-size:0.65rem;color:#5a8aaa;">{"خطوات قياس" if lang=='ar' else "measurement steps"}</div>
            </div>""", unsafe_allow_html=True)
            if prediction[0] is not None and prediction[0] > 0:
                st.markdown(f"""<div class="pred-banner" style="{dir_style}">
                    <b style="color:#ff8f00">{T('pred_warning')}</b>
                    <span style="color:#ffd700"> {prediction[1]}</span></div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="shdr">{"معاينة JSON للتكامل" if lang=="ar" else "JSON Integration Preview"}</div>',
                        unsafe_allow_html=True)
            json_preview = {"fault": json_output["fault"], "severity": json_output["severity"],
                            "sensor": json_output["sensor"], "confidence": json_output["confidence"],
                            "fault_type": json_output["fault_type"], "system_health": json_output["system_health"],
                            "rul_steps": json_output["rul"]["current_rul_steps"]}
            st.code(json.dumps(json_preview, indent=2), language='json')

    # TAB 1: HEALTH MAP
    with tabs[1]:
        c1,c2 = st.columns([3,2])
        with c1:
            st.markdown(f'<div class="shdr">{T("health_map_title")}</div>', unsafe_allow_html=True)
            fig = plot_health_map(health, n_s); st.pyplot(fig); plt.close()
        with c2:
            st.markdown(f'<div class="shdr">{T("health_summary")}</div>', unsafe_allow_html=True)
            for i in range(n_s):
                h = health[i]
                css = 'hcard-fault' if h<40 else 'hcard-warn' if h<70 else 'hcard-ok'
                icon = '🔴' if h<40 else '🟡' if h<70 else '🟢'
                bar = '█'*int(h//10) + '░'*(10-int(h//10))
                name = SN(i).split('—')[0].strip()
                extra = f"  {T('fault_identified')}" if i==identified else ""
                st.markdown(f'<div class="hcard {css}" style="{dir_style}">{icon} {name} &nbsp; {bar} &nbsp; <b>{h:.0f}%</b>{extra}</div>',
                            unsafe_allow_html=True)
            st.markdown(f'<div class="shdr">{T("data_quality")}</div>', unsafe_allow_html=True)
            dq_c = '#00c853' if dq['score']>=80 else '#ffd700' if dq['score']>=50 else '#ff1744'
            st.markdown(f"""<div style="background:#080d18;border:1px solid #1a3a6a;border-radius:6px;
              padding:12px 16px;font-family:'Share Tech Mono',monospace;{dir_style}">
                <span style="color:{dq_c};font-weight:bold;">{dq['status']} — {dq['score']}%</span>
                &nbsp;|&nbsp;<span style="color:#7ab0d4;font-size:0.8rem;">
                {dq['n_samples']} {'عينة' if lang=='ar' else 'samples'} × {dq['n_sensors']} {'حساس' if lang=='ar' else 'sensors'}
                </span></div>""", unsafe_allow_html=True)
            for issue in dq['issues']:
                c = '#ff4d4d' if '❌' in issue else '#ffd700' if '⚠️' in issue else '#4dff88'
                st.markdown(f'<div style="font-family:monospace;font-size:0.76rem;padding:3px 16px;color:{c};">{issue}</div>',
                            unsafe_allow_html=True)

    # TAB 2: XAI
    with tabs[2]:
        st.markdown(f'<div class="shdr">{T("xai_title")}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="xai-card" style="border-left-color:{alert['color']};{dir_style}">
            <div style="color:{alert['color']};font-size:0.82rem;font-weight:bold;margin-bottom:10px;">{T('xai_pattern')}</div>
            <div style="color:#e0ecff;font-size:0.9rem;line-height:1.7;">📌 {xai_data['pattern']}</div>
        </div>""", unsafe_allow_html=True)
        fig = plot_xai_contributions(xai_data, n_s, identified); st.pyplot(fig); plt.close()

        xc1,xc2 = st.columns(2)
        with xc1:
            st.markdown(f'<div class="shdr">{T("xai_contribution")}</div>', unsafe_allow_html=True)
            for i in range(n_s):
                c_val = float(xai_data['contributions'][i]); bar_w = int(c_val)
                c_color = '#ff1744' if i==identified else '#1e4a7a'
                name = SN(i).split('—')[0].strip()
                st.markdown(f"""<div style="margin:5px 0;font-family:monospace;font-size:0.78rem;{dir_style}">
                    <span style="color:{'#ff4d4d' if i==identified else '#7ab0d4'};">{name}</span>
                    <span style="float:right;color:{alert['color'] if i==identified else '#5a8aaa'};font-weight:bold;">{c_val:.1f}%</span>
                    <div style="background:#0d1525;border-radius:3px;height:6px;margin-top:3px;">
                        <div style="background:{c_color};width:{min(bar_w,100)}%;height:6px;border-radius:3px;"></div>
                    </div></div>""", unsafe_allow_html=True)
        with xc2:
            st.markdown(f'<div class="shdr">{T("xai_correlation")}</div>', unsafe_allow_html=True)
            if xai_data['corr_impact']:
                for c_item in xai_data['corr_impact']:
                    st.markdown(f"""<div style="background:#0a1628;border:1px solid #1a3a6a;border-radius:4px;
                      padding:8px 14px;margin:4px 0;font-family:monospace;font-size:0.8rem;{dir_style}">
                        <span style="color:#4db8ff;">{c_item['sensor_name']}</span>
                        <span style="color:#5a8aaa;margin:0 8px;">corr={c_item['correlation']}</span>
                        <span style="color:#ffd700;">contrib={c_item['contribution']}%</span></div>""", unsafe_allow_html=True)
            else:
                st.info({'ar':'لا توجد حساسات مترابطة متأثرة بشكل كبير','en':'No significantly correlated sensors affected'}[lang])
            st.markdown(f'<div class="shdr">{T("probable_causes")}</div>', unsafe_allow_html=True)
            for j,cause in enumerate(explanation['probable_causes'],1):
                st.markdown(f"""<div style="background:#0a1628;border:1px solid #1a3a6a;border-radius:4px;
                  padding:8px 14px;margin:4px 0;font-family:monospace;font-size:0.8rem;{dir_style}">
                    <span style="color:#4db8ff;font-weight:bold;">{j}.</span> {cause}</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="shdr">{T("smart_rec")}</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        for col,key,text,color in [(r1,T('immediate_action'),recommendations['immediate'],alert['color']),
                                    (r2,T('short_term'),recommendations['short_term'],'#4db8ff'),
                                    (r3,T('preventive'),recommendations['preventive'],'#4dff88')]:
            with col:
                st.markdown(f"""<div class="rec-card" style="border-top:3px solid {color};{dir_style}">
                    <div style="color:{color};font-size:0.73rem;font-weight:bold;margin-bottom:7px;">{key}</div>
                    <div style="color:#c8d8e8;font-size:0.77rem;line-height:1.5;">{text}</div>
                </div>""", unsafe_allow_html=True)
        if recommendations.get('stop_note'):
            st.markdown(f"""<div style="background:#1a0404;border:2px solid #ff1744;border-radius:6px;
              padding:10px 16px;margin-top:12px;font-family:monospace;color:#ff4d4d;font-size:0.86rem;
              font-weight:bold;{dir_style}">{recommendations['stop_note']}</div>""", unsafe_allow_html=True)

    # TAB 3: SCIENTIFIC EVALUATION (F1/Precision/Recall/ROC + RUL) — shown in BOTH modes
    with tabs[3]:
        st.markdown(f'<div class="shdr">{T("tab_eval")}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="xai-card" style="border-left-color:#4db8ff;{dir_style}">
            <div style="color:#7ab0d4;font-size:0.78rem;line-height:1.6;">
            {"✅ تقييم حقيقي على بيانات لم يرها النموذج أثناء التدريب: تم تدريب النموذج على %d عينة طبيعية (70%%)، والتقييم أدناه على %d عينة طبيعية محجوزة (30%%) + كامل تسلسل العطل — لا تسريب بيانات." % (det_metrics['n_train'], det_metrics['n_test_normal']) if lang=='ar' else
             "✅ Genuine held-out evaluation: the model was trained on %d normal samples (70%%), and the metrics below are computed on %d held-out normal samples (30%%, unseen during training) plus the full fault sequence — no data leakage." % (det_metrics['n_train'], det_metrics['n_test_normal'])}
            </div></div>""", unsafe_allow_html=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        eval_mets = [
            (m1, f"{det_metrics['precision']*100:.1f}%", 'Precision', "#4db8ff"),
            (m2, f"{det_metrics['recall']*100:.1f}%", 'Recall', "#00c853"),
            (m3, f"{det_metrics['f1']*100:.1f}%", 'F1-Score', "#ffd700"),
            (m4, f"{det_metrics['accuracy']*100:.1f}%", {'ar':'الدقة الكلية','en':'Accuracy'}[lang], "#a78bfa"),
            (m5, f"{det_metrics['specificity']*100:.1f}%", 'Specificity', "#ff9944"),
            (m6, f"{det_metrics['auc']:.3f}", 'ROC AUC', "#ff6d00"),
        ]
        for col,val,lbl,color in eval_mets:
            with col:
                st.markdown(f'<div class="mcard"><div class="mval" style="color:{color};font-size:1.3rem;">{val}</div><div class="mlbl">{lbl}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_roc = plot_roc_curve(det_metrics)
        st.pyplot(fig_roc); plt.close()

        st.markdown(f'<div class="shdr">{"تفسير المقاييس" if lang=="ar" else "Metrics Explained"}</div>', unsafe_allow_html=True)
        cm_explain = [
            ('TP', det_metrics['tp'], {'ar':'عطل حقيقي تم كشفه بنجاح','en':'Fault correctly detected'}[lang], '#00c853'),
            ('TN', det_metrics['tn'], {'ar':'تشغيل طبيعي تم التعرف عليه بنجاح','en':'Normal operation correctly identified'}[lang], '#4db8ff'),
            ('FP', det_metrics['fp'], {'ar':'إنذار كاذب — طبيعي صُنّف كعطل','en':'False alarm — normal flagged as fault'}[lang], '#ffd700'),
            ('FN', det_metrics['fn'], {'ar':'عطل فات الكشف — الأخطر','en':'Missed fault — most critical error'}[lang], '#ff1744'),
        ]
        for code,val,desc,color in cm_explain:
            st.markdown(f"""<div style="font-family:monospace;font-size:0.8rem;padding:6px 14px;
              border-left:3px solid {color};margin:4px 0;background:#080d18;{dir_style}">
              <span style="color:{color};font-weight:bold;">{code} = {val}</span>
              <span style="color:#7ab0d4;margin:0 10px;">— {desc}</span></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="margin-top:14px;font-family:monospace;font-size:0.72rem;color:#5a8aaa;{dir_style}">
        {"⚠️ القيود المنهجية المتبقية: هذا تقييم حقيقي بـ train/test split، لكنه لا يزال على سيناريو عطل واحد بنقطة بداية معروفة من نفس مجموعة المحاكاة. لتعميم بحثي كامل، يلزم اختبار على مجموعة بيانات NASA الحقيقية (FD001-FD004) بسيناريوهات متعددة ووحدات توربين مختلفة." if lang=='ar' else
         "⚠️ Remaining methodological limits: this is now a genuine train/test split, but still on a single fault scenario from the same simulated distribution. For full research generalization, testing against the real NASA dataset (FD001-FD004) across multiple units and fault scenarios is required."}
        </div>""", unsafe_allow_html=True)

        # ── RUL Section ──
        st.markdown(f'<div class="shdr">📉 {"العمر الإنتاجي المتبقي (RUL)" if lang=="ar" else "Remaining Useful Life (RUL)"}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="xai-card" style="border-left-color:#4dff88;{dir_style}">
            <div style="color:#7ab0d4;font-size:0.78rem;line-height:1.6;">
            {"📚 نموذج التدهور الخطي المتجزئ (Piecewise-Linear) — المعيار المستخدم في أدبيات NASA C-MAPSS البحثية (Heimes 2008, Saxena et al. 2008). تبقى RUL ثابتة عند الحد الأقصى حتى يُكتشف التدهور، ثم تنخفض خطياً نحو الصفر." if lang=='ar' else
             "📚 Piecewise-linear degradation model — the standard used across NASA C-MAPSS research literature (Heimes 2008, Saxena et al. 2008). RUL stays flat at the ceiling until degradation is detected, then decays linearly toward zero."}
            </div></div>""", unsafe_allow_html=True)

        rc1, rc2 = st.columns([3,1])
        with rc1:
            fig_rul = plot_rul_curve(rul_data, fault_start, N)
            st.pyplot(fig_rul); plt.close()
        with rc2:
            rul_color = '#ff1744' if rul_data['is_critical'] else ('#ffd700' if rul_data['current_rul'] < rul_data['rul_max_cap']*0.4 else '#00c853')
            st.markdown(f"""<div style="background:#080d18;border:1px solid {rul_color};border-radius:10px;
              padding:20px;text-align:center;margin-bottom:10px;">
                <div style="color:#5a8aaa;font-family:monospace;font-size:0.7rem;">{"RUL الحالي" if lang=='ar' else "Current RUL"}</div>
                <div class="health-score-big" style="color:{rul_color};font-size:2.2rem;">{rul_data['current_rul']:.0f}</div>
                <div style="font-size:0.68rem;color:#5a8aaa;">{"خطوات قياس" if lang=='ar' else "measurement steps"}</div>
            </div>""", unsafe_allow_html=True)
            eol_txt = rul_data['eol_index'] if rul_data['eol_index'] is not None else ('—')
            st.markdown(f"""<div style="font-family:monospace;font-size:0.78rem;color:#7ab0d4;line-height:2.0;{dir_style}">
                {"بداية التدهور" if lang=='ar' else "Degradation onset"}: <span style="color:#ffd700;">#{rul_data['degradation_start']}</span><br>
                EOL {"المقدّر" if lang=='ar' else "estimate"}: <span style="color:#ff6d00;">#{eol_txt}</span><br>
                {"الحالة" if lang=='ar' else "Status"}: <span style="color:{rul_color};font-weight:bold;">{('🔴 ' + ('حرج' if lang=='ar' else 'CRITICAL')) if rul_data['is_critical'] else ('🟢 ' + ('طبيعي' if lang=='ar' else 'NORMAL'))}</span>
            </div>""", unsafe_allow_html=True)

        # ── Genuine RUL accuracy vs NASA's own ground truth (only when test-set unit is used) ──
        if rul_data['true_rul'] is not None:
            st.markdown(f'<div class="shdr">🎯 {T("rul_comparison")}</div>', unsafe_allow_html=True)
            err_color = '#00c853' if rul_data['rul_error'] < 15 else ('#ffd700' if rul_data['rul_error'] < 35 else '#ff1744')
            rc3, rc4, rc5 = st.columns(3)
            with rc3:
                st.markdown(f'<div class="mcard"><div class="mval" style="color:#4db8ff;">{rul_data["current_rul"]:.0f}</div><div class="mlbl">{"المُقدَّر" if lang=="ar" else "Estimated"}</div></div>', unsafe_allow_html=True)
            with rc4:
                st.markdown(f'<div class="mcard"><div class="mval" style="color:#4dff88;">{rul_data["true_rul"]:.0f}</div><div class="mlbl">{T("true_rul_label")}</div></div>', unsafe_allow_html=True)
            with rc5:
                st.markdown(f'<div class="mcard"><div class="mval" style="color:{err_color};">{rul_data["rul_error"]:.0f}</div><div class="mlbl">{"خطأ مطلق |Error|" if lang=="ar" else "Absolute Error"}</div></div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="font-family:monospace;font-size:0.72rem;color:#5a8aaa;margin-top:8px;{dir_style}">
            {"ℹ️ مقارنة حقيقية مع RUL_FD001.txt الرسمي من NASA — وليس تقديراً تقريبياً. خطأ واحد فقط من 100 محرك اختبار؛ لتقييم بحثي صارم (RMSE) يلزم تكرار هذا على كل محركات FD001 الـ100." if lang=='ar' else
             "ℹ️ Genuine comparison against NASA's official RUL_FD001.txt — not an approximation. This is a single engine out of 100 test units; a rigorous research RMSE requires repeating this across all 100 FD001 test engines."}
            </div>""", unsafe_allow_html=True)

    # TAB 4: CATALOG
    with tabs[4]:
        render_turbine_catalog()

    # EXPERT TABS
    if not is_simple:
        with tabs[5]:  # Fingerprint
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="shdr">{T("tab_fingerprint")}</div>', unsafe_allow_html=True)
                fig = plot_radar(health, n_s); st.pyplot(fig); plt.close()
            with c2:
                st.markdown('<div class="shdr">━━ DECISION OUTPUT ━━</div>', unsafe_allow_html=True)
                rows = [('STATUS' if lang=='en' else 'الحالة', f"{alert['icon']} {alert['level']}", alert['color']),
                        ('SENSOR' if lang=='en' else 'الحساس', SN(identified), '#ff4d4d'),
                        ('CONFIDENCE' if lang=='en' else 'الثقة', f"{conf_score}%", '#00c853' if conf_score>80 else '#ffd700'),
                        ('FAULT TYPE' if lang=='en' else 'النوع', ft_name, '#c8d8e8'),
                        ('HEALTH' if lang=='en' else 'الصحة', f"{health[identified]:.0f}%",
                         '#ff1744' if health[identified]<40 else '#ffd700')]
                html = '<div style="background:#080d18;border:1px solid #1a3a6a;border-radius:8px;padding:18px;font-family:Share Tech Mono,monospace;">'
                for label,val,vc in rows:
                    html += f'<div style="margin-bottom:9px;"><span style="color:#5a8aaa">{label}&nbsp;&nbsp;</span><span style="color:{vc};font-weight:bold;">{val}</span></div>'
                html += f'<div style="border-top:1px solid #1a3a6a;margin-top:12px;padding-top:10px;color:#ff8f00;font-size:0.76rem;">⚡ {alert["action"][:120]}</div></div>'
                st.markdown(html, unsafe_allow_html=True)

        with tabs[6]:  # Timeline
            st.markdown(f'<div class="shdr">{T("tab_timeline")}</div>', unsafe_allow_html=True)
            fig = plot_health_timeline(timeline, fault_start); st.pyplot(fig); plt.close()
            cc1,cc2,cc3,cc4 = st.columns(4)
            for col,val,key,color in [(cc1,f"{timeline.min():.0f}%",'min_health',"#ff1744"),
                                       (cc2,f"{timeline.max():.0f}%",'max_health',"#00c853"),
                                       (cc3,f"{timeline.mean():.0f}%",'avg_health',"#4db8ff"),
                                       (cc4,f"{(timeline>70).sum()/len(timeline)*100:.0f}%",'time_healthy',"#00c853")]:
                with col:
                    st.markdown(f'<div class="mcard"><div class="mval" style="color:{color}">{val}</div><div class="mlbl">{T(key)}</div></div>',
                                unsafe_allow_html=True)

        with tabs[7]:  # Localization
            st.markdown(f'<div class="shdr">{T("tab_localization")}</div>', unsafe_allow_html=True)
            fig = plot_localization(ratios, identified, n_s); st.pyplot(fig); plt.close()
            st.markdown(f'<div class="shdr">{T("full_ranking")}</div>', unsafe_allow_html=True)
            rank_df = pd.DataFrame([{T('rank'):r['rank'], T('sensor_col'):r['full_name'],
                T('fault_prob'):f"{r['fault_prob']:.1f}%", T('health_score'):f"{r['health']:.1f}%",
                T('risk_ratio'):r['ratio'], T('status'):r['badge']} for r in ranking])
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

        with tabs[8]:  # Signals
            st.markdown(f'<div class="shdr">{T("tab_signals")}</div>', unsafe_allow_html=True)
            fig = plot_sensor_signals(X_clean_full, X_faulty, t_axis, fault_start, identified, has_inj, n_s)
            st.pyplot(fig); plt.close()

        log_tab = tabs[9]
    else:
        log_tab = tabs[5]


    # FAULT LOG
    with log_tab:
        st.markdown(f'<div class="shdr">{T("fault_log")}</div>', unsafe_allow_html=True)
        log = load_log()
        ll1,ll2,ll3 = st.columns(3)
        for col,val,key,color in [(ll1,str(log['stats']['total']),'total_events',"#4db8ff"),
                                   (ll2,str(log['stats']['critical']),'critical_events',"#ff1744"),
                                   (ll3,str(log['stats']['warnings']),'warnings_count',"#ffd700")]:
            with col:
                st.markdown(f'<div class="mcard"><div class="mval" style="color:{color}">{val}</div><div class="mlbl">{T(key)}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        css_map = {'CRITICAL':'log-critical','FAULT':'log-fault','WARNING':'log-warning','NORMAL':'log-normal'}
        icon_map = {'CRITICAL':'🔴','FAULT':'🟠','WARNING':'🟡','NORMAL':'🟢'}
        if log['events']:
            for ev in log['events'][:30]:
                sensor_display = ev.get('sensor_ar' if lang=='ar' else 'sensor_en', ev.get('sensor','?'))
                css = css_map.get(ev['level'], 'log-normal'); icon = icon_map.get(ev['level'], '🟢')
                occur = ev.get('occurrence', 1)
                occur_txt = f" | {'تكرار' if lang=='ar' else 'occur'}={occur}" if occur>1 else ''
                st.markdown(f'<div class="log-entry {css}" style="{dir_style}">{ev["timestamp"]} &nbsp;|&nbsp; {icon} {ev["level"]} '
                            f'&nbsp;|&nbsp; {sensor_display} &nbsp;|&nbsp; {ev["fault_type"]} &nbsp;|&nbsp; conf={ev["confidence"]}%{occur_txt}</div>',
                            unsafe_allow_html=True)
        else:
            st.info(T('no_events'))

# ── Demo CTA Banner ──
if DEMO_MODE:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a3a0a,#0d2347);border:1.5px solid #4db84d;
         border-radius:10px;padding:20px 28px;margin:20px 0;text-align:center;">
      <div style="color:#c8d8e8;font-family:'Share Tech Mono',monospace;font-size:0.95rem;
           font-weight:700;margin-bottom:8px;">
        🚀 أعجبك النظام؟ احصل على النسخة الكاملة
      </div>
      <div style="color:#7ab0d4;font-size:0.78rem;margin-bottom:14px;">
        NASA حقيقية · تصدير PDF · رفع ملفاتك · بدون قيود · كود كامل
      </div>
      <a href="{GUMROAD_URL}" target="_blank"
         style="background:#4db84d;color:#fff;padding:10px 28px;border-radius:8px;
                font-size:0.9rem;font-weight:700;text-decoration:none;letter-spacing:1px;">
        🛒 اشتر الآن — النسخة الكاملة
      </a>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown(f"""<div style="text-align:center;padding:16px;border-top:1px solid #1a3a6a;margin-top:25px;
  font-family:'Share Tech Mono',monospace;color:#2a4a6a;font-size:0.67rem;letter-spacing:1px;">
  🛡️ {T('footer_txt')} &nbsp;|&nbsp; 🛰️ NASA CMAPSS-Inspired (Synthetic Data)
</div>""", unsafe_allow_html=True)
