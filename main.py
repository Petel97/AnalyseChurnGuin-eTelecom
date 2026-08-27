import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Guinée Telecom - Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card.red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .prediction-box.low {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .prediction-box.high {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FONCTIONS DE CHARGEMENT
# ============================================

@st.cache_data
def load_data():
    """Charge le dataset"""
    df = pd.read_csv("data/guinee_telecom_churn_FR.csv")
    return df

@st.cache_resource
def train_and_save_model():
    """Entraîne le modèle et le sauvegarde"""
    
    # Chargement des données
    df = load_data()
    
    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    categorical_cols = ['region', 'sexe', 'type_abonnement', 'forfait_international', 
                        'moyen_paiement', 'messagerie_vocale']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    
    df_encoded['resiliation'] = label_encoder.fit_transform(df_encoded['resiliation'])
    
    # Sélection des features
    features = [
        'region', 'sexe', 'age', 'revenu_estime_gnf', 'anciennete_mois',
        'type_abonnement', 'messagerie_vocale', 'recharge_mensuelle_moy_gnf',
        'minutes_jour', 'minutes_nuit', 'minutes_internationales', 'donnees_mo',
        'nombre_sms', 'appels_service_client', 'pannes_signalees_30j',
        'retard_paiement_jours'
    ]
    
    X = df_encoded[features]
    y = df_encoded['resiliation']
    
    # Séparation Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement du meilleur modèle (Régression Logistique)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    }
    
    # Sauvegarde du modèle et du scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Sauvegarde des métadonnées
    metadata = {
        'features': features,
        'categorical_cols': categorical_cols,
        'label_encoders': {col: label_encoder.classes_.tolist() for col in categorical_cols},
        'metrics': metrics,
        'model_type': 'LogisticRegression'
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    return model, scaler, metadata, metrics

@st.cache_resource
def load_model():
    """Charge le modèle sauvegardé ou l'entraîne s'il n'existe pas"""
    if os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        metadata = joblib.load('models/metadata.pkl')
        return model, scaler, metadata
    else:
        return train_and_save_model()

# ============================================
# FONCTION DE PRÉDICTION POUR UN CLIENT
# ============================================

def predict_churn(model, scaler, metadata, input_data):
    """Effectue une prédiction pour un nouveau client"""
    
    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    for col in metadata['categorical_cols']:
        if col in input_data.columns:
            # On utilise les classes connues
            known_classes = metadata['label_encoders'][col]
            if input_data[col].iloc[0] in known_classes:
                input_data[col] = label_encoder.fit_transform(input_data[col])
            else:
                # Si une nouvelle classe apparaît, on l'ignore
                input_data[col] = 0
    
    # Sélection des features dans le bon ordre
    features = metadata['features']
    X_input = input_data[features]
    
    # Normalisation
    X_scaled = scaler.transform(X_input)
    
    # Prédiction
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    return prediction, proba

# ============================================
# APPLICATION STREAMLIT
# ============================================

# Chargement du modèle
model, scaler, metadata = load_model()

# Sidebar - Navigation
st.sidebar.image("https://via.placeholder.com/200x60?text=Guinee+Telecom", use_container_width=True)
#st.sidebar.image("https://via.placeholder.com/200x60?text=GuineetTelecom", use_container_width=True)
st.sidebar.markdown("## 📊 Navigation")

pages = {
    "🏠 Accueil": "home",
    "📈 Analyse des données": "eda",
    "🎯 Prédiction client": "predict",
    "📊 Performance du modèle": "model_performance"
}

page = st.sidebar.radio("", list(pages.keys()))

# Footer de la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Version:** 2.0.0  
**Auteur:** Abdourahamane Diallo  
**Email:** peteldiallo97@gmail.com  
**LinkedIn:** [www.linkedin.com/in/petel-diallo](https://www.linkedin.com/in/petel-diallo)

---
*© 2026 Guinée Telecom - Tous droits réservés*
""")


# ============================================
# PAGE ACCUEIL
# ============================================
if pages[page] == "home":
    st.markdown('<p class="main-header">📊 Guinée Telecom - Analyse du Churn</p>', unsafe_allow_html=True)
    
    df = load_data()
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>{len(df):,}</h3>
            <p>👥 Clients Totaux</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = (df['resiliation'] == 'Oui').mean()
        st.markdown(f"""
        <div class="metric-card red">
            <h3>{churn_rate:.1%}</h3>
            <p>⚠️ Taux de Churn</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        retention_rate = 1 - churn_rate
        st.markdown(f"""
        <div class="metric-card green">
            <h3>{retention_rate:.1%}</h3>
            <p>✅ Taux de Rétention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card orange">
            <h3>{df['revenu_estime_gnf'].mean():,.0f} GNF</h3>
            <p>💰 Revenu Moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Présentation du projet
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objectif du projet
        
        Cette application a pour but de **prédire le risque de résiliation (churn)** 
        des clients de Guinée Telecom, à l'aide d'un modèle de Machine Learning.
        
        **Modèle utilisé :** Régression Logistique  
        **Performance :** Accuracy de **75.8%** et AUC-ROC de **0.75**
        
        #### 📌 Fonctionnalités :
        - 🔍 **Analyse exploratoire** des données
        - 🎯 **Prédiction personnalisée** pour un client
        - 📊 **Tableau de bord** de performance du modèle
        """)
    
    with col2:
        st.info("""
        **📈 Statistiques clés**
        - Régions : 8
        - Variables : 16
        - Période : 2023-2024
        - Dataset : 5 000 clients
        """)
    
    # Aperçu des données
    st.markdown("---")
    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================
# PAGE ANALYSE EXPLORATOIRE
# ============================================
elif pages[page] == "eda":
    st.markdown('<p class="main-header">📈 Analyse Exploratoire des Données</p>', unsafe_allow_html=True)
    
    df = load_data()
    
    # Filtres dans la sidebar
    st.sidebar.markdown("### 🔍 Filtres")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Distribution de la variable cible
    st.subheader("📊 Répartition des résiliations")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        churn_counts = df['resiliation'].value_counts()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=['#4CAF50', '#F44336'])
        plt.title('Nombre de clients résiliés vs non résiliés')
        plt.xlabel('Résiliation')
        plt.ylabel('Nombre de clients')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#4CAF50', '#F44336']
        plt.pie(churn_counts, labels=['Non résilié', 'Résilié'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Répartition des résiliations')
        st.pyplot(fig)
    
    # 2. Analyse par région
    st.subheader("📍 Résiliation par région")
    fig, ax = plt.subplots(figsize=(12, 6))
    region_churn = pd.crosstab(df['region'], df['resiliation'], normalize='index') * 100
    region_churn.plot(kind='bar', stacked=True, ax=ax, color=['#4CAF50', '#F44336'])
    plt.title('Taux de résiliation par région (%)')
    plt.xlabel('Région')
    plt.ylabel('Pourcentage (%)')
    plt.legend(['Non résilié', 'Résilié'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # 3. Variables numériques
    st.subheader("📊 Distribution des variables clés")
    
    selected_var = st.selectbox("Sélectionner une variable", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=selected_var, hue='resiliation', multiple='stack', 
                     palette=['#4CAF50', '#F44336'], bins=40)
        plt.title(f'Distribution de {selected_var}')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='resiliation', y=selected_var, palette=['#4CAF50', '#F44336'])
        plt.title(f'Boxplot de {selected_var} par résiliation')
        st.pyplot(fig)
    
    # 4. Matrice de corrélation
    st.subheader("🔗 Matrice de corrélation")
    
    # Encodage des variables catégorielles pour la corrélation
    df_corr = df.copy()
    for col in ['region', 'sexe', 'type_abonnement', 'forfait_international', 
                'moyen_paiement', 'messagerie_vocale']:
        df_corr[col] = LabelEncoder().fit_transform(df_corr[col])
    df_corr['resiliation'] = LabelEncoder().fit_transform(df_corr['resiliation'])
    
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, ax=ax)
    plt.title('Matrice de corrélation')
    st.pyplot(fig)

# ============================================
# PAGE PRÉDICTION
# ============================================
elif pages[page] == "predict":
    st.markdown('<p class="main-header">🎯 Prédiction du risque de résiliation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📝 Saisie des informations du client
    
    Remplissez les champs ci-dessous pour obtenir une prédiction personnalisée.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("📍 Région", 
            ['Conakry', 'Kankan', 'Boké', 'Faranah', "N'Zérékoré", 'Labé', 'Kindia', 'Mamou'])
        sexe = st.selectbox("👤 Sexe", ['Homme', 'Femme'])
        age = st.slider("🎂 Âge", 18, 71, 34)
        revenu = st.number_input("💰 Revenu estimé (GNF)", 
            min_value=300000, max_value=4694000, value=734000, step=10000)
        anciennete = st.slider("📅 Ancienneté (mois)", 1, 96, 17)
        type_abo = st.selectbox("📱 Type d'abonnement", ['Prépayé', 'Postpayé'])
    
    with col2:
        messagerie = st.selectbox("🎤 Messagerie vocale", ['Oui', 'Non'])
        recharge = st.number_input("💳 Recharge mensuelle moyenne (GNF)", 
            min_value=10000, max_value=307000, value=43000, step=1000)
        minutes_jour = st.slider("☀️ Minutes jour", 0, 438, 179)
        minutes_nuit = st.slider("🌙 Minutes nuit", 0, 243, 89)
        minutes_int = st.slider("🌍 Minutes internationales", 0, 109, 3)
        donnees = st.number_input("📶 Données (Mo)", min_value=50, max_value=6341, value=2477, step=10)
        sms = st.slider("✉️ Nombre de SMS", 0, 222, 17)
        appels_sc = st.slider("📞 Appels service client", 0, 6, 1)
        pannes = st.slider("🔧 Pannes signalées (30j)", 0, 5, 0)
        retard = st.slider("⏰ Retard paiement (jours)", 0, 36, 1)
    
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        'region': [region],
        'sexe': [sexe],
        'age': [age],
        'revenu_estime_gnf': [revenu],
        'anciennete_mois': [anciennete],
        'type_abonnement': [type_abo],
        'messagerie_vocale': [messagerie],
        'recharge_mensuelle_moy_gnf': [recharge],
        'minutes_jour': [minutes_jour],
        'minutes_nuit': [minutes_nuit],
        'minutes_internationales': [minutes_int],
        'donnees_mo': [donnees],
        'nombre_sms': [sms],
        'appels_service_client': [appels_sc],
        'pannes_signalees_30j': [pannes],
        'retard_paiement_jours': [retard]
    })
    
    # Bouton de prédiction
    st.markdown("---")
    
    if st.button("🔍 Prédire le risque de résiliation", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            prediction, proba = predict_churn(model, scaler, metadata, input_data)
        
        # Affichage du résultat
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 0:
                st.markdown("""
                <div class="prediction-box low">
                    ✅ Ce client présente un <strong>faible risque</strong> de résiliation
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box high">
                    ⚠️ Ce client présente un <strong>risque élevé</strong> de résiliation
                </div>
                """, unsafe_allow_html=True)
            
            # Probabilités détaillées
            st.markdown("---")
            st.subheader("📊 Probabilités détaillées")
            
            proba_df = pd.DataFrame({
                'Statut': ['Maintien', 'Résiliation'],
                'Probabilité': [proba[0]*100, proba[1]*100]
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#4CAF50', '#F44336']
            bars = ax.bar(proba_df['Statut'], proba_df['Probabilité'], color=colors)
            ax.set_ylabel('Probabilité (%)')
            ax.set_ylim(0, 100)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Indicateurs de risque
            st.markdown("### ⚡ Facteurs de risque")
            
            risk_factors = []
            if pannes > 0:
                risk_factors.append(f"🔧 {pannes} panne(s) signalée(s)")
            if retard > 3:
                risk_factors.append(f"⏰ Retard de paiement de {retard} jours")
            if appels_sc > 2:
                risk_factors.append(f"📞 {appels_sc} appels au service client")
            if anciennete < 6:
                risk_factors.append(f"📅 Nouveau client ({anciennete} mois)")
            if recharge < 29000:
                risk_factors.append(f"💳 Recharge mensuelle faible ({recharge:,.0f} GNF)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ Aucun facteur de risque majeur détecté")

# ============================================
# PAGE PERFORMANCE DU MODÈLE
# ============================================
elif pages[page] == "model_performance":
    st.markdown('<p class="main-header">📊 Performance du modèle</p>', unsafe_allow_html=True)
    
    # Métriques du modèle
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = metadata['metrics']
    
    with col1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>{metrics['accuracy']*100:.1f}%</h3>
            <p>🎯 Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card green">
            <h3>{metrics['precision']*100:.1f}%</h3>
            <p>📌 Précision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card orange">
            <h3>{metrics['recall']*100:.1f}%</h3>
            <p>🔍 Rappel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card red">
            <h3>{metrics['f1']*100:.1f}%</h3>
            <p>📊 F1-score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Détails du modèle
    st.markdown("---")
    st.subheader("📋 Détails du modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Type de modèle :** {metadata['model_type']}  
        **Variables :** {len(metadata['features'])}  
        **Données d'entraînement :** 4 000 clients  
        **Données de test :** 1 000 clients  
        **Régularisation :** L2 (penalty)
        """)
    
    with col2:
        st.markdown(f"""
        **AUC-ROC :** {metrics['roc_auc']:.3f}  
        **Accuracy :** {metrics['accuracy']*100:.1f}%  
        **Précision (classe 1) :** {metrics['precision']*100:.1f}%  
        **Rappel (classe 1) :** {metrics['recall']*100:.1f}%
        """)
    
    # Matrice de confusion
    st.markdown("---")
    st.subheader("📊 Matrice de confusion")
    
    # Recalcul de la matrice de confusion pour l'affichage
    df = load_data()
    label_encoder = LabelEncoder()
    categorical_cols = ['region', 'sexe', 'type_abonnement', 'forfait_international', 
                        'moyen_paiement', 'messagerie_vocale']
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    df_encoded['resiliation'] = label_encoder.fit_transform(df_encoded['resiliation'])
    
    features = metadata['features']
    X = df_encoded[features]
    y = df_encoded['resiliation']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation avec le scaler sauvegardé
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non résilié', 'Résilié'],
                yticklabels=['Non résilié', 'Résilié'], ax=ax)
    plt.title('Matrice de confusion')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    st.pyplot(fig)
    
    # Rapport de classification
    st.subheader("📋 Rapport de classification")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)
    
    # Courbe ROC
    st.subheader("📈 Courbe ROC")
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.3f}', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Taux de faux positifs (FPR)')
    ax.set_ylabel('Taux de vrais positifs (TPR)')
    ax.set_title('Courbe ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("📊 Importance des variables")
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Variable': metadata['features'],
            'Coefficient': coefs
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
        ax.barh(feature_importance['Variable'], feature_importance['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient')
        ax.set_title('Coefficients de la régression logistique')
        st.pyplot(fig)
        
        st.info("""
        **Interprétation :**
        - Les coefficients **positifs** (en vert) augmentent le risque de résiliation
        - Les coefficients **négatifs** (en rouge) diminuent le risque de résiliation
        """)