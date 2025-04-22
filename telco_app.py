import streamlit as st
import pandas as pd
# import zipfile # No longer needed
# import os      # No longer needed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Using this model
import numpy as np

# Set up page
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📉 Telco Customer Churn Predictor")

# --- Inject CSS for Styling (Simplified - Button color now via config.toml) ---
st.markdown("""
<style>
    /* Ensure styles only target the main block, not sidebar */
    div[data-testid="stAppViewContainer"] > div[data-testid="stBlock"] {
        /* --- Styling Starts --- */

        /* Primary Button styles are now handled by primaryColor in config.toml */

        /* Style the Header for Prediction Result */
        div[data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] h3 {
            color: #465a70; /* Darker Muted Navy for Header */
            padding-top: 1em;
            border-top: 1px solid #e0e0e0;
            margin-top: 1em;
        }

        /* Style the Expander Header for 'About' section */
        div[data-testid="stExpander"] summary {
            background-color: #f8f9fa; /* Very light grey */
            border: 1px solid #d3d3d3; /* Light Grey border */
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            margin-top: 1em;
            transition: background-color 0.3s ease;
        }
         div[data-testid="stExpander"] summary:hover {
            background-color: #e9ecef;
        }
         div[data-testid="stExpander"] p,
         div[data-testid="stExpander"] li,
         div[data-testid="stExpander"] h3 {
             color: inherit; /* Inherit text color from theme */
         }


        /* Style the 'Go back' Link Button */
        a.stLinkButton {
           background-color: #EAEAEA; /* Light Grey */
           color: #31333F !important; /* Dark text */
           padding: 0.5em 1em;
           border-radius: 0.25rem;
           text-decoration: none;
           border: 1px solid #D3D3D3; /* Subtle border */
           display: inline-block;
           line-height: normal;
           font-weight: normal; /* Less emphasis */
           margin-top: 1em;
           transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        a.stLinkButton:hover {
            background-color: #DCDCDC;
            border-color: #BEBEBE;
            color: #31333F !important;
            text-decoration: none;
        }
         a.stLinkButton:active {
            background-color: #C0C0C0;
         }

        /* --- Styling Ends --- */
    }
</style>
""", unsafe_allow_html=True)
# End of CSS block

# --- Initialize Session State Flags ---
if 'predict_pressed' not in st.session_state:
    st.session_state.predict_pressed = False
if 'prediction_confirmed' not in st.session_state:
    st.session_state.prediction_confirmed = False
# -------------------------------------

# --- Data Loading ---
def load_data():
    """Loads, cleans, and returns the Telco churn dataset."""
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    try:
        df = pd.read_csv(csv_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found.")
        st.info("Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the script's folder.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.stop()
# --------------------

# --- Model Training (Using RandomForestClassifier) ---
@st.cache_data
def train_model(_df):
    """Trains the RandomForest model using original logic and returns components."""
    st.info("⚙️ Training model (this happens once)...")
    df_model = _df.drop(columns=['customerID'])
    categorical_cols = df_model.select_dtypes(include='object').columns.drop('Churn')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    if 'Churn' in df_model.columns:
        le_churn = LabelEncoder()
        df_model['Churn'] = le_churn.fit_transform(df_model['Churn'])
        label_encoders['Churn'] = le_churn
    else:
        st.error("Target 'Churn' column not found.")
        st.stop()

    try:
        X = df_model.drop(columns=['Churn'])
        y = df_model['Churn']
    except KeyError:
        st.error("Could not separate features/target.")
        st.stop()

    num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if not all(col in X.columns for col in num_cols_to_scale):
        st.error(f"Numeric columns for scaling not found.")
        st.stop()

    scaler = StandardScaler()
    X[num_cols_to_scale] = scaler.fit_transform(X[num_cols_to_scale])

    # --- Using RandomForestClassifier ---
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    # ------------------------------------
    model.fit(X, y)
    st.info("✅ Model training complete.")
    return model, label_encoders, scaler, X.columns.tolist()
# --------------------------

# --- Main Application Logic ---
df = load_data()

if df is not None:
    model, label_encoders, scaler, feature_names = train_model(df.copy())

    # --- Define Default Values for Reset ---
    try:
        default_values = {
            'gender_input': label_encoders['gender'].classes_[0],
            'senior_input': "Yes", 'partner_input': label_encoders['Partner'].classes_[0],
            'dependents_input': label_encoders['Dependents'].classes_[0], 'tenure_input': 12,
            'phone_input': label_encoders['PhoneService'].classes_[0], 'multiple_input': label_encoders['MultipleLines'].classes_[0],
            'internet_input': label_encoders['InternetService'].classes_[0], 'onlinesec_input': label_encoders['OnlineSecurity'].classes_[0],
            'onlinebackup_input': label_encoders['OnlineBackup'].classes_[0], 'protection_input': label_encoders['DeviceProtection'].classes_[0],
            'tech_input': label_encoders['TechSupport'].classes_[0], 'tv_input': label_encoders['StreamingTV'].classes_[0],
            'movies_input': label_encoders['StreamingMovies'].classes_[0], 'contract_input': label_encoders['Contract'].classes_[0],
            'paperless_input': label_encoders['PaperlessBilling'].classes_[0], 'payment_input': label_encoders['PaymentMethod'].classes_[0],
            'charges_input': 65.0
        }
    except Exception as e:
        st.error(f"Error defining default values: {e}")
        st.stop()
    # ----------------------------------------

    # --- Updated Reset Callback ---
    def reset_widgets():
        """Callback to reset input widgets AND control flags."""
        for k, v in default_values.items():
            if k in st.session_state:
                st.session_state[k] = v
        # Reset control flags
        st.session_state.predict_pressed = False
        st.session_state.prediction_confirmed = False
    # ---------------------------

    # --- Sidebar Inputs ---
    st.sidebar.header("📝 Customer Input")
    if st.session_state.get('prediction_confirmed', False):
         st.sidebar.caption("Inputs locked. Press 'Reset Inputs' below prediction to change.")
    else:
        st.sidebar.caption("Adjust inputs, then click 'Predict Churn' in the main area.")

    # --- Function to get user inputs (Accepts is_disabled) ---
    def get_user_input(is_disabled):
        gender = st.sidebar.selectbox("Gender", options=label_encoders['gender'].classes_, key='gender_input', help="Customer's gender", disabled=is_disabled)
        senior_selection = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"], key='senior_input', help="Is the customer a senior citizen (65+)?", disabled=is_disabled)
        senior_map = {"Yes": 1, "No": 0}; senior = senior_map[senior_selection]
        partner = st.sidebar.selectbox("Has Partner?", options=label_encoders['Partner'].classes_, key='partner_input', help="Does the customer have a partner?", disabled=is_disabled)
        dependents = st.sidebar.selectbox("Has Dependents?", options=label_encoders['Dependents'].classes_, key='dependents_input', help="Does the customer have dependents?", disabled=is_disabled)
        tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12, key='tenure_input', help="Number of months the customer has stayed", disabled=is_disabled)
        phone = st.sidebar.selectbox("Phone Service", options=label_encoders['PhoneService'].classes_, key='phone_input', help="Does the customer have phone service?", disabled=is_disabled)
        multiple = st.sidebar.selectbox("Multiple Lines", options=label_encoders['MultipleLines'].classes_, key='multiple_input', help="Does the customer have multiple phone lines?", disabled=is_disabled)
        internet = st.sidebar.selectbox("Internet Service", options=label_encoders['InternetService'].classes_, key='internet_input', help="Type of internet service", disabled=is_disabled)
        online_sec = st.sidebar.selectbox("Online Security", options=label_encoders['OnlineSecurity'].classes_, key='onlinesec_input', help="Does the customer have online security service?", disabled=is_disabled)
        online_backup = st.sidebar.selectbox("Online Backup", options=label_encoders['OnlineBackup'].classes_, key='onlinebackup_input', help="Does the customer have online backup service?", disabled=is_disabled)
        protection = st.sidebar.selectbox("Device Protection", options=label_encoders['DeviceProtection'].classes_, key='protection_input', help="Does the customer have device protection service?", disabled=is_disabled)
        tech = st.sidebar.selectbox("Tech Support", options=label_encoders['TechSupport'].classes_, key='tech_input', help="Does the customer have tech support service?", disabled=is_disabled)
        tv = st.sidebar.selectbox("Streaming TV", options=label_encoders['StreamingTV'].classes_, key='tv_input', help="Does the customer stream TV?", disabled=is_disabled)
        movies = st.sidebar.selectbox("Streaming Movies", options=label_encoders['StreamingMovies'].classes_, key='movies_input', help="Does the customer stream movies?", disabled=is_disabled)
        contract = st.sidebar.selectbox("Contract", options=label_encoders['Contract'].classes_, key='contract_input', help="Customer's contract term", disabled=is_disabled)
        paperless = st.sidebar.selectbox("Paperless Billing", options=label_encoders['PaperlessBilling'].classes_, key='paperless_input', help="Does the customer use paperless billing?", disabled=is_disabled)
        payment = st.sidebar.selectbox("Payment Method", options=label_encoders['PaymentMethod'].classes_, key='payment_input', help="Customer's payment method", disabled=is_disabled)
        charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.05, key='charges_input', help="Customer's current monthly charge", disabled=is_disabled)
        total = float(charges * tenure)
        input_dict = {'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,'OnlineSecurity': online_sec, 'OnlineBackup': online_backup, 'DeviceProtection': protection,'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': charges,'TotalCharges': total }
        try: ordered_input = {col: input_dict[col] for col in feature_names}
        except KeyError as e: st.error(f"Input key error: {e}"); st.stop()
        return pd.Series(ordered_input)
    # --------------------

    # --- Call get_user_input with the current disabled state ---
    user_input = get_user_input(is_disabled=st.session_state.get('prediction_confirmed', False))
    # --------------------------------------------------------

    # --- Prepare input for model prediction ---
    def prepare_input(user_input_series, label_encoders_map, scaler_obj, feature_order):
        input_df = pd.DataFrame([user_input_series])
        for col, le in label_encoders_map.items():
            if col in input_df.columns and col not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']:
                 try: input_df[col] = le.transform(input_df[col])
                 except ValueError:
                     st.warning(f"Unseen value '{input_df[col].iloc[0]}' in '{col}'. Using default.")
                     default_val = 'No'; input_df[col] = le.transform([default_val])[0] if default_val in le.classes_ else -1
                 except Exception as e: st.error(f"Encoding error '{col}': {e}"); input_df[col] = -1
        num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        try:
            input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)
            input_df[num_cols_to_scale] = scaler_obj.transform(input_df[num_cols_to_scale])
        except Exception as e: st.error(f"Scaling error: {e}"); return None
        try: input_df = input_df[feature_order]
        except KeyError as e: st.error(f"Feature order error: {e}"); return None
        return input_df.values
    # ----------------------------------------

    # --- Main Area Interaction ---
    # Removed st.markdown("---") here, CSS provides separation via margin/padding

    # Display Predict button IF no prediction is confirmed yet
    if not st.session_state.get('prediction_confirmed', False):
        if st.button("➡️ Predict Churn", key="predict_button_main", help="Click after setting all inputs."):
            st.session_state.predict_pressed = True
            st.session_state.prediction_confirmed = False
            st.rerun()

    # --- Confirmation Step ---
    if st.session_state.get('predict_pressed', False) and not st.session_state.get('prediction_confirmed', False):
        st.warning("**Are you sure with these category inputs?**")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            # This button will now use the primaryColor from config.toml
            if st.button("✅ Yes, Predict!", key="confirm_yes"):
                st.session_state.prediction_confirmed = True
                st.session_state.predict_pressed = False
                st.rerun()
        with col_cancel:
             # This button uses the default secondary style
            if st.button("❌ No, Change Inputs", key="confirm_no"):
                st.session_state.predict_pressed = False
                st.session_state.prediction_confirmed = False
                st.rerun()
    # ------------------------

    # --- Predict and Display Results ---
    if st.session_state.get('prediction_confirmed', False):
        if 'model' in locals():
            prepared_input = prepare_input(user_input, label_encoders, scaler, feature_names)
            if prepared_input is not None:
                proba = model.predict_proba(prepared_input)[0][1] # Using the RandomForest model

                # The CSS styles the h3 tag below now
                st.markdown("### 🔮 Prediction Result")
                st.metric(label="Churn Probability", value=f"{proba:.2%}", delta=None)

                if proba > 0.5:
                    st.error("⚠️ High Risk of Churn")
                else:
                    st.success("✅ Low Risk of Churn")

                # Reset button appears here (uses default secondary style)
                st.button("🔄 Reset Inputs", key="reset_button_after_predict", on_click=reset_widgets, help="Click to reset inputs, hide prediction, and unlock inputs.")
            else:
                 st.error("Could not generate prediction due to input preparation error.")
        else:
            st.error("Model components not loaded correctly. Cannot predict.")
    # --- End of Conditional Display ---

    # --- Footer and Extended Information ---
    # Removed st.markdown("---") before expander as CSS adds margin
    with st.expander("ℹ️ About This App & Random Forest", expanded=False):
        st.markdown(
            """
            This application predicts the likelihood of **Telco customer churn** (whether a customer will stop using the service)
            based on various input factors like their tenure, contract type, services used, and billing information.

            **Model Used:** The prediction is generated using a **Random Forest Classifier**. This is a powerful machine learning algorithm
            known for its accuracy and robustness, trained on the publicly available 'WA_Fn-UseC_-Telco-Customer-Churn.csv' dataset.
            Below is a more detailed explanation of how it works.

            **Data Requirement:** To function correctly, this application requires the
            `WA_Fn-UseC_-Telco-Customer-Churn.csv` file to be present in the same folder as the script.
            """
        )
        st.markdown("---")
        st.subheader("What is a Random Forest?")
        st.markdown(
             """
            **Technical Definition:** A Random Forest is an **ensemble learning method** used primarily for classification and regression tasks. It operates by constructing a multitude of **decision trees** during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

            Key concepts include:
            * **Decision Trees:** Simple models that use a tree-like structure of 'if-then-else' conditions to make predictions.
            * **Ensemble:** Combining multiple models (the trees) to produce a better result than any single model.
            * **Bagging (Bootstrap Aggregating):** Each tree is trained on a random subset of the original data (sampled with replacement). This introduces variation.
            * **Feature Randomness:** When splitting a node in a tree, only a random subset of the features is considered. This further diversifies the trees and reduces correlation between them.
            * **Voting:** For classification (like churn prediction), each tree 'votes' for a class, and the forest outputs the class with the most votes.

            This combination helps to significantly reduce **overfitting** (where a model learns the training data too well but fails on new data) and generally improves prediction accuracy.
            """
        )
        st.subheader("Random Forest Explained Simply (for anyone!)")
        st.markdown(
             """
            Imagine you want to decide if a customer is likely to leave (churn). Instead of asking just one person, you ask a **large group of slightly different 'experts'** for their opinion.

            1.  **Many Experts (Trees):** A Random Forest builds many simple decision-making flowcharts (like "if Contract is Month-to-month AND Tenure < 6 months, then maybe Churn?"). These are the 'Decision Trees'.
            2.  **Different Views:** Each 'expert' (tree) doesn't get to see *all* the information. They each look at a randomly chosen *part* of the customer data and a random *selection* of customer characteristics (like only looking at contract and payment method, while another looks at tenure and online services). This makes sure the experts have different perspectives.
            3.  **Voting Time:** You show the details of a *new* customer to all these experts. Each expert (tree) gives their prediction: "Churn" or "Not Churn".
            4.  **Final Decision:** You count the votes. If most experts vote "Churn", the Random Forest predicts "Churn". If most vote "Not Churn", that's the final prediction.

            By combining many slightly different opinions, the Random Forest avoids relying too much on any single viewpoint and usually makes a more reliable and accurate final decision! 🤔🌳🌳🌳🗳️
            """
        )
    # --------------

    # --- Add Redirect Link Button Below Expander ---
    # Removed st.markdown("---") before link button as CSS adds margin
    st.link_button("Go back", "https://hoangviet05.com/", help="Visit HoangViet05.com (opens in new tab)")
    # ---------------------------------------------


else: # df is None
    st.error("⛔ Failed to load data. Application cannot proceed.")
# -----------------------------