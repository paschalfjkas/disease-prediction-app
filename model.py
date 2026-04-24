import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class DiseasePredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.symptoms_list = []
        self.disease_list = []
        self.is_trained = False
        
        self.symptom_severity = {}
        self.disease_description = {}
        self.disease_precaution = {}

    def load_data(self):
        dataset_path = os.path.join(DATA_DIR, "dataset.csv")
        severity_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
        description_path = os.path.join(DATA_DIR, "symptom_Description.csv")
        precaution_path = os.path.join(DATA_DIR, "symptom_precaution.csv")

        # Load description
        if os.path.exists(description_path):
            desc_df = pd.read_csv(description_path)
            self.disease_description = dict(zip(desc_df['Disease'].str.strip(), desc_df['Description'].str.strip()))
            
        # Load precaution
        if os.path.exists(precaution_path):
            prec_df = pd.read_csv(precaution_path)
            self.disease_precaution = {}
            for index, row in prec_df.iterrows():
                disease = str(row['Disease']).strip()
                precautions = [str(p).strip() for p in row[1:] if pd.notna(p)]
                self.disease_precaution[disease] = precautions

        # Load severity
        if os.path.exists(severity_path):
            sev_df = pd.read_csv(severity_path)
            self.symptom_severity = dict(zip(sev_df['Symptom'].str.strip(), sev_df['weight']))

        # Load main dataset
        df = pd.read_csv(dataset_path)
        
        # Extract unique symptoms
        cols = [i for i in df.columns if i != 'Disease']
        
        # Flatten all symptoms
        symptoms = df[cols].values.flatten()
        symptoms = [str(s).strip() for s in symptoms if pd.notna(s) and str(s).strip() != '']
        self.symptoms_list = sorted(list(set(symptoms)))
        self.disease_list = df['Disease'].unique().tolist()
        
        # Create binary feature matrix efficiently
        row_dicts = []
        for i in range(len(df)):
            row_dict = {s: 0 for s in self.symptoms_list}
            row_symptoms = df.iloc[i, 1:].values
            for symptom in row_symptoms:
                if pd.notna(symptom) and str(symptom).strip() != '':
                    row_dict[str(symptom).strip()] = 1
            row_dicts.append(row_dict)
            
        X = pd.DataFrame(row_dicts)
                    
        y = df['Disease']
        
        return X, y

    def train(self):
        print("Loading and preprocessing data...")
        X, y = self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model trained with accuracy: {acc:.4f}")
        
        self.is_trained = True
        
        # Save model and artifacts
        joblib.dump(self.model, os.path.join(DATA_DIR, "rf_model.joblib"))
        joblib.dump(self.symptoms_list, os.path.join(DATA_DIR, "symptoms_list.joblib"))
        
        return acc

    def load_model(self):
        model_path = os.path.join(DATA_DIR, "rf_model.joblib")
        symptoms_path = os.path.join(DATA_DIR, "symptoms_list.joblib")
        
        # We also need to load the dictionaries
        self.load_data() 
        
        if os.path.exists(model_path) and os.path.exists(symptoms_path):
            self.model = joblib.load(model_path)
            self.symptoms_list = joblib.load(symptoms_path)
            self.is_trained = True
            return True
        return False

    def predict(self, input_symptoms):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        # Clean input
        input_symptoms = [s.strip() for s in input_symptoms]
        
        X_input = pd.DataFrame(0, index=[0], columns=self.symptoms_list)
        for symptom in input_symptoms:
            if symptom in self.symptoms_list:
                X_input.loc[0, symptom] = 1
                
        prediction = self.model.predict(X_input)[0]
        
        # Get prediction probabilities for top 3
        probs = self.model.predict_proba(X_input)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_diseases = [(self.model.classes_[i], probs[i]) for i in top3_idx]
        
        description = self.disease_description.get(prediction.strip(), "No description available.")
        precautions = self.disease_precaution.get(prediction.strip(), ["No precautions listed."])
        
        return {
            'disease': prediction,
            'description': description,
            'precautions': precautions,
            'top3': top3_diseases
        }

if __name__ == "__main__":
    model = DiseasePredictionModel()
    model.train()
