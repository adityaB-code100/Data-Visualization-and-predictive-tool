from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pickle
import io
import traceback

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import plotly.express as px
from help_fun import load_csv_to_dataframe, clean_dataframe, infer_column_kinds, build_figure

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
# Global variables to store uploaded data and model
data = None
model = None
le_dict = {}  # to store LabelEncoders for each column
imputer = None
column_info = {}  # to store column information

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction_dashboard')
def prediction_dashboard():
    return render_template('prediction_dashboard.html')

@app.route('/visualize_dashboard')
def visualize_dashboard():
    return render_template('visualize_dashboard.html')




@app.route('/upload_file', methods=['POST'])
def upload_file():
    global data, column_info
    try:
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify(error="No file selected"), 400
            
        # Reset global variables
        data = None
        column_info = {}
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify(error="Invalid file type. Please upload CSV or Excel files."), 400

        if data.empty:
            return jsonify(error="The uploaded file is empty"), 400
            
        # Analyze columns and store information
        columns = data.columns.tolist()
        column_info = {}
        for col in columns:
            # Get all unique values for categorical columns
            unique_values = data[col].dropna().unique().tolist() if not data[col].dropna().empty else []
            column_info[col] = {
                'dtype': str(data[col].dtype),
                'null_count': int(data[col].isnull().sum()),
                'unique_count': int(data[col].nunique()),
                'sample_values': data[col].dropna().head(3).tolist() if not data[col].dropna().empty else [],
                'unique_values': unique_values,
                'is_numerical': pd.api.types.is_numeric_dtype(data[col])
            }
        
        return jsonify({
            'columns': columns,
            'column_info': column_info,
            'shape': data.shape,
            'message': f"File uploaded successfully! Found {len(columns)} columns and {data.shape[0]} rows."
        })
        
    except Exception as e:
        return jsonify(error=f"Error processing file: {str(e)}"), 500

@app.route('/train', methods=['POST'])
def train_model():
    global data, model, le_dict, imputer
    try:
        if data is None:
            return jsonify(error="No data uploaded. Please upload a file first."), 400
            
        target = request.form.get('target')
        features = request.form.getlist('features[]')
        
        if not target or not features:
            return jsonify(error="Please select both target and feature columns."), 400
            
        if target in features:
            return jsonify(error="Target column cannot be the same as feature columns."), 400
            
        # Check if all selected columns exist
        missing_cols = [col for col in features + [target] if col not in data.columns]
        if missing_cols:
            return jsonify(error=f"Columns not found: {', '.join(missing_cols)}"), 400
        
        # Validate that target column is numerical
        if not pd.api.types.is_numeric_dtype(data[target]):
            return jsonify(error=f"Target column '{target}' must be numerical. Please select a numerical column for prediction."), 400
        
        df = data.copy()
        le_dict = {}
        
        # Handle missing values first
        for col in features + [target]:
            if df[col].dtype == 'object':
                # For categorical columns, fill with mode
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna('Unknown')
            else:
                # For numerical columns, fill with median
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical features
        for col in features + [target]:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

        X = df[features]
        y = df[target]
        
        # Check for any remaining NaN values
        if X.isnull().any().any() or y.isnull().any():
            return jsonify(error="Unable to handle all missing values. Please check your data."), 400
        
        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R² score for model evaluation
        r2_score = model.score(X, y)
        
        return jsonify({
            'message': "Model trained successfully!",
            'features': features,
            'target': target,
            'r2_score': round(r2_score, 4),
            'feature_count': len(features),
            'sample_count': len(X)
        })
        
    except Exception as e:
        return jsonify(error=f"Error training model: {str(e)}"), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, le_dict, data
    try:
        if model is None:
            return jsonify(error="No model trained. Please train a model first."), 400
            
        input_data = request.json
        if not input_data:
            return jsonify(error="No input data provided."), 400
        
        # Create DataFrame from input
        df_input = pd.DataFrame([input_data])
        
        # Handle missing values in input
        for col in df_input.columns:
            if col in data.columns:
                if data[col].dtype == 'object':
                    # For categorical columns, use mode if missing
                    mode_value = data[col].mode()
                    if not mode_value.empty and pd.isna(df_input[col].iloc[0]):
                        df_input[col] = mode_value[0]
                    elif pd.isna(df_input[col].iloc[0]):
                        df_input[col] = 'Unknown'
                else:
                    # For numerical columns, use median if missing
                    if pd.isna(df_input[col].iloc[0]):
                        df_input[col] = data[col].median()
        
        # Encode categorical features if any
        for col in df_input.columns:
            if col in le_dict:
                le = le_dict[col]
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except ValueError as e:
                    return jsonify(error=f"Invalid value for column '{col}': {str(e)}"), 400
        
        # Ensure all required features are present
        required_features = [col for col in model.feature_names_in_] if hasattr(model, 'feature_names_in_') else []
        missing_features = [col for col in required_features if col not in df_input.columns]
        if missing_features:
            return jsonify(error=f"Missing required features: {', '.join(missing_features)}"), 400
        
        # Reorder columns to match training data
        if hasattr(model, 'feature_names_in_'):
            df_input = df_input[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(df_input)[0]
        
        return jsonify({
            'prediction': round(float(prediction), 4),
            'input_data': input_data,
            'message': "Prediction completed successfully!"
        })
        
    except Exception as e:
        return jsonify(error=f"Error making prediction: {str(e)}"), 500
    







# In-memory store for uploaded dataframes. In production, prefer a persistent cache or object storage.
DATAFRAMES: Dict[str, pd.DataFrame] = {}




@app.route("/upload", methods=["POST"]) 
def upload():
	file = request.files.get("file")
	if not file or file.filename == "":
		flash("Please choose a CSV file to upload.")
		return redirect(url_for("index"))
	try:
		raw_df = load_csv_to_dataframe(file)
		df = clean_dataframe(raw_df)
		upload_id = str(uuid.uuid4())
		DATAFRAMES[upload_id] = df
		return redirect(url_for("configure", upload_id=upload_id))
	except Exception as e:
		flash(f"Failed to read CSV: {e}")
		return redirect(url_for("index"))


@app.route("/configure/<upload_id>", methods=["GET", "POST"]) 
def configure(upload_id: str):
	df = DATAFRAMES.get(upload_id)
	if df is None:
		flash("Upload not found or expired. Please upload again.")
		return redirect(url_for("index"))

	kinds = infer_column_kinds(df)
	chart_types = [
		("line", "Line"),
		("bar", "Bar"),
		("pie", "Pie"),
		("scatter", "Scatter"),
		("histogram", "Histogram"),
		("box", "Box"),
	]

	fig_html = None
	selected = {"chart": None, "x": None, "y": None, "color": None, "agg": None, "title": None, "palette": None}

	if request.method == "POST":
		chart = request.form.get("chart")
		x = request.form.get("x") or None
		y = request.form.get("y") or None
		color = request.form.get("color") or None
		agg = request.form.get("agg") or None
		title = request.form.get("title") or ""
		palette = request.form.get("palette") or None
		selected = {"chart": chart, "x": x, "y": y, "color": color, "agg": agg, "title": title, "palette": palette}
		try:
			fig = build_figure(df, chart, x, y, color, agg, title, palette)
			fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
		except Exception as e:
			flash(f"Failed to build chart: {e}")

	return render_template(
		"configure.html",
		upload_id=upload_id,
		columns=list(df.columns),
		kinds=kinds,
		chart_types=chart_types,
		fig_html=fig_html,
		selected=selected,
	)


@app.route("/download/<upload_id>", methods=["POST"]) 
def download_png(upload_id: str):
	df = DATAFRAMES.get(upload_id)
	if df is None:
		flash("Upload not found or expired. Please upload again.")
		return redirect(url_for("index"))
	chart = request.form.get("chart")
	x = request.form.get("x") or None
	y = request.form.get("y") or None
	color = request.form.get("color") or None
	agg = request.form.get("agg") or None
	title = request.form.get("title") or ""
	palette = request.form.get("palette") or None
	fig = build_figure(df, chart, x, y, color, agg, title, palette)
	buf = io.BytesIO()
	fig.write_image(buf, format="png")
	buf.seek(0)
	filename = f"chart_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
	return send_file(buf, mimetype="image/png", as_attachment=True, download_name=filename)






if __name__ == '__main__':
    app.run(debug=True)
