# from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, flash
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression
# from email_validator import validate_email, EmailNotValidError
# from sklearn.impute import SimpleImputer
# import pickle, io, traceback, os, uuid
# from datetime import datetime
# from typing import Dict
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from flask_pymongo import PyMongo
# from bson import ObjectId
# from urllib.parse import urlparse, urljoin
# from flask import Flask, abort, jsonify, request, redirect, url_for, session, render_template, flash
# from werkzeug.security import generate_password_hash, check_password_hash
# from bson.objectid import ObjectId
# from functools import wraps
# from pymongo import MongoClient
# import os, json
# import traceback
# from datetime import datetime
# import io
# import gridfs
# from bson import ObjectId
# from pymongo import MongoClient
# import base64

# # Custom helper imports
# from help_fun import load_csv_to_dataframe, clean_dataframe, infer_column_kinds, build_figure


# # -------------------- Flask Setup --------------------
# app = Flask(__name__)
# app.secret_key = "dev-secret-key"  # make sure this stays constant

# mongo_uri = "mongodb://localhost:27017/"
# client = MongoClient(mongo_uri)

# # Select DB from URI or explicitly
# db = client["ai_app_db"]
# fs = gridfs.GridFS(db)  # GridFS instance
# charts_collection = db["charts"]

# # Collections
# users_collection = db.users







# # ----------------- Routes -----------------
# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if "user" not in session:
#             flash("Please login first!", "danger")
#             return redirect(url_for("login"))
#         return f(*args, **kwargs)
#     return decorated_function

# @app.route("/register", methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#             if users_collection.find_one({"email": request.form["email"]}):
#                 flash("Email already exists! Please login.", "danger")
#                 return redirect(url_for("login"))

#             data = {
#                 "name": request.form["name"],
#                 "email": request.form["email"],
#                 "work": request.form["work"],
#                 "password": generate_password_hash(request.form["password"])
#             }
#             users_collection.insert_one(data)
#             flash("Personal account registered successfully!", "success")
#             return redirect(url_for("login"))

        
#     return render_template("register.html")

# # ----------- LOGIN -----------
# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form["email"]
#         password = request.form["password"]
#         user = users_collection.find_one({"email": email})
#         if user and check_password_hash(user["password"], password):
#                 session["user"] = str(user["_id"])
#                 flash(f"Welcome, {user['name']}!", "success")
#                 return redirect(url_for("index"))

     

#         flash("Invalid credentials!", "danger")

#     return render_template("login.html")

# # ----------- FORGOT PASSWORD -----------
# @app.route("/forgot", methods=["GET", "POST"])
# def forgot_password():
#     if request.method == "POST":
#         email = request.form["email"]
#         new_password = generate_password_hash(request.form["new_password"])

#         users_collection.update_one({"email": email}, {"$set": {"password": new_password}})
#         flash("Password updated successfully!", "success")
#         return redirect(url_for("login"))

        

#     return render_template("forgot.html")

# # ----------- EDIT PROFILE -----------
# @app.route("/edit", methods=["GET", "POST"])
# @login_required
# def edit_profile():
   
#     user = users_collection.find_one({"_id": ObjectId(session["user"])})
#     if request.method == "POST":
#             update_data = {
#                 "name": request.form["name"],
#                 "email": request.form["email"],
#             }
#             if request.form.get("password"):
#                 update_data["password"] = generate_password_hash(request.form["password"])
#             users_collection.update_one({"_id": user["_id"]}, {"$set": update_data})
#             flash("Profile updated successfully!", "success")
#             return redirect(url_for("profile"))
#     return render_template("edit_profile.html", user=user)


# # ----------- LOGOUT -----------
# @app.route("/logout")
# def logout():
#     session.clear()
#     flash("Logged out successfully!", "info")
#     return redirect(url_for("index"))

# # ----------- PROFILE -----------
# @app.route("/profile", methods=["GET", "POST"])
# @login_required
# def profile():
#     if session.get("type") == "personal":
#         user = users_collection.find_one({"_id": ObjectId(session["user"])})
#         if not user:
#             return "User not found", 404
       
#         return render_template("user_profile.html", user=user)

 

#     return redirect(url_for("login"))
# # -------------------- Global Variables --------------------
# data = None
# model = None
# le_dict = {}
# imputer = None
# column_info = {}
# DATAFRAMES: Dict[str, pd.DataFrame] = {}




# # -------------------- Protected Routes --------------------
# @app.route("/")
# @login_required
# def index():
#     return render_template("index.html")


# @app.route("/prediction_dashboard")
# @login_required
# def prediction_dashboard():
#     return render_template("prediction_dashboard.html")


# @app.route("/visualize_dashboard")
# @login_required
# def visualize_dashboard():
#     return render_template("visualize_dashboard.html")


# # -------------------- File Upload + ML Model --------------------
# @app.route("/upload_file", methods=["POST"])
# def upload_file():
#     global data, column_info
#     try:
#         file = request.files["file"]
#         if not file or file.filename == "":
#             return jsonify(error="No file selected"), 400

#         if file.filename.endswith(".csv"):
#             data = pd.read_csv(file)
#         elif file.filename.endswith((".xls", ".xlsx")):
#             data = pd.read_excel(file)
#         else:
#             return jsonify(error="Invalid file type. Please upload CSV or Excel files."), 400

#         if data.empty:
#             return jsonify(error="The uploaded file is empty"), 400

#         columns = data.columns.tolist()
#         column_info = {}
#         for col in columns:
#             unique_values = data[col].dropna().unique().tolist() if not data[col].dropna().empty else []
#             column_info[col] = {
#                 "dtype": str(data[col].dtype),
#                 "null_count": int(data[col].isnull().sum()),
#                 "unique_count": int(data[col].nunique()),
#                 "sample_values": data[col].dropna().head(3).tolist() if not data[col].dropna().empty else [],
#                 "unique_values": unique_values,
#                 "is_numerical": pd.api.types.is_numeric_dtype(data[col]),
#             }

#         return jsonify({
#             "columns": columns,
#             "column_info": column_info,
#             "shape": data.shape,
#             "message": f"File uploaded successfully! Found {len(columns)} columns and {data.shape[0]} rows."
#         })
#     except Exception as e:
#         return jsonify(error=f"Error processing file: {str(e)}"), 500


# @app.route("/train", methods=["POST"])
# def train_model():
#     global data, model, le_dict
#     try:
#         if data is None:
#             return jsonify(error="No data uploaded. Please upload a file first."), 400

#         target = request.form.get("target")
#         features = request.form.getlist("features[]")

#         if not target or not features:
#             return jsonify(error="Please select both target and feature columns."), 400

#         if target in features:
#             return jsonify(error="Target column cannot be the same as feature columns."), 400

#         df = data.copy()
#         le_dict = {}

#         for col in features + [target]:
#             if df[col].dtype == "object":
#                 df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
#             else:
#                 df[col] = df[col].fillna(df[col].median())

#         for col in features + [target]:
#             if df[col].dtype == "object":
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col].astype(str))
#                 le_dict[col] = le

#         X = df[features]
#         y = df[target]
#         model = LinearRegression().fit(X, y)
#         r2_score = model.score(X, y)

#         return jsonify({
#             "message": "Model trained successfully!",
#             "r2_score": round(r2_score, 4),
#             "features": features,
#             "target": target,
#         })
#     except Exception as e:
#         return jsonify(error=f"Error training model: {str(e)}"), 500


# @app.route("/predict", methods=["POST"])
# def predict():
#     global model, le_dict, data
#     try:
#         if model is None:
#             return jsonify(error="No model trained. Please train a model first."), 400

#         input_data = request.json
#         if not input_data:
#             return jsonify(error="No input data provided."), 400

#         df_input = pd.DataFrame([input_data])

#         for col in df_input.columns:
#             if col in le_dict:
#                 le = le_dict[col]
#                 df_input[col] = le.transform(df_input[col].astype(str))

#         df_input = df_input[model.feature_names_in_]
#         prediction = model.predict(df_input)[0]

#         return jsonify({
#             "prediction": round(float(prediction), 4),
#             "message": "Prediction successful!"
#         })
#     except Exception as e:
#         return jsonify(error=f"Error making prediction: {str(e)}"), 500


# # -------------------- Data Visualization --------------------
# @app.route("/upload", methods=["POST"])
# def upload():
#     file = request.files.get("file")
#     if not file or file.filename == "":
#         flash("Please choose a CSV file to upload.")
#         return redirect(url_for("index"))

#     try:
#         raw_df = load_csv_to_dataframe(file)
#         df = clean_dataframe(raw_df)
#         upload_id = str(uuid.uuid4())
#         DATAFRAMES[upload_id] = df
#         return redirect(url_for("configure", upload_id=upload_id))
#     except Exception as e:
#         flash(f"Failed to read CSV: {e}")
#         return redirect(url_for("index"))


# # @app.route("/configure/<upload_id>", methods=["GET", "POST"])
# # def configure(upload_id):
# #     df = DATAFRAMES.get(upload_id)
# #     if df is None:
# #         flash("Upload not found or expired. Please upload again.")
# #         return redirect(url_for("index"))

# #     kinds = infer_column_kinds(df)
# #     chart_types = [("line", "Line"), ("bar", "Bar"), ("pie", "Pie"), ("scatter", "Scatter"),
# #                    ("histogram", "Histogram"), ("box", "Box")]

# #     fig_html = None
# #     selected = {"chart": None, "x": None, "y": None, "color": None, "agg": None, "title": None, "palette": None}

# #     if request.method == "POST":
# #         try:
# #             fig = build_figure(
# #                 df,
# #                 request.form.get("chart"),
# #                 request.form.get("x") or None,
# #                 request.form.get("y") or None,
# #                 request.form.get("color") or None,
# #                 request.form.get("agg") or None,
# #                 request.form.get("title") or "",
# #                 request.form.get("palette") or None,
# #             )
# #             fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
# #         except Exception as e:
# #             flash(f"Failed to build chart: {e}")

# #     return render_template(
# #         "configure.html",
# #         upload_id=upload_id,
# #         columns=list(df.columns),
# #         kinds=kinds,
# #         chart_types=chart_types,
# #         fig_html=fig_html,
# #         selected=selected,
# #     )


# # @app.route("/download/<upload_id>", methods=["POST"])
# # def download_png(upload_id):
# #     df = DATAFRAMES.get(upload_id)
# #     if df is None:
# #         flash("Upload not found or expired. Please upload again.")
# #         return redirect(url_for("index"))

# #     fig = build_figure(
# #         df,
# #         request.form.get("chart"),
# #         request.form.get("x") or None,
# #         request.form.get("y") or None,
# #         request.form.get("color") or None,
# #         request.form.get("agg") or None,
# #         request.form.get("title") or "",
# #         request.form.get("palette") or None,
# #     )
# #     buf = io.BytesIO()
# #     fig.write_image(buf, format="png")
# #     buf.seek(0)
# #     filename = f"chart_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
# #     return send_file(buf, mimetype="image/png", as_attachment=True, download_name=filename)


# # @app.route("/save_chart/<upload_id>", methods=["POST"])
# # def save_chart(upload_id):
# #     # Get the dataframe
# #     df = DATAFRAMES.get(upload_id)
# #     if df is None:
# #         flash("Upload not found or expired. Please upload again.")
# #         return redirect(url_for("index"))

# #     # Replace this with actual logged-in username
# #     username = "John"  # Replace with current user from session or flask-login

# #     # Get form values
# #     chart_type = request.form.get("chart")
# #     x_col = request.form.get("x") or None
# #     y_col = request.form.get("y") or None
# #     color = request.form.get("color") or None
# #     agg = request.form.get("agg") or None
# #     title = request.form.get("title") or f"{chart_type} Chart"
# #     palette = request.form.get("palette") or None

# #     if not chart_type:
# #         flash("Please select a chart type!")
# #         return redirect(url_for("configure", upload_id=upload_id))

# #     try:
# #         # Build figure
# #         fig = build_figure(
# #             df,
# #             chart_type,
# #             x_col,
# #             y_col,
# #             color,
# #             agg,
# #             title,
# #             palette,
# #         )

# #         # Convert figure to Base64 string
# #         buf = io.BytesIO()
# #         fig.write_image(buf, format="png")
# #         buf.seek(0)
# #         img_base64 = base64.b64encode(buf.read()).decode("utf-8")

# #         # Prepare chart object
# #         chart_obj = {
# #             "name": title,
# #             "chart_type": chart_type,
# #             "x": x_col,
# #             "y": y_col,
# #             "color": color,
# #             "agg": agg,
# #             "palette": palette,
# #             "image_base64": img_base64,
# #             "created_at": datetime.utcnow()
# #         }

# #         # Save inside user's charts object
# #         user_doc = charts_collection.find_one({"username": username})
# #         if user_doc:
# #             # Generate next chart key
# #             existing_charts = user_doc.get("charts", {})
# #             chart_num = len(existing_charts) + 1
# #             chart_key = f"chart_{chart_num}"
# #             existing_charts[chart_key] = chart_obj

# #             # Update the document
# #             charts_collection.update_one(
# #                 {"username": username},
# #                 {"$set": {"charts": existing_charts}}
# #             )
# #         else:
# #             # Create new user document
# #             charts_collection.insert_one({
# #                 "username": username,
# #                 "charts": {"chart_1": chart_obj}
# #             })

# #         flash(f"Chart '{title}' saved successfully!")
# #         return redirect(url_for("configure", upload_id=upload_id))

# #     except Exception as e:
# #         flash(f"Failed to save chart: {e}")
# #         return redirect(url_for("configure", upload_id=upload_id))



# @app.route("/configure/<upload_id>", methods=["GET", "POST"])
# def configure(upload_id):
#     df = DATAFRAMES.get(upload_id)
#     if df is None:
#         flash("Upload not found or expired. Please upload again.")
#         return redirect(url_for("index"))

#     kinds = infer_column_kinds(df)
#     chart_types = [("line", "Line"), ("bar", "Bar"), ("pie", "Pie"),
#                    ("scatter", "Scatter"), ("histogram", "Histogram"), ("box", "Box")]

#     fig_html = None
#     selected = {"chart": "line", "x": None, "y": None, "color": None,
#                 "agg": None, "title": "", "palette": None}  # default chart type

#     if request.method == "POST":
#         # Read form data and ensure defaults
#         chart_type = request.form.get("chart") or "line"
#         x_col = request.form.get("x") or None
#         y_col = request.form.get("y") or None
#         color = request.form.get("color") or None
#         agg = request.form.get("agg") or None
#         title = request.form.get("title") or f"{chart_type} Chart"
#         palette = request.form.get("palette") or None

#         selected = {
#             "chart": chart_type,
#             "x": x_col,
#             "y": y_col,
#             "color": color,
#             "agg": agg,
#             "title": title,
#             "palette": palette
#         }

#         try:
#             fig = build_figure(df, chart_type, x_col, y_col, color, agg, title, palette)
#             fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
#         except Exception as e:
#             flash(f"Failed to build chart: {e}")

#     return render_template(
#         "configure.html",
#         upload_id=upload_id,
#         columns=list(df.columns),
#         kinds=kinds,
#         chart_types=chart_types,
#         fig_html=fig_html,
#         selected=selected
#     )


# @app.route("/download/<upload_id>", methods=["POST"])
# def download_png(upload_id):
#     df = DATAFRAMES.get(upload_id)
#     if df is None:
#         flash("Upload not found or expired. Please upload again.")
#         return redirect(url_for("index"))

#     # Use default chart type if missing
#     chart_type = request.form.get("chart") or "line"

#     fig = build_figure(
#         df,
#         chart_type,
#         request.form.get("x") or None,
#         request.form.get("y") or None,
#         request.form.get("color") or None,
#         request.form.get("agg") or None,
#         request.form.get("title") or "",
#         request.form.get("palette") or None,
#     )
#     buf = io.BytesIO()
#     fig.write_image(buf, format="png")
#     buf.seek(0)
#     filename = f"chart_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
#     return send_file(buf, mimetype="image/png", as_attachment=True, download_name=filename)



# @app.route("/save_chart/<upload_id>", methods=["POST"])
# def save_chart(upload_id):
#     df = DATAFRAMES.get(upload_id)
#     if df is None:
#         flash("Upload not found or expired. Please upload again.")
#         return redirect(url_for("index"))

#     # Get logged-in user ID from session
#     user_id = session.get("user")
#     if not user_id:
#         flash("Please log in first.")
#         return redirect(url_for("login"))

#     # Fetch user document from users_collection using ObjectId
#     user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
#     if not user_doc:
#         flash("User not found.")
#         return redirect(url_for("login"))

#     # Use email as identifier for storing charts
#     user_email = user_doc.get("email")

#     # Get form values
#     chart_type = request.form.get("chart") or "line"
#     x_col = request.form.get("x") or None
#     y_col = request.form.get("y") or None
#     color = request.form.get("color") or None
#     agg = request.form.get("agg") or None
#     title = request.form.get("title") or f"{chart_type} Chart"
#     palette = request.form.get("palette") or None

#     try:
#         # Build figure
#         fig = build_figure(df, chart_type, x_col, y_col, color, agg, title, palette)
#         buf = io.BytesIO()
#         fig.write_image(buf, format="png")
#         buf.seek(0)
#         img_base64 = base64.b64encode(buf.read()).decode("utf-8")

#         # Chart object
#         chart_obj = {
#             "name": title,
#             "chart_type": chart_type,
#             "x": x_col,
#             "y": y_col,
#             "color": color,
#             "agg": agg,
#             "palette": palette,
#             "image_base64": img_base64,
#             "created_at": datetime.utcnow()
#         }

#         # Store in charts_collection under user's email
#         user_chart_doc = charts_collection.find_one({"email": user_email})
#         if user_chart_doc:
#             existing_charts = user_chart_doc.get("charts", {})
#             chart_num = len(existing_charts) + 1
#             chart_key = f"chart_{chart_num}"
#             existing_charts[chart_key] = chart_obj
#             charts_collection.update_one(
#                 {"email": user_email},
#                 {"$set": {"charts": existing_charts}}
#             )
#         else:
#             charts_collection.insert_one({
#                 "email": user_email,
#                 "charts": {"chart_1": chart_obj}
#             })

#         flash(f"Chart '{title}' saved successfully!")
#         return redirect(url_for("configure", upload_id=upload_id))

#     except Exception as e:
#         flash(f"Failed to save chart: {e}")
#         return redirect(url_for("configure", upload_id=upload_id))


# # -------------------- Run App --------------------
# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from bson import ObjectId
from functools import wraps
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import io
import uuid
import base64

# Custom helpers
from help_fun import load_csv_to_dataframe, clean_dataframe, infer_column_kinds, build_figure

# -------------------- Flask Setup --------------------
app = Flask(__name__)
app.secret_key = "dev-secret-key"

# MongoDB Setup
mongo_uri = "mongodb://localhost:27017/"
db = PyMongo(app, uri=mongo_uri).cx["ai_app_db"]
users_collection = db["users"]
charts_collection = db["charts"]

# -------------------- Session / Login Helpers --------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login first!", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- Authentication --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        if users_collection.find_one({"email": email}):
            flash("Email already exists! Please login.", "danger")
            return redirect(url_for("login"))

        users_collection.insert_one({
            "name": request.form["name"],
            "email": email,
            "work": request.form.get("work", ""),
            "password": generate_password_hash(request.form["password"])
        })
        flash("Account registered successfully!", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user"] = str(user["_id"])
            flash(f"Welcome, {user['name']}!", "success")
            return redirect(url_for("index"))
        flash("Invalid credentials!", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("index"))

# -------------------- Profile --------------------
@app.route("/profile")
@login_required
def profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    return render_template("user_profile.html", user=user)

@app.route("/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    if request.method == "POST":
        update_data = {"name": request.form["name"], "email": request.form["email"]}
        if request.form.get("password"):
            update_data["password"] = generate_password_hash(request.form["password"])
        users_collection.update_one({"_id": user["_id"]}, {"$set": update_data})
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    return render_template("edit_profile.html", user=user)

@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        new_password = generate_password_hash(request.form["new_password"])
        users_collection.update_one({"email": email}, {"$set": {"password": new_password}})
        flash("Password updated successfully!", "success")
        return redirect(url_for("login"))
    return render_template("forgot.html")

# -------------------- Global Variables --------------------
DATAFRAMES = {}
model = None
le_dict = {}

# -------------------- Dashboard --------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/prediction_dashboard")
@login_required
def prediction_dashboard():
    return render_template("prediction_dashboard.html")

@app.route("/visualize_dashboard")
@login_required
def visualize_dashboard():
    return render_template("visualize_dashboard.html")

# -------------------- File Upload + ML --------------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    try:
        file = request.files["file"]
        if not file:
            return jsonify(error="No file selected"), 400

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return jsonify(error="Invalid file type."), 400

        if df.empty:
            return jsonify(error="The uploaded file is empty"), 400

        columns = df.columns.tolist()
        column_info = {}
        for col in columns:
            column_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist() if not df[col].dropna().empty else [],
                "is_numerical": pd.api.types.is_numeric_dtype(df[col]),
            }

        return jsonify({
            "columns": columns,
            "column_info": column_info,
            "shape": df.shape,
            "message": f"File uploaded successfully! Found {len(columns)} columns and {df.shape[0]} rows."
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/train", methods=["POST"])
def train_model():
    global model, le_dict
    try:
        if not DATAFRAMES:
            return jsonify(error="No data uploaded."), 400

        target = request.form.get("target")
        features = request.form.getlist("features[]")
        if not target or not features:
            return jsonify(error="Select target and features."), 400
        if target in features:
            return jsonify(error="Target cannot be in features."), 400

        df = DATAFRAMES.get("latest").copy() if "latest" in DATAFRAMES else pd.DataFrame()
        le_dict = {}

        for col in features + [target]:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

        for col in features + [target]:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

        X = df[features]
        y = df[target]
        model = LinearRegression().fit(X, y)
        r2_score = model.score(X, y)

        return jsonify({"message": "Model trained successfully!", "r2_score": round(r2_score, 4)})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify(error="No model trained."), 400
        df_input = pd.DataFrame([request.json])
        for col in df_input.columns:
            if col in le_dict:
                le = le_dict[col]
                df_input[col] = le.transform(df_input[col].astype(str))
        prediction = model.predict(df_input[model.feature_names_in_])[0]
        return jsonify({"prediction": round(float(prediction), 4)})
    except Exception as e:
        return jsonify(error=str(e)), 500

# -------------------- Chart Visualization --------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        flash("Please select a CSV file.")
        return redirect(url_for("index"))
    try:
        df = clean_dataframe(load_csv_to_dataframe(file))
        upload_id = str(uuid.uuid4())
        DATAFRAMES[upload_id] = df
        return redirect(url_for("configure", upload_id=upload_id))
    except Exception as e:
        flash(f"Failed to read CSV: {e}")
        return redirect(url_for("index"))

@app.route("/configure/<upload_id>", methods=["GET", "POST"])
@login_required
def configure(upload_id):
    df = DATAFRAMES.get(upload_id)
    if df is None:
        flash("Upload not found.")
        return redirect(url_for("index"))

    kinds = infer_column_kinds(df)
    chart_types = [("line", "Line"), ("bar", "Bar"), ("pie", "Pie"),
                   ("scatter", "Scatter"), ("histogram", "Histogram"), ("box", "Box")]

    fig_html = None
    selected = {"chart": "line", "x": None, "y": None, "color": None, "agg": None, "title": "", "palette": None}

    if request.method == "POST":
        chart_type = request.form.get("chart") or "line"
        x_col = request.form.get("x") or None
        y_col = request.form.get("y") or None
        color = request.form.get("color") or None
        agg = request.form.get("agg") or None
        title = request.form.get("title") or f"{chart_type} Chart"
        palette = request.form.get("palette") or None

        selected.update({"chart": chart_type, "x": x_col, "y": y_col, "color": color,
                         "agg": agg, "title": title, "palette": palette})

        try:
            fig = build_figure(df, chart_type, x_col, y_col, color, agg, title, palette)
            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as e:
            flash(f"Failed to build chart: {e}")

    return render_template("configure.html", upload_id=upload_id, columns=list(df.columns),
                           kinds=kinds, chart_types=chart_types, fig_html=fig_html, selected=selected)

@app.route("/download/<upload_id>", methods=["POST"])
@login_required
def download_png(upload_id):
    df = DATAFRAMES.get(upload_id)
    if df is None:
        flash("Upload not found.")
        return redirect(url_for("index"))

    fig = build_figure(
        df,
        request.form.get("chart") or "line",
        request.form.get("x") or None,
        request.form.get("y") or None,
        request.form.get("color") or None,
        request.form.get("agg") or None,
        request.form.get("title") or "",
        request.form.get("palette") or None,
    )
    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    filename = f"chart_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name=filename)

@app.route("/save_chart/<upload_id>", methods=["POST"])
@login_required
def save_chart(upload_id):
    df = DATAFRAMES.get(upload_id)
    if df is None:
        flash("Upload not found.")
        return redirect(url_for("index"))

    user_doc = users_collection.find_one({"_id": ObjectId(session["user"])})
    user_email = user_doc.get("email")

    chart_type = request.form.get("chart") or "line"
    x_col = request.form.get("x") or None
    y_col = request.form.get("y") or None
    color = request.form.get("color") or None
    agg = request.form.get("agg") or None
    title = request.form.get("title") or f"{chart_type} Chart"
    palette = request.form.get("palette") or None

    try:
        fig = build_figure(df, chart_type, x_col, y_col, color, agg, title, palette)
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        chart_obj = {"name": title, "chart_type": chart_type, "x": x_col, "y": y_col, "color": color,
                     "agg": agg, "palette": palette, "image_base64": img_base64, "created_at": datetime.utcnow()}

        user_chart_doc = charts_collection.find_one({"email": user_email})
        if user_chart_doc:
            charts = user_chart_doc.get("charts", {})
            chart_key = f"chart_{len(charts)+1}"
            charts[chart_key] = chart_obj
            charts_collection.update_one({"email": user_email}, {"$set": {"charts": charts}})
        else:
            charts_collection.insert_one({"email": user_email, "charts": {"chart_1": chart_obj}})

        flash(f"Chart '{title}' saved successfully!")
        return redirect(url_for("configure", upload_id=upload_id))
    except Exception as e:
        flash(f"Failed to save chart: {e}")
        return redirect(url_for("configure", upload_id=upload_id))

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
