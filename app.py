from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file, abort
from flask_pymongo import PyMongo
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Email
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from functools import wraps
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import io
import uuid
import base64
import json

# -------------------- Load Config --------------------
with open("config.json") as f:
    config = json.load(f)

# -------------------- Flask App Setup --------------------
app = Flask(__name__)
app.secret_key = config.get("SECRET_KEY", "dev-secret-key")

# -------------------- MongoDB Setup --------------------
app.config["MONGO_URI"] = config["MONGO_URI"]
mongo = PyMongo(app)
db = mongo.db
users_collection = db["users"]
charts_collection = db["charts"]
contacts_collection = db["contacts"]

# -------------------- Flask-Mail Setup --------------------
app.config.update(
    MAIL_SERVER=config["mail_server"],
    MAIL_PORT=config["mail_port"],
    MAIL_USE_TLS=config["mail_use_tls"],
    MAIL_USE_SSL=config["mail_use_ssl"],
    MAIL_USERNAME=config["gmail_user"],
    MAIL_PASSWORD=config["gmail_password"]
)
mail = Mail(app)

# -------------------- Timezone --------------------
india_tz = pytz.timezone("Asia/Kolkata")

# -------------------- Global Variables --------------------
DATAFRAMES = {}
model = None
le_dict = {}

# -------------------- Login Required Decorator --------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login first!", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- Contact Form --------------------
class ContactForm(FlaskForm):
    name = StringField("Full Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    subject = StringField("Subject", validators=[DataRequired()])
    reason = SelectField("Reason", choices=[("", "Select"), ("General Inquiry", "General Inquiry"),
                                            ("Support", "Support"), ("Partnership", "Partnership"),
                                            ("Feedback", "Feedback")])
    phone = StringField("Phone")
    company = StringField("Company")
    message = TextAreaField("Message", validators=[DataRequired()])

# -------------------- Authentication Routes --------------------
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
@login_required
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    user_charts_doc = charts_collection.find_one({"email": user["email"]})
    charts = []
    if user_charts_doc and "charts" in user_charts_doc:
        charts = [{"id": k, **v} for k, v in user_charts_doc["charts"].items()]
    return render_template("profile.html", user=user, charts=charts)


@app.route("/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    if request.method == "POST":
        update_data = {
            "name": request.form.get("name"),
            "work": request.form.get("work"),
            "email": request.form.get("email")
        }
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

# -------------------- Dashboard Routes --------------------
@app.route("/")
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

# -------------------- File Upload & ML --------------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify(error="No file selected"), 400
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return jsonify(error="Invalid file type."), 400
        if df.empty:
            return jsonify(error="The uploaded file is empty"), 400
        DATAFRAMES["latest"] = df
        columns = df.columns.tolist()
        column_info = {}
        for col in columns:
            column_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).astype(str).tolist(),
                "is_numerical": pd.api.types.is_numeric_dtype(df[col]),
                "is_date": pd.api.types.is_datetime64_any_dtype(df[col]),
                "unique_values": df[col].dropna().unique().astype(str).tolist()[:10]
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
        target = request.form.get("target")
        features = request.form.getlist("features[]")
        if not target or not features:
            return jsonify(error="Select target and features."), 400
        if target in features:
            return jsonify(error="Target cannot be in features."), 400
        if "latest" not in DATAFRAMES:
            return jsonify(error="No data uploaded."), 400
        df = DATAFRAMES["latest"].copy()
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
        return jsonify({
            "message": "Model trained successfully!",
            "r2_score": round(r2_score, 4),
            "sample_count": len(df),
            "feature_count": len(features)
        })
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify(error="No model trained."), 400
        data = request.get_json()
        df_input = pd.DataFrame([data])
        for col in df_input.columns:
            if col in le_dict:
                le = le_dict[col]
                df_input[col] = df_input[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df_input[col] = le.transform(df_input[col])
        prediction = model.predict(df_input[model.feature_names_in_])[0]
        return jsonify({"prediction": round(float(prediction), 4), "message": "Prediction successful!"})
    except Exception as e:
        return jsonify(error=str(e)), 500

# -------------------- Contact Form --------------------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        try:
            contact_entry = {field.name: field.data for field in form}
            contacts_collection.insert_one(contact_entry)
            msg = Message(
                subject=f"New Contact: {form.subject.data}",
                sender=config["gmail_user"],
                recipients=[config["ADMIN_EMAIL"]],
                body=f"""
Name: {form.name.data}
Email: {form.email.data}
Reason: {form.reason.data}
Phone: {form.phone.data}
Company: {form.company.data}
Message: {form.message.data}
"""
            )
            mail.send(msg)
            flash("Message sent successfully!", "success")
            return redirect(url_for("contact"))
        except Exception as e:
            print(e)
            flash("Failed to send message.", "danger")
    return render_template("contact.html", form=form)

# -------------------- After Request & Error Handlers --------------------
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    print(f"[ERROR] Internal server error: {e}")
    return render_template("500.html"), 500

# -------------------- Run Flask App --------------------
if __name__ == "__main__":
    app.run(debug=True)
