
from flask import Flask, abort, render_template, request, jsonify, redirect, url_for, send_file, flash, session
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
import pytz
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Email
from flask_mail import Mail, Message
import json

import os


india_tz = pytz.timezone("Asia/Kolkata")

# Custom helpers
from help_fun import load_csv_to_dataframe, clean_dataframe, infer_column_kinds, build_figure
app = Flask(__name__)

# Load configuration from environment variables or config.json
config = {}
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    # If config.json doesn't exist, use environment variables
    pass

# -------------------- MongoDB Setup --------------------
# Use environment variable if available, otherwise fallback to config.json
uri = os.getenv("MONGO_URI", config.get("MONGO_URI1", config.get("MONGO_URI", "")))
if not uri:
    raise ValueError("MONGO_URI environment variable or MONGO_URI1/MONGO_URI in config.json is required")
app.config["MONGO_URI"] = uri

mongo = PyMongo(app)
db = mongo.db
users_collection = db["users"]
charts_collection = db["charts"]
contacts_collection = db["contacts"]

# -------------------- Flask-Mail Setup --------------------
app.secret_key = os.getenv("SECRET_KEY", config.get("SECRET_KEY", "dev-secret-key-change-in-production"))



# Flask-Mail configuration for Gmail SSL (use environment variables with fallback to config.json)
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER", config.get("mail_server", "smtp.gmail.com")),
    MAIL_PORT=int(os.getenv("MAIL_PORT", config.get("mail_port", 465))),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", str(config.get("mail_use_tls", False))).lower() == "true",
    MAIL_USE_SSL=os.getenv("MAIL_USE_SSL", str(config.get("mail_use_ssl", True))).lower() == "true",
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", config.get("gmail_user", "")),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", config.get("gmail_password", "")),
)
mail = Mail(app)

# Flask-WTF Form
class ContactForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    subject = StringField('Subject', validators=[DataRequired()])
    message = TextAreaField('Message', validators=[DataRequired()])

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
@login_required

def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("index"))

# -------------------- Profile --------------------
@app.route("/profile")
@login_required
def profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    # Get the logged-in user
   # Find all charts linked to this user's email
    user_charts_doc = charts_collection.find_one({"email": user["email"]})
    
    # Convert embedded chart objects into a list
    charts = []
    if user_charts_doc and "charts" in user_charts_doc:
        charts = [
            {"id": chart_id, **chart_data}
            for chart_id, chart_data in user_charts_doc["charts"].items()
        ]
    
    return render_template("profile.html", user=user, charts=charts)





@app.route("/view_chart/<chart_id>")
@login_required
def view_chart(chart_id):
    # Fetch logged-in user
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    
    # Fetch charts document by email
    charts_doc = charts_collection.find_one({"email": user["email"]})
    if not charts_doc or "charts" not in charts_doc:
        abort(404)
    
    # Find the requested chart
    chart = charts_doc["charts"].get(chart_id)
    if not chart:
        abort(404)
    
    # Pass chart to the template
    return render_template("view_chart.html", chart=chart, chart_id=chart_id)

@app.route("/delete_chart/<chart_id>", methods=["POST"])
@login_required
def delete_chart(chart_id):
    # Get logged-in user
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    if not user:
        abort(401)

    # Find user's chart document by email
    charts_doc = charts_collection.find_one({"email": user["email"]})
    if not charts_doc:
        abort(404)

    # Ensure chart exists before deleting
    if chart_id not in charts_doc.get("charts", {}):
        flash("Chart not found.", "error")
        return redirect(url_for("profile"))

    # Remove the specific chart key
    charts_collection.update_one(
        {"email": user["email"]},
        {"$unset": {f"charts.{chart_id}": ""}}
    )

    flash("Chart deleted successfully!", "success")
    return redirect(url_for("profile"))


@app.route("/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    user = users_collection.find_one({"_id": ObjectId(session["user"])})
    if request.method == "POST":
        update_data = {
            "name": request.form.get("name"),
            "work": request.form.get("work"),
        }

       
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

# -------------------- Contact --------------------
# Contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if request.method == 'POST':
        try:
            # Save to MongoDB
            contact_entry = {
                "name": request.form['name'],
                "email": request.form['email'],
                "subject": request.form['subject'],
                "reason": request.form.get('reason'),
                "phone": request.form.get('phone'),
                "company": request.form.get('company'),
                "message": request.form['message']
            }
            contacts_collection.insert_one(contact_entry)

            # Send email
            admin_email = os.getenv("ADMIN_EMAIL", config.get("ADMIN_EMAIL", ""))
            msg = Message(
                subject=f"New Contact: {request.form['subject']}",
                sender=os.getenv("MAIL_USERNAME", config.get("gmail_user", "")),
                recipients=[admin_email],
                body=f"""
                    Name: {request.form['name']}
                    Email: {request.form['email']}
                    Reason: {request.form.get('reason')}
                    Phone: {request.form.get('phone')}
                    Company: {request.form.get('company')}
                    Message: {request.form['message']}
                                    """
            )
            mail.send(msg)

            return jsonify({'status': 'success', 'message': 'Contact Request sent successfully!'})
        except Exception as e:
            print(e)
            return jsonify({'status': 'error', 'message': 'Failed to send Contact Request.'})

    return render_template('contact.html', form=form)

# -------------------- Global Variables --------------------
DATAFRAMES = {}
model = None
le_dict = {}
target=None

# -------------------- Dashboard --------------------
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


@app.route("/upload_file", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
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
                "unique_values": df[col].dropna().unique().astype(str).tolist()[:10],
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
    global model, le_dict,target
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
    global model, le_dict,target

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

        return jsonify({
            "target": target,
            "prediction": round(float(prediction), 4),
            "message": "Prediction successful!"
        })
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
        title = request.form.get("title") or f"{chart_type} Chart"

        selected.update({"chart": chart_type, "x": x_col, "y": y_col, "title": title})

        try:
            fig = build_figure(df, chart_type, x_col, y_col, title)
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
        request.form.get("title") or "",
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
   
    title = request.form.get("title") or f"{chart_type} Chart"

    try:
        fig = build_figure(df, chart_type, x_col, y_col,  title)
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        chart_obj = {"name": title, "chart_type": chart_type, "x": x_col, "y": y_col, 
                      "image_base64": img_base64, "created_at":  datetime.now(india_tz).strftime("%Y-%m-%d %H:%M:%S")}

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






if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
