# Deployment Guide for Render

This guide will walk you through deploying the Data Visualization and Prediction application on Render.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Render Account Setup](#render-account-setup)
4. [Deployment Steps](#deployment-steps)
5. [Environment Variables Configuration](#environment-variables-configuration)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ A **GitHub**, **GitLab**, or **Bitbucket** account
- ✅ Your project code pushed to a repository
- ✅ A **MongoDB Atlas** account (or MongoDB instance)
- ✅ A **Gmail account** with App Password enabled (for email functionality)
- ✅ A **Render** account (sign up at [render.com](https://render.com))

---

## Pre-Deployment Checklist

### 1. Verify Files Are Ready

Ensure these files exist in your repository:

- ✅ `app.py` - Main Flask application
- ✅ `requirements.txt` - Python dependencies (without `bson` - it's included with `pymongo`)
- ✅ `Procfile` - Process definition for gunicorn
- ✅ `runtime.txt` - Python version specification (`python-3.11.0`)
- ✅ `render.yaml` - Render configuration (optional but recommended)
- ✅ `.gitignore` - Should include `config.json`

### 2. MongoDB Setup

1. Create a MongoDB Atlas account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (free tier is fine)
3. Create a database user:
   - Go to **Database Access** → **Add New Database User**
   - Choose **Password** authentication
   - Remember the username and password
4. Whitelist IP addresses:
   - Go to **Network Access** → **Add IP Address**
   - Click **Allow Access from Anywhere** (for Render) or add Render's IP ranges
5. Get your connection string:
   - Click **Connect** on your cluster
   - Choose **Connect your application**
   - Copy the connection string (looks like: `mongodb+srv://username:password@cluster.mongodb.net/dbname?retryWrites=true&w=majority`)
   - **Important**: Replace `<password>` with your actual database user password

### 3. Gmail App Password Setup

For email functionality to work, you need a Gmail App Password:

1. Go to your Google Account settings
2. Navigate to **Security** → **2-Step Verification** (enable if not already enabled)
3. Go to **App passwords** (or search for it)
4. Select **Mail** and **Other (Custom name)**
5. Enter "Render Deployment" as the name
6. Click **Generate**
7. Copy the 16-character password (you'll need this for `MAIL_PASSWORD`)

---

## Render Account Setup

1. Go to [render.com](https://render.com) and sign up/login
2. Connect your GitHub/GitLab/Bitbucket account:
   - Go to **Account Settings** → **Connected Accounts**
   - Connect your Git provider

---

## Deployment Steps

### Option A: Using render.yaml (Recommended)

If you have `render.yaml` in your repository:

1. **Push your code to GitHub/GitLab/Bitbucket**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create a new Web Service in Render**
   - In Render Dashboard, click **New +** → **Blueprint**
   - Connect your repository
   - Render will detect `render.yaml` and configure the service automatically
   - Click **Apply**

3. **Set Environment Variables** (see next section)

### Option B: Manual Setup (Without render.yaml)

1. **Push your code to GitHub/GitLab/Bitbucket**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create a new Web Service**
   - In Render Dashboard, click **New +** → **Web Service**
   - Connect your repository
   - Select the repository containing your project

3. **Configure the Service**
   - **Name**: `data-visualization-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or specify if code is in a subdirectory)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. **Select Plan**
   - Choose **Free** plan (or upgrade if needed)

5. **Click "Create Web Service"**

---

## Environment Variables Configuration

After creating the service, you need to set environment variables. Go to your service → **Environment** tab and add:

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.mongodb.net/dbname?retryWrites=true&w=majority` |
| `SECRET_KEY` | Flask secret key (for sessions) | Generate a random string (Render can auto-generate) |
| `MAIL_USERNAME` | Gmail address for sending emails | `your-email@gmail.com` |
| `MAIL_PASSWORD` | Gmail app password (16 chars) | `abcd efgh ijkl mnop` |
| `ADMIN_EMAIL` | Admin email for contact form | `admin@example.com` |

### Optional Variables (with defaults)

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `MAIL_SERVER` | `smtp.gmail.com` | SMTP server |
| `MAIL_PORT` | `465` | SMTP port |
| `MAIL_USE_TLS` | `false` | Use TLS |
| `MAIL_USE_SSL` | `true` | Use SSL |
| `FLASK_ENV` | `production` | Flask environment |

### How to Set Environment Variables in Render

1. Go to your service dashboard
2. Click on **Environment** tab (left sidebar)
3. Click **Add Environment Variable**
4. Enter the **Key** and **Value**
5. Click **Save Changes**
6. The service will automatically redeploy

### Generating a Secure SECRET_KEY

You can use Python to generate a secure secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Or use Render's auto-generate feature if available.

---

## Post-Deployment Verification

After deployment completes:

1. **Check Build Logs**
   - Go to **Logs** tab in your Render service
   - Verify that:
     - All packages installed successfully
     - No errors during build
     - Gunicorn started successfully

2. **Test the Application**
   - Open your Render URL (e.g., `https://your-app.onrender.com`)
   - Verify the homepage loads
   - Test registration/login functionality
   - Test file upload feature
   - Test prediction and visualization features

3. **Check Application Logs**
   - Monitor the **Logs** tab for any runtime errors
   - Check for database connection errors
   - Verify email sending (if applicable)

---

## Troubleshooting

### Build Fails

**Problem**: Build fails during `pip install`

**Solutions**:
- Check `requirements.txt` for syntax errors
- Verify all package names are correct
- Check if any packages are incompatible with Python 3.11
- Review build logs for specific error messages
- **Important**: Do NOT include `bson` as a separate package in `requirements.txt` - it's included with `pymongo` and will cause conflicts

### Application Crashes on Start

**Problem**: Service shows "Service unavailable" or crashes

**Solutions**:
- Check that `Procfile` exists and contains: `web: gunicorn app:app`
- Verify `gunicorn` is in `requirements.txt`
- Check application logs for Python errors
- Ensure `MONGO_URI` is correctly set
- Verify the connection string format is correct
- **ImportError with bson**: If you see `ImportError: cannot import name 'SON' from 'bson'`, remove `bson` from `requirements.txt` (it's included with `pymongo`)
- Verify `runtime.txt` exists with `python-3.11.0` to ensure correct Python version

### MongoDB Connection Issues

**Problem**: "Cannot connect to MongoDB" errors

**Solutions**:
- Verify `MONGO_URI` environment variable is set correctly
- Check MongoDB Atlas Network Access allows Render IPs (or 0.0.0.0/0)
- Verify database user credentials are correct
- Check if your MongoDB cluster is running
- Ensure the connection string includes the database name

### Email Not Working

**Problem**: Contact form emails not sending

**Solutions**:
- Verify `MAIL_USERNAME` and `MAIL_PASSWORD` are set
- Ensure you're using Gmail App Password (not regular password)
- Check that 2-Step Verification is enabled on Gmail
- Verify `MAIL_USE_SSL` is set to `true` and `MAIL_PORT` is `465`
- Check application logs for SMTP errors

### Static Files Not Loading

**Problem**: CSS/images not displaying

**Solutions**:
- Verify `static/` folder structure is correct
- Check that templates use correct paths (e.g., `/static/images/logo.png`)
- Clear browser cache
- Check Render logs for 404 errors on static files

### Port Binding Issues

**Problem**: "Address already in use" or port errors

**Solutions**:
- Render automatically sets the `PORT` environment variable
- Ensure your `app.py` uses: `port = int(os.getenv("PORT", 5000))`
- Don't hardcode port numbers in production code

### Slow Performance

**Problem**: Application is slow or times out

**Solutions**:
- Free tier has limitations (spins down after inactivity)
- Consider upgrading to a paid plan for better performance
- Optimize database queries
- Reduce file upload size limits if applicable
- Check MongoDB query performance

---

## Render-Specific Notes

### Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds (cold start)
- 512 MB RAM limit
- 100 GB bandwidth per month
- No persistent disk storage

### Custom Domain (Optional)

To use a custom domain:

1. Go to your service → **Settings** → **Custom Domains**
2. Add your domain
3. Update DNS records as instructed by Render
4. SSL certificate is automatically provisioned

### Health Checks

Render automatically performs health checks on:
- HTTP GET requests to the root path (`/`)
- If your app returns 200 OK, service is healthy

---

## Updating Your Deployment

To update your application after making changes:

1. **Commit and push your changes**
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```

2. **Automatic Deployment**
   - Render automatically detects pushes to the connected branch
   - A new deployment will start automatically
   - You can monitor progress in the **Logs** tab

3. **Manual Deployment** (if needed)
   - Go to your service dashboard
   - Click **Manual Deploy** → **Deploy latest commit**

---

## Security Best Practices

1. ✅ **Never commit `config.json`** with sensitive data
2. ✅ **Use environment variables** for all secrets
3. ✅ **Generate strong SECRET_KEY** (32+ characters, random)
4. ✅ **Use Gmail App Passwords** (not regular passwords)
5. ✅ **Restrict MongoDB network access** when possible
6. ✅ **Enable HTTPS** (Render provides this automatically)
7. ✅ **Regularly update dependencies** for security patches

---

## Support Resources

- [Render Documentation](https://render.com/docs)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)

---

## Quick Reference

### Essential Commands (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (development)
python app.py

# Run with gunicorn (production-like)
gunicorn app:app

# Test build process
pip install -r requirements.txt && gunicorn app:app
```

### Important File Locations

- **Main App**: `app.py`
- **Dependencies**: `requirements.txt`
- **Process File**: `Procfile`
- **Render Config**: `render.yaml`
- **Static Files**: `static/`
- **Templates**: `templates/`

---

## Next Steps After Deployment

1. ✅ Test all features thoroughly
2. ✅ Set up monitoring/alerting (optional)
3. ✅ Configure custom domain (optional)
4. ✅ Set up database backups
5. ✅ Document your deployment process
6. ✅ Share your deployed application URL

---

**Happy Deploying! 🚀**

If you encounter any issues not covered in this guide, check the Render documentation or application logs for more specific error messages.

