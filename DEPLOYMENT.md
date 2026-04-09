# 🚀 Deployment Guide — Vercel

This guide explains how to deploy the **ATS Resume Analyzer** to Vercel with:
- **Frontend**: HTML/CSS/JS static site (Vercel free tier)
- **Backend**: FastAPI serverless functions (Vercel serverless)

---

## Option 1: Deploy Frontend + Backend on Same Vercel Project ⭐ (Recommended)

### Step 1: Connect GitHub to Vercel

1. Push your code to GitHub: https://github.com/pranavkarvekar/ATS-Resume-Analyzer
2. Go to [vercel.com](https://vercel.com) and sign in (or sign up)
3. Click **"Add New..."** → **"Project"**
4. Select your GitHub repository
5. Click **"Import"**

### Step 2: Configure Environment Variables

1. In the Vercel project settings, go to **Settings** → **Environment Variables**
2. Add your API key:
   - **Name**: `GROQ_API_KEY`
   - **Value**: `your_groq_api_key_here`
3. Click **"Save"**

### Step 3: Configure Build & Deploy

Vercel should auto-detect your project structure. Set the following:

**For Frontend (Root)**:
- **Build Command**: `echo 'Static frontend'`
- **Output Directory**: `.` (current directory)

**For Backend API (api/)**:
- This will auto-detect with the `api/vercel.json` config

### Step 4: Deploy

1. Click **"Deploy"**
2. Wait for the deployment to complete (~2-3 minutes)
3. Your URLs will be:
   - **Frontend**: `https://your-project.vercel.app`
   - **API**: `https://your-project.vercel.app/api`

### Step 5: Update app.js

After deployment, update the API URL in your frontend:

```javascript
// In app.js, update the fallback URL to your actual API:
return 'https://your-project.vercel.app'; // Your Vercel domain
```

---

## Option 2: Deploy Backend on Separate Vercel Project

If you prefer separate deployments:

### Create Backend-Only Project

1. Create a new folder: `ats-api` (copy `api/` folder contents here)
2. Create `vercel.json` in the root:
```json
{
  "builds": [
    { "src": "main.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "main.py" }
  ],
  "env": {
    "GROQ_API_KEY": "@GROQ_API_KEY"
  }
}
```

3. Push to GitHub in a separate repo: `ats-api`
4. Deploy to Vercel following the same steps as Option 1
5. Your backend URL: `https://ats-api.vercel.app`
6. Update `app.js` to point to this URL

### Deploy Frontend Separately

1. Keep the frontend files in your main repo
2. Deploy to Vercel
3. Update `app.js`:
```javascript
const API_BASE = 'https://ats-api.vercel.app';
```

---

## Option 3: Deploy Backend on Render (Free Alternative)

If you prefer using Render instead of Vercel for the backend:

1. Go to [render.com](https://render.com)
2. Create a new **Web Service**
3. Connect your GitHub repository
4. Set **Build Command**: `pip install -r api/requirements.txt`
5. Set **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `GROQ_API_KEY=your_key_here`
7. Deploy
8. Your backend URL: `https://ats-api-xxxx.onrender.com`
9. Update `app.js`:
```javascript
const API_BASE = 'https://ats-api-xxxx.onrender.com';
```

---

## 🔍 CORS Issues?

If you see CORS errors in the browser console, make sure the API has CORS configured:

✅ **Already configured in `api/main.py`**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Data Files on Vercel

**FAISS Index & Resume Dataset**: 

⚠️ **Important**: Vercel has a 50MB deployment size limit per function. If your `data/` folder is too large:

**Option A**: Remove the data files before deployment
```bash
# Before pushing to Vercel
rm -rf data/faiss.index
rm -rf Resume/Resume.csv
```
This will automatically fallback to non-RAG mode (still works!).

**Option B**: Use a database/cloud storage
- Store `Resume.csv` on AWS S3 or Google Cloud Storage
- Download at runtime in `api/main.py`

---

## ✅ Testing After Deployment

1. Visit your Vercel frontend: `https://your-project.vercel.app`
2. Paste a job description
3. Upload a test PDF
4. Click "Run Resume Review"
5. Check browser console for logs (F12 → Console)

---

## 🐛 Debugging

### Check API Health
Open browser DevTools (F12) and check:
```javascript
fetch('https://your-project.vercel.app/api/health').then(r => r.json()).then(console.log)
```

### Enable Request Logging
Update `app.js` to log API calls:
```javascript
console.log('🔌 Fetching:', `${API_BASE}/api/analyze`);
```

### Check Vercel Logs
1. Go to your Vercel project
2. Click **"Deployments"** 
3. Select the latest deployment
4. Click **"Logs"** to see real-time logs

---

## 📝 env.example File (Optional)

Create `.env.example` to document required variables:

```
GROQ_API_KEY=your_groq_api_key_from_console.groq.com
```

---

## 🎉 You're Done!

Your ATS Resume Analyzer is now live on Vercel! 

**Frontend URL**: https://your-project.vercel.app
**API URL**: https://your-project.vercel.app/api

Share it with others and get analyzing! 📊
