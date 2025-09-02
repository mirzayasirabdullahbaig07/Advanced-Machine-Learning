# ============================================================
# ANACONDA, VIRTUAL ENVIRONMENTS & KAGGLE / COLAB
# ============================================================

# What is Anaconda?
# Anaconda is a distribution of Python and R focused on Data Science & ML.
# - It comes with Python + package manager (conda) + many ML libraries preinstalled.
# - It helps manage multiple environments and dependencies easily.

# ============================================================
# Installation of Anaconda
# ============================================================
# 1. Go to https://www.anaconda.com/download
# 2. Download installer for your OS (Windows/Linux/Mac).
# 3. Run the installer → Install for "Just Me" → Add Anaconda to PATH (recommended).
# 4. Verify installation in terminal / Anaconda Prompt:
#       conda --version
#       python --version

# ============================================================
# What is a Virtual Environment?
# ============================================================
# A virtual environment is an isolated workspace for Python projects.
# - Each project can have its own dependencies (libraries & versions).
# - Prevents conflicts between different projects.
#
# Example:
#   Project A needs numpy==1.20
#   Project B needs numpy==1.25
# Without environments → conflicts
# With environments → no problem

# ============================================================
# Create & Activate Virtual Environment (with conda)
# ============================================================
# Command to create a virtual environment:
# (replace "machinelearning" with your env name)
#
#   conda create --name machinelearning python=3.10
#
# Activate the environment:
#   conda activate machinelearning
#
# Deactivate environment:
#   conda deactivate
#
# List all environments:
#   conda env list

# Install libraries inside env:
#   conda install numpy pandas matplotlib scikit-learn
#
# OR using pip inside env:
#   pip install tensorflow torch keras seaborn

# ============================================================
# Kaggle for ML/DL
# ============================================================
# Kaggle is an online platform for:
# - Datasets (millions of free datasets for ML/DL)
# - Competitions (solve ML challenges, win prizes)
# - Notebooks (cloud Jupyter notebooks with GPU support)

# Steps:
# 1. Create account → https://www.kaggle.com
# 2. Install Kaggle API (to download datasets in Colab/Local):
#       pip install kaggle
# 3. Generate API key → Account Settings → Create New API Token
#    This downloads kaggle.json (keep it safe).
# 4. Place kaggle.json in the correct folder:
#       Linux/Mac: ~/.kaggle/kaggle.json
#       Windows: C:\Users\<username>\.kaggle\kaggle.json
# 5. Use in code:
#       from kaggle.api.kaggle_api_extended import KaggleApi
#       api = KaggleApi()
#       api.authenticate()
#       api.dataset_download_files('zynicide/wine-reviews', path='./data', unzip=True)

# ============================================================
# Google Colab
# ============================================================
# Google Colab = free cloud-based Jupyter notebook with GPU/TPU support.
# - No installation required (runs in browser).
# - Good for ML/DL projects when you don’t have a powerful PC.
#
# How to use:
# 1. Go to https://colab.research.google.com
# 2. Sign in with Google account.
# 3. Create new notebook.
# 4. Install libraries (if not preinstalled):
#       !pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch
# 5. Mount Google Drive for storage:
#       from google.colab import drive
#       drive.mount('/content/drive')
#
# You can also connect Kaggle with Colab by uploading kaggle.json file.

# ============================================================
# Summary
# ============================================================
# - Anaconda = Python + package/environment manager for ML.
# - Virtual Env = isolated workspace for projects (no dependency conflicts).
# - Kaggle = datasets + competitions + notebooks.
# - Colab = free GPU cloud Jupyter notebooks.
