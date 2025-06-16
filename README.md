# 🎨 Flux ComfyUI Installer - Ultimate AI Image Generation Setup

<div align="center">

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/your-repo/flux-comfyui-installer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)](https://github.com/your-repo/flux-comfyui-installer)

**Automated installer for ComfyUI with Flux models - Get up and running in minutes!**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎮 User Guide](#-using-comfyui-with-flux) • [🔧 Troubleshooting](#-troubleshooting) • [🤝 Community](#-community--support)

</div>

---

## 📸 **Preview: Ultra-Realistic Results**

<div align="center">

| **Input Prompt** | **Generated Result** | **Generation Time** |
|:---------------:|:-------------------:|:------------------:|
| *"Professional headshot of a 25-year-old woman with natural lighting"* | ![Portrait Example](https://via.placeholder.com/300x300/667eea/ffffff?text=Ultra+Realistic+Portrait) | **~15 seconds** |
| *"Elderly man with weathered features and kind eyes"* | ![Elderly Example](https://via.placeholder.com/300x300/764ba2/ffffff?text=Character+Detail) | **~15 seconds** |
| *"Majestic landscape with mountains and lake, golden hour"* | ![Landscape Example](https://via.placeholder.com/300x300/f093fb/ffffff?text=Landscape+Quality) | **~20 seconds** |

*Generate images with unprecedented realism using Flux models through ComfyUI's professional interface*

</div>

---

## ✨ **What Makes This Installer Special**

### 🎯 **Revolutionary Technology**
- **Flux AI Models**: State-of-the-art FLUX.1-Schnell and FLUX.1-Dev models
- **ComfyUI Professional**: Full-featured node-based workflow interface
- **GPU Acceleration**: Optimized for NVIDIA, AMD, and Intel GPUs
- **Automated Setup**: Complete installation with one Python script

### 💻 **User-Friendly Installation**
- **One-Command Setup**: Single script installs everything automatically
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Smart Dependencies**: Automatically handles Python environments and packages
- **Model Management**: Pre-configured with working Flux models

### 🔧 **Professional Features**
- **Custom Workflows**: Includes portrait, landscape, and basic generation workflows
- **User Directories**: Easy-access folders for models, LoRAs, and outputs
- **ComfyUI Manager**: Pre-installed for easy node management
- **Startup Scripts**: Simple launch scripts for immediate use

---

## 📋 **System Requirements**

### **🟢 Minimum Requirements**
```
Operating System: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
GPU:              NVIDIA GTX 1060 6GB / AMD RX 580 8GB / Intel Arc A380
RAM:              8GB system memory
Storage:          15GB free space (SSD recommended)
Python:           3.8+ (installer will guide you if missing)
Network:          Broadband internet (for model downloads)
```

### **🔵 Recommended Configuration**
```
GPU:              NVIDIA RTX 3070 8GB+ / AMD RX 6700 XT+ / Intel Arc A750+
RAM:              16GB+ system memory
Storage:          25GB+ SSD space
Python:           3.9+ with pip
Network:          High-speed broadband (100+ Mbps)
```

### **🟡 Optimal Performance**
```
GPU:              NVIDIA RTX 4080+ / RTX 3090+ (12GB+ VRAM)
RAM:              32GB+ system memory  
Storage:          NVMe SSD with 50GB+ free space
```

---

## 🚀 **Quick Start**

### **📦 Automated Installation - Everything in One Script!**

The Flux ComfyUI Installer handles all the complex setup automatically. Just run one command!

#### **Prerequisites Installation**

Before running the installer, ensure you have the required tools:

**Windows (using Chocolatey - Recommended):**
```powershell
# Install Chocolatey (if not already installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install required tools
choco install python git -y

# Verify installations
python --version    # Should be 3.8+
git --version       # Any recent version
```

**Windows (Manual Installation):**
1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Git**: Download from [git-scm.com](https://git-scm.com/)

**Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install required packages
sudo apt install python3 python3-pip python3-venv git -y

# Verify installations
python3 --version   # Should be 3.8+
git --version       # Any recent version
```

**Linux (CentOS/RHEL/Fedora):**
```bash
# For CentOS/RHEL
sudo yum install python3 python3-pip git -y

# For Fedora
sudo dnf install python3 python3-pip git -y

# Verify installations
python3 --version
git --version
```

**macOS (using Homebrew - Recommended):**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python git

# Verify installations
python3 --version   # Should be 3.8+
git --version       # Any recent version
```

**macOS (Manual Installation):**
1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Git**: Download from [git-scm.com](https://git-scm.com/) or install Xcode Command Line Tools

#### **Windows Installation**

```powershell
# Download the installer
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/your-repo/flux-comfyui-installer/main/flux_comfyui_installer.py" -OutFile "flux_comfyui_installer.py"

# Run the automated installer
python flux_comfyui_installer.py

# Start ComfyUI when installation completes
cd flux-comfyui
start_comfyui.bat

# Open browser to: http://localhost:8188
# If localhost doesn't work, try: http://0.0.0.0:8188 or http://127.0.0.1:8188
```

#### **Linux/macOS Installation**

```bash
# Download the installer
curl -O https://raw.githubusercontent.com/your-repo/flux-comfyui-installer/main/flux_comfyui_installer.py

# Run the automated installer
python3 flux_comfyui_installer.py

# Start ComfyUI when installation completes
cd flux-comfyui
./start_comfyui.sh

# Open browser to: http://localhost:8188
# If localhost doesn't work, try: http://0.0.0.0:8188 or http://127.0.0.1:8188
```

**That's it!** The installer handles everything automatically:
- ✅ Downloads and sets up ComfyUI
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies (PyTorch, etc.)
- ✅ Downloads Flux models
- ✅ Creates ready-to-use workflows
- ✅ Sets up convenient startup scripts

**Then open your browser to http://localhost:8188 to access ComfyUI!**  
*(If localhost doesn't work, try http://0.0.0.0:8188 or http://127.0.0.1:8188)*

---

## 🔨 **Installation Options**

### **🎯 Basic Installation (Recommended)**
```bash
# Full installation with all models
python flux_comfyui_installer.py
```

### **⚡ Quick Test Installation**
```bash
# Skip model downloads for faster setup
python flux_comfyui_installer.py --skip-models
```

### **🎨 Custom Installation**
```bash
# Custom directory name
python flux_comfyui_installer.py --install-dir "my-flux-studio"

# See all options
python flux_comfyui_installer.py --help
```

### **🔍 Installation Process**

The installer automatically performs these steps:

1. **✅ Prerequisites Check** - Validates Python, Git, disk space
2. **📁 Directory Setup** - Creates organized folder structure
3. **⬇️ ComfyUI Download** - Clones latest ComfyUI from official repository
4. **🐍 Environment Setup** - Creates virtual environment with dependencies
5. **📦 Model Downloads** - Downloads Flux models and enhancement tools
6. **🔧 Workflow Creation** - Creates optimized workflows for different use cases
7. **🔗 Directory Linking** - Links user directories for easy access
8. **📜 Script Creation** - Generates startup scripts and documentation

---

## 🎮 **Using ComfyUI with Flux**

### **🚀 First Launch**

After installation completes:

```
📁 flux-comfyui/
├── 🟢 start_comfyui.bat (Windows)    # ← Double-click to start!
├── 🟢 start_comfyui.sh (Linux/Mac)   # ← Run to start!
├── 🏃 start_comfyui_highperf.bat     # High performance mode
├── 📄 USER_GUIDE.md                  # Comprehensive guide
└── 📁 ComfyUI/                       # Main ComfyUI installation
    ├── main.py                       # ComfyUI entry point
    ├── models/                       # AI models directory
    └── workflows/                    # Workflow definitions
```

**🎯 To Start ComfyUI:**
1. **Double-click startup script** (Windows) or **run in terminal** (Linux/Mac)
2. **Wait for server to start** (30-60 seconds first time)
3. **Open browser** to http://localhost:8188
   - If that doesn't work, try: http://0.0.0.0:8188
   - Or try: http://127.0.0.1:8188
4. **Start generating images!**

### **🖥️ ComfyUI Interface Overview**

<div align="center">

![ComfyUI Interface](https://via.placeholder.com/1200x800/1a1a2e/ffffff?text=ComfyUI+Professional+Interface)

</div>

ComfyUI provides a **professional node-based interface** for AI image generation:

#### **Node-Based Workflow**
- **📝 Text Nodes**: Prompt input and text encoding
- **🎨 Model Nodes**: Flux model loading and configuration  
- **⚙️ Sampler Nodes**: Generation parameters and quality controls
- **🖼️ Output Nodes**: Image saving and post-processing

#### **Pre-Built Workflows**
- **flux_schnell_basic.json** - Quick 4-step generation
- **flux_portrait.json** - Optimized for portrait photography
- **flux_landscape.json** - Optimized for landscape scenes

#### **Professional Controls**
- **Real-time Preview** - See generation progress live
- **Queue System** - Batch multiple generations
- **History Management** - Browse and reproduce previous results
- **Custom Nodes** - Extend functionality with community additions

### **🎨 Step-by-Step Generation**

#### **1. Load a Workflow**

In ComfyUI web interface:
1. **Click "Load"** button
2. **Navigate to** `user_workflows` folder
3. **Select** `flux_portrait.json` for portraits or `flux_landscape.json` for landscapes
4. **Workflow loads** automatically in the node editor

#### **2. Customize Your Prompt**

Find the **"Positive Prompt"** node:
```
Professional headshot of a [age]-year-old [gender] with [features]. 
[Lighting description], [style description], ultra realistic, high detail.
```

**Example Prompts:**
- `"Professional headshot of a 35-year-old businesswoman with confident expression, natural lighting"`
- `"Majestic mountain landscape with lake reflection, golden hour lighting, cinematic composition"`
- `"Cute cartoon character with friendly expression, colorful background, digital art style"`

#### **3. Configure Generation Settings**

**🎯 Node Settings for Best Results:**

| **Workflow** | **Steps** | **CFG** | **Resolution** | **Speed** |
|:----------:|:--------:|:------:|:-------------:|:--------:|
| **Quick Test** | 4 | 1.0 | 768×768 | ~15 seconds |
| **Balanced** | 4 | 1.0 | 1024×1024 | ~25 seconds |
| **High Quality** | 8 | 2.0 | 1216×1216 | ~45 seconds |

#### **4. Generate Images**

1. **Click "Queue Prompt"** in ComfyUI
2. **Watch real-time progress** in the interface
3. **Images appear** in the output node when complete
4. **Auto-saved** to `user_outputs` folder with metadata

### **📁 User Directory Structure**

The installer creates **easy-access directories** inside the ComfyUI folder:

```
📁 flux-comfyui/
├── 📂 ComfyUI/                  # Main ComfyUI installation
│   ├── main.py                  # ComfyUI server entry point
│   ├── venv/                    # Python virtual environment
│   ├── models/                  # ComfyUI's internal model directories
│   │   ├── checkpoints/         # Where ComfyUI looks for models
│   │   ├── loras/               # LoRA model directory
│   │   └── upscale_models/      # Upscaling models
│   ├── 📂 user_models/          # 🎯 Add your custom Flux models here
│   │   ├── flux1-schnell.safetensors (Pre-installed)
│   │   ├── flux1-dev.safetensors (If downloaded)
│   │   └── README.md
│   ├── 📂 user_loras/           # 🎨 LoRA enhancement files
│   │   └── README.md (Instructions for adding LoRAs)
│   ├── 📂 user_workflows/       # ⚙️ Ready-to-use ComfyUI workflows
│   │   ├── flux_schnell_basic.json
│   │   ├── flux_portrait.json
│   │   └── flux_landscape.json
│   ├── 📂 user_outputs/         # 📸 Generated images (auto-organized)
│   │   └── README.md
│   ├── workflows/               # Internal workflow definitions
│   └── custom_nodes/            # ComfyUI-Manager for extensions
├── 📂 user_models/              # → Convenience link to ComfyUI/user_models/
├── 📂 user_workflows/           # → Convenience link to ComfyUI/user_workflows/
├── start_comfyui.bat/.sh        # Startup scripts
└── USER_GUIDE.md                # Comprehensive user guide
```

#### **🔄 Adding New Models**

1. **Download** .safetensors model files from Hugging Face
2. **Place them** in `ComfyUI/user_models/` directory
3. **Restart ComfyUI** using the startup script
4. **New models appear** in CheckpointLoader nodes automatically

#### **💡 Pro Tips for Best Results**

**🎯 Flux Model Differences:**
- **Flux Schnell**: Fast (4 steps), good quality, lower VRAM
- **Flux Dev**: Higher quality (8+ steps), more VRAM required

**⚙️ Performance Optimization:**

| **Your GPU** | **Recommended Workflow** | **Expected Speed** |
|:----------:|:------------------------:|:-----------------:|
| **6GB VRAM** | flux_schnell_basic.json, 1024×1024 | ~25 seconds |
| **8GB VRAM** | flux_portrait.json, 1024×1024 | ~30 seconds |
| **12GB+ VRAM** | flux_landscape.json, 1216×1216 | ~45 seconds |

---

## 🔧 **Troubleshooting**

### **🚨 Common Issues & Solutions**

#### **❌ "Python not found" or "Git not found"**
```bash
# Install missing prerequisites:

# Windows (using Chocolatey)
choco install python git -y

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip git -y

# CentOS/RHEL
sudo yum install python3 python3-pip git -y

# Fedora
sudo dnf install python3 python3-pip git -y

# macOS (using Homebrew)
brew install python git

# Verify installations
python --version    # Should be 3.8+
git --version       # Any recent version
```

#### **❌ "ComfyUI server failed to start"**
```bash
# Check if Python environment was created correctly
cd flux-comfyui/ComfyUI
ls venv/  # Should contain Python virtual environment

# Try manual startup to see errors
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows

python main.py --port 8188

# Then try accessing:
# http://localhost:8188
# http://0.0.0.0:8188  
# http://127.0.0.1:8188
```

#### **❌ "Model loading failed" or "Out of memory"**
```bash
# Check VRAM usage
nvidia-smi  # For NVIDIA GPUs

# Solutions:
1. Close other GPU applications
2. Use flux-schnell instead of flux-dev (lower VRAM)
3. Reduce image resolution in workflow nodes
4. Add --lowvram flag to startup script
```

#### **❌ "PyTorch CUDA not working"**
```bash
# Check PyTorch installation
cd flux-comfyui/ComfyUI
source venv/bin/activate  # Linux/Mac
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **❌ "Workflow doesn't load"**
```bash
# Check workflow file exists
ls user_workflows/flux_portrait.json

# Verify JSON format
python -m json.tool user_workflows/flux_portrait.json

# Reset workflows if corrupted
python flux_comfyui_installer.py --install-dir flux-comfyui
# (Reinstalls workflows without affecting models)
```

### **📊 Performance Optimization**

#### **🚀 Speed Settings (for testing):**
- Use `flux_schnell_basic.json` workflow
- Resolution: 768×768 in EmptyLatentImage node
- Steps: 4 in KSampler node

#### **💎 Quality Settings (for final images):**
- Use `flux_portrait.json` or `flux_landscape.json`
- Resolution: 1216×1216 or higher
- Steps: 8+ in KSampler node

#### **⚖️ Memory Optimization:**
```bash
# For low VRAM systems, edit startup script:
# Add these flags to the python main.py command:

--lowvram          # For 4-6GB VRAM
--novram           # For <4GB VRAM  
--cpu              # Force CPU generation (slow but works anywhere)
```

### **🔍 Advanced Diagnostics**

#### **Check Installation:**
```bash
cd flux-comfyui
tree -L 3  # Linux/Mac
dir /s     # Windows

# Should show:
# ├── ComfyUI/
# │   ├── user_models/
# │   ├── user_workflows/
# │   └── user_outputs/
# └── start_comfyui.*
```

#### **Test Python Environment:**
```bash
cd flux-comfyui/ComfyUI
source venv/bin/activate
python -c "import torch, transformers, safetensors; print('All packages OK')"
```

---

## 🛠️ **Advanced Configuration**

### **🎨 Custom Workflows**

ComfyUI's node-based system allows infinite customization:

#### **Creating Custom Workflows:**
1. **Start with a basic workflow** from `user_workflows/`
2. **Modify nodes** in the ComfyUI interface
3. **Save workflow** using ComfyUI's save function
4. **Share workflows** by copying JSON files

#### **Popular Workflow Modifications:**
- **Add LoRA nodes** for style enhancement
- **Chain upscaling** for higher resolution
- **Batch generation** for multiple images
- **ControlNet integration** for guided generation

### **🔧 ComfyUI Manager**

Pre-installed ComfyUI Manager allows easy node installation:

1. **Open ComfyUI** in browser
2. **Click "Manager"** button
3. **Browse custom nodes** from the community
4. **Install with one click**

### **📦 Model Management**

#### **Supported Model Types:**
- **Checkpoints**: .safetensors, .ckpt files in `user_models/`
- **LoRAs**: .safetensors files in `user_loras/`
- **VAEs**: .safetensors files in `ComfyUI/models/vae/`
- **Upscalers**: .pth files in `ComfyUI/models/upscale_models/`

#### **Adding Models:**
```bash
# Download from Hugging Face
wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
mv flux1-dev.safetensors flux-comfyui/user_models/

# Restart ComfyUI to detect new models
```

---

## 🌟 **Community & Support**

### **📞 Getting Help**

| **Issue Type** | **Best Channel** | **Response Time** |
|:----------:|:----------:|:----------:|
| **🐛 Installation Issues** | [GitHub Issues](https://github.com/your-repo/flux-comfyui-installer/issues) | 24-48 hours |
| **💬 Usage Questions** | [ComfyUI Discord](https://discord.gg/comfyui) | Real-time |
| **📖 Workflow Help** | [ComfyUI Documentation](https://docs.comfy.org/) | Always available |
| **🚀 Feature Requests** | [GitHub Discussions](https://github.com/your-repo/flux-comfyui-installer/discussions) | Weekly review |

### **🤝 Contributing**

We welcome contributions to improve the installer:

#### **🐛 Reporting Bugs:**
1. **Check existing issues** to avoid duplicates
2. **Include system information** (OS, Python version, GPU)
3. **Provide installation logs** and error messages
4. **Test with minimal reproduction** steps

#### **💡 Suggesting Improvements:**
1. **Search existing discussions** first
2. **Describe the enhancement** clearly
3. **Explain the use case** and benefits
4. **Consider cross-platform compatibility**

#### **🛠️ Code Contributions:**
```bash
# Fork the repository
git clone https://github.com/your-fork/flux-comfyui-installer.git

# Create feature branch
git checkout -b feature/improved-installation

# Test thoroughly on your platform
python flux_comfyui_installer.py --skip-models

# Commit and push
git commit -m "Improve installation reliability"
git push origin feature/improved-installation

# Create Pull Request
```

### **🎓 Learning Resources**

#### **📺 ComfyUI Tutorials:**
- **ComfyUI Basics** - Understanding the node interface
- **Workflow Creation** - Building custom generation pipelines
- **Flux Model Guide** - Getting the best results from Flux
- **Performance Optimization** - Hardware-specific tips

#### **📖 Written Guides:**
- **ComfyUI Official Documentation** - Complete reference
- **Flux Model Papers** - Understanding the technology
- **Workflow Sharing** - Community workflow collections
- **Advanced Techniques** - Pro tips and tricks

---

## 📜 **License & Legal**

### **📄 Licensing:**
- **Installer Code**: MIT License - Free for all use
- **ComfyUI**: GPL License - Open source included
- **Flux Models**: Various licenses (check model pages)
- **Generated Images**: You own rights to your creations

### **⚖️ Usage Rights:**
- ✅ **Personal use** - Unlimited generation
- ✅ **Commercial use** - Check individual model licenses
- ✅ **Modification** - Customize the installer and workflows
- ✅ **Distribution** - Share the installer (follow licenses)
- ✅ **Educational use** - Perfect for learning AI

### **🛡️ Privacy & Security:**
- **Local processing** - Everything runs on your machine
- **No data collection** - No tracking or analytics
- **Offline capable** - Works without internet after setup
- **Your data stays yours** - Models and outputs remain local

---

## 🙏 **Acknowledgments**

### **🌟 Special Thanks:**

- **ComfyUI Team** - Creating the most powerful AI image interface
- **Black Forest Labs** - Developing the incredible Flux models
- **Python Community** - Providing excellent tools and libraries
- **Open Source Contributors** - Making AI accessible to everyone

### **🔬 Technology Stack:**
- **Backend**: ComfyUI + Python + PyTorch
- **Models**: FLUX.1-Schnell, FLUX.1-Dev, various upscalers
- **Dependencies**: transformers, accelerate, safetensors
- **Installation**: Cross-platform Python installer
- **Interface**: ComfyUI's professional web interface

---

<div align="center">

## 🚀 **Ready to Generate Amazing Images?**

### [⬇️ Download Installer](https://github.com/your-repo/flux-comfyui-installer/releases) | [📖 View Documentation](https://github.com/your-repo/flux-comfyui-installer/wiki) | [💬 Join Community](https://discord.gg/comfyui)

---

**Professional AI image generation with Flux models - Automated setup in minutes!**

*Transform your creative workflow with ComfyUI and Flux!*

```bash
# Get started in one command:
python flux_comfyui_installer.py
```

[![GitHub stars](https://img.shields.io/github/stars/your-repo/flux-comfyui-installer.svg?style=social&label=Star)](https://github.com/your-repo/flux-comfyui-installer)
[![GitHub forks](https://img.shields.io/github/forks/your-repo/flux-comfyui-installer.svg?style=social&label=Fork)](https://github.com/your-repo/flux-comfyui-installer/fork)

</div>