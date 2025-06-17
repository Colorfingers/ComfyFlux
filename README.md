#!/usr/bin/env python3
"""
Flux ComfyUI Installer
Automates the setup of ComfyUI with Flux models and optimized workflows
"""

import os
import sys
import subprocess
import json
import shutil
import urllib.request
import platform
from pathlib import Path
import tempfile
import argparse

class FluxComfyUIInstaller:
    def __init__(self, install_dir="flux-comfyui"):
        self.install_dir = Path(install_dir).resolve()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="flux_install_"))
        self.comfyui_dir = self.install_dir
        
        # Track failed downloads and installs
        self.failed_downloads = []
        self.failed_installs = []
        
        # Detect WSL environment
        self.is_wsl = self.detect_wsl()
        if self.is_wsl:
            self.log("WSL environment detected")
            
        # Repository and model URLs
        self.comfyui_repo = "https://github.com/comfyanonymous/ComfyUI.git"
        
        # Working model URLs - Updated December 2024
        # Using ComfyUI-Org optimized versions that work out of the box
        self.flux_model_urls = {
            "flux-schnell-fp8": "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors",
            "flux-dev-fp8": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
        }
        
        # Additional models with working URLs
        self.additional_models = {
            "upscaler": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN_4x.pth",
            "vae": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
            "clip_l": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
            "t5xxl": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
        }
        
    def detect_wsl(self):
        """Detect if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                return 'microsoft' in content or 'wsl' in content
        except:
            return False
        
    def log(self, message):
        print(f"[FLUX INSTALLER] {message}")
        
    def run_command(self, cmd, cwd=None, shell=True):
        """Run a command and handle errors"""
        try:
            self.log(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=shell, cwd=cwd, check=True, 
                                  capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.log(f"Error running command: {e}")
            self.log(f"Stdout: {e.stdout}")
            self.log(f"Stderr: {e.stderr}")
            raise
            
    def check_prerequisites(self):
        """Check if required tools are installed"""
        self.log("Checking prerequisites...")
        
        errors = []
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version < (3, 8):
                errors.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            else:
                self.log(f"OK Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        except Exception as e:
            errors.append(f"Python check failed: {e}")
            
        # Check Git
        try:
            git_output = self.run_command("git --version").strip()
            self.log(f"OK {git_output}")
        except Exception as e:
            errors.append("Git not found. Please install Git from https://git-scm.com")
            
        # Check disk space
        try:
            free_bytes = shutil.disk_usage(self.install_dir.parent if self.install_dir.parent.exists() else Path.cwd()).free
            free_gb = free_bytes / (1024**3)
            if free_gb < 15:
                errors.append(f"Insufficient disk space: {free_gb:.1f}GB available, 15GB+ required")
            else:
                self.log(f"OK Disk space: {free_gb:.1f}GB available")
        except Exception as e:
            self.log(f"Warning: Could not check disk space: {e}")
            
        if errors:
            self.log("FAILED Prerequisites check failed:")
            for error in errors:
                self.log(f"  - {error}")
            raise Exception("Please fix the above issues before running the installer")
            
        self.log("SUCCESS All prerequisites satisfied!")
        
    def setup_directory_structure(self):
        """Create the ComfyUI directory structure"""
        self.log("Setting up directory structure...")
        
        # Remove existing directory if it exists
        if self.install_dir.exists():
            self.log(f"Removing existing directory: {self.install_dir}")
            shutil.rmtree(self.install_dir)
            
        self.install_dir.mkdir(parents=True)
        
    def setup_user_directories(self, comfyui_dir):
        """Create user-accessible directories inside ComfyUI folder"""
        self.log("Setting up user directories inside ComfyUI...")
        
        # Create user directories inside ComfyUI directory
        user_dirs = {
            "user_models": "User models directory - place your custom models here",
            "user_workflows": "User workflows directory - save your custom workflows here",
            "user_loras": "User LoRAs directory - place additional LoRA files here",
            "user_outputs": "Output directory - generated images will be saved here"
        }
        
        for dir_name, description in user_dirs.items():
            # Create directories inside ComfyUI folder
            dir_path = comfyui_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            
            # Also create convenience symlinks at root level for easy access
            root_link = self.install_dir / dir_name
            try:
                if root_link.exists() or root_link.is_symlink():
                    if root_link.is_dir() and not root_link.is_symlink():
                        shutil.rmtree(root_link)
                    elif root_link.is_symlink():
                        root_link.unlink()
                
                if platform.system() == "Windows":
                    # Create junction on Windows
                    self.run_command(f'mklink /J "{root_link}" "{dir_path}"')
                else:
                    # Create symlink on Unix
                    root_link.symlink_to(dir_path.resolve())
                
                self.log(f"SUCCESS Created link: {dir_name} -> ComfyUI/{dir_name}")
            except Exception as e:
                self.log(f"WARNING: Could not create convenience link for {dir_name}: {e}")
            
            # Create README files
            readme_content = f"""# {dir_name.replace('_', ' ').title()}

{description}

## Location:
This directory is located inside the ComfyUI folder at: ComfyUI/{dir_name}/

## Instructions:
- Place files in this directory to make them available to ComfyUI
- Restart ComfyUI after adding new files for them to be detected
- Check the console output for any loading errors

## Supported Formats:
"""
            
            if "models" in dir_name:
                readme_content += "- .safetensors, .ckpt, .pt, .bin files"
            elif "workflows" in dir_name:
                readme_content += "- .json workflow files (ComfyUI format)"
            elif "loras" in dir_name:
                readme_content += "- .safetensors LoRA files"
            elif "outputs" in dir_name:
                readme_content += "- Generated images in PNG format with metadata"
                
            with open(dir_path / "README.md", "w") as f:
                f.write(readme_content)
                
    def clone_comfyui(self):
        """Clone ComfyUI repository"""
        self.log("Cloning ComfyUI...")
        comfyui_source_dir = self.install_dir / "ComfyUI"
        self.run_command(f"git clone {self.comfyui_repo} {comfyui_source_dir}")
        return comfyui_source_dir
        
    def setup_python_environment(self, comfyui_dir):
        """Setup Python virtual environment and install dependencies"""
        self.log("Setting up Python environment...")
        
        # Create virtual environment
        venv_dir = comfyui_dir / "venv"
        try:
            self.run_command(f"python -m venv {venv_dir}")
        except Exception as e:
            self.log(f"Error creating virtual environment: {e}")
            raise Exception("Failed to create Python virtual environment")
        
        # Get pip and python commands
        if platform.system() == "Windows":
            pip_cmd = f'"{venv_dir}\\Scripts\\pip.exe"'
            python_cmd = f'"{venv_dir}\\Scripts\\python.exe"'
        else:
            pip_cmd = f'"{venv_dir}/bin/pip"'
            python_cmd = f'"{venv_dir}/bin/python"'
            
        try:
            # Upgrade pip
            self.log("Upgrading pip...")
            self.run_command(f"{pip_cmd} install --upgrade pip")
            
            # Install PyTorch with CUDA support
            self.log("Installing PyTorch with CUDA support...")
            pytorch_cmd = f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            
            try:
                self.run_command(pytorch_cmd)
                self.log("SUCCESS PyTorch with CUDA installed successfully")
            except Exception as e:
                self.log(f"Warning: CUDA PyTorch failed: {e}")
                self.log("Installing CPU-only PyTorch...")
                self.run_command(f"{pip_cmd} install torch torchvision torchaudio")
                self.log("WARNING CPU-only PyTorch installed (performance will be slower)")
            
        except Exception as e:
            self.log(f"Error installing PyTorch: {e}")
            raise Exception("Failed to install PyTorch")
        
        # Install ComfyUI requirements
        requirements_file = comfyui_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                self.log("Installing ComfyUI requirements...")
                self.run_command(f"{pip_cmd} install -r {requirements_file}")
            except Exception as e:
                self.log(f"Warning: Some requirements failed: {e}")
        
        # Install essential packages
        essential_packages = [
            "pillow", "numpy", "opencv-python", "transformers", 
            "accelerate", "safetensors", "tqdm", "sentencepiece", 
            "protobuf", "compel"
        ]
        
        for package in essential_packages:
            try:
                self.log(f"Installing {package}...")
                self.run_command(f"{pip_cmd} install {package}")
            except Exception as e:
                self.log(f"Warning: Could not install {package}: {e}")
                
        # Install ComfyUI Manager
        self.install_comfyui_manager(comfyui_dir, pip_cmd)
        
        return python_cmd, pip_cmd
        
    def install_comfyui_manager(self, comfyui_dir, pip_cmd):
        """Install ComfyUI Manager for easier node management"""
        self.log("Installing ComfyUI Manager...")
        
        try:
            custom_nodes_dir = comfyui_dir / "custom_nodes"
            custom_nodes_dir.mkdir(exist_ok=True)
            
            manager_dir = custom_nodes_dir / "ComfyUI-Manager"
            if not manager_dir.exists():
                manager_repo = "https://github.com/ltdrdata/ComfyUI-Manager.git"
                self.run_command(f"git clone {manager_repo} {manager_dir}")
                
                # Install manager requirements
                manager_requirements = manager_dir / "requirements.txt"
                if manager_requirements.exists():
                    try:
                        self.run_command(f"{pip_cmd} install -r {manager_requirements}")
                        self.log("SUCCESS ComfyUI Manager installed successfully")
                    except Exception as e:
                        self.log(f"Warning: Manager requirements failed: {e}")
                else:
                    self.log("SUCCESS ComfyUI Manager installed")
            else:
                self.log("SUCCESS ComfyUI Manager already exists")
                
        except Exception as e:
            self.log(f"Warning: Could not install ComfyUI Manager: {e}")
            
    def download_flux_models(self, comfyui_dir):
        """Download Flux models and additional assets"""
        self.log("Downloading Flux models...")
        
        # Create model directories
        models_dir = comfyui_dir / "models"
        checkpoints_dir = models_dir / "checkpoints"
        clip_dir = models_dir / "clip"
        upscale_dir = models_dir / "upscale_models"
        vae_dir = models_dir / "vae"
        
        for dir_path in [checkpoints_dir, clip_dir, upscale_dir, vae_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"Created directory: {dir_path}")
        
        # User models directory inside ComfyUI
        user_models_dir = comfyui_dir / "user_models"
        user_models_dir.mkdir(exist_ok=True)
        self.log(f"Created user directory: {user_models_dir}")
        
        # Track download results
        downloads_attempted = 0
        downloads_successful = 0
        failed_downloads = []
        
        # Define all downloads to attempt
        download_tasks = [
            {
                "name": "Flux Schnell FP8",
                "url": self.flux_model_urls["flux-schnell-fp8"],
                "path": checkpoints_dir / "flux1-schnell-fp8.safetensors",
                "user_path": user_models_dir / "flux1-schnell-fp8.safetensors",
                "required": True,
                "description": "Main Flux model for fast generation"
            },
            {
                "name": "Flux Dev FP8",
                "url": self.flux_model_urls["flux-dev-fp8"],
                "path": checkpoints_dir / "flux1-dev-fp8.safetensors",
                "user_path": user_models_dir / "flux1-dev-fp8.safetensors",
                "required": False,
                "description": "Higher quality Flux model (optional)"
            },
            {
                "name": "CLIP-L Text Encoder",
                "url": self.additional_models["clip_l"],
                "path": clip_dir / "clip_l.safetensors",
                "user_path": None,
                "required": False,
                "description": "Text encoder for better prompt understanding"
            },
            {
                "name": "T5-XXL Text Encoder",
                "url": self.additional_models["t5xxl"],
                "path": clip_dir / "t5xxl_fp8_e4m3fn.safetensors",
                "user_path": None,
                "required": False,
                "description": "Advanced text encoder for complex prompts"
            },
            {
                "name": "Flux VAE",
                "url": self.additional_models["vae"],
                "path": vae_dir / "ae.safetensors",
                "user_path": None,
                "required": False,
                "description": "Variational autoencoder for image processing"
            },
            {
                "name": "ESRGAN Upscaler",
                "url": self.additional_models["upscaler"],
                "path": upscale_dir / "ESRGAN_4x.pth",
                "user_path": None,
                "required": False,
                "description": "4x image upscaling model"
            }
        ]
        
        # Attempt all downloads
        for task in download_tasks:
            if task["path"].exists():
                self.log(f"{task['name']} already exists, skipping download")
                continue
                
            self.log(f"Downloading {task['name']}...")
            downloads_attempted += 1
            
            if self.download_with_progress(task["url"], task["path"]):
                downloads_successful += 1
                # Copy to user directory if specified
                if task["user_path"]:
                    try:
                        shutil.copy2(task["path"], task["user_path"])
                        self.log(f"Copied {task['name']} to user directory")
                    except Exception as e:
                        self.log(f"WARNING: Could not copy {task['name']} to user directory: {e}")
            else:
                failed_downloads.append({
                    "name": task["name"],
                    "url": task["url"],
                    "required": task["required"],
                    "description": task["description"]
                })
                
        # Store failed downloads for later reporting
        self.failed_downloads = failed_downloads
        
        # Report download summary
        self.log(f"Download Summary: {downloads_successful}/{downloads_attempted} successful")
        
        if failed_downloads:
            self.log(f"WARNING: {len(failed_downloads)} downloads failed (will be reported at end of installation)")
        
        # Only fail installation if ALL critical models failed
        critical_models_exist = any([
            (checkpoints_dir / "flux1-schnell-fp8.safetensors").exists(),
            (checkpoints_dir / "flux1-dev-fp8.safetensors").exists()
        ])
        
        if not critical_models_exist and downloads_attempted > 0:
            self.log("CRITICAL: No Flux models were downloaded successfully!")
            self.log("Installation will continue, but you'll need to manually download models")
            # Don't raise exception - let installation continue
        
        self.log("Model download phase completed")
        
    def report_failed_downloads(self):
        """Report all failed downloads at the end of installation"""
        if not hasattr(self, 'failed_downloads') or not self.failed_downloads:
            return
            
        self.log("")
        self.log("=" * 60)
        self.log("DOWNLOAD FAILURES SUMMARY")
        self.log("=" * 60)
        self.log(f"The following {len(self.failed_downloads)} items failed to download:")
        self.log("")
        
        for item in self.failed_downloads:
            status = "REQUIRED" if item["required"] else "OPTIONAL"
            self.log(f"Failed: {item['name']} ({status})")
            self.log(f"   Description: {item['description']}")
            self.log(f"   URL: {item['url']}")
            self.log("")
            
        self.log("MANUAL DOWNLOAD INSTRUCTIONS:")
        self.log("You can manually download these files and place them in the correct directories:")
        self.log("")
        
        for item in self.failed_downloads:
            if "flux1-schnell-fp8" in item["name"]:
                self.log(f"• {item['name']}: Place in ComfyUI/models/checkpoints/")
                self.log(f"  Also copy to: ComfyUI/user_models/")
            elif "flux1-dev-fp8" in item["name"]:
                self.log(f"• {item['name']}: Place in ComfyUI/models/checkpoints/")
                self.log(f"  Also copy to: ComfyUI/user_models/")
            elif "clip" in item["name"].lower():
                self.log(f"• {item['name']}: Place in ComfyUI/models/clip/")
            elif "vae" in item["name"].lower():
                self.log(f"• {item['name']}: Place in ComfyUI/models/vae/")
            elif "upscaler" in item["name"].lower():
                self.log(f"• {item['name']}: Place in ComfyUI/models/upscale_models/")
                
        self.log("")
        self.log("ALTERNATIVE SOURCES:")
        self.log("• ComfyUI Examples: https://comfyanonymous.github.io/ComfyUI_examples/flux/")
        self.log("• Hugging Face Comfy-Org: https://huggingface.co/Comfy-Org")
        self.log("• Direct links in the URLs above")
        self.log("")
        self.log("After manually downloading, restart ComfyUI to detect the new models.")
        self.log("=" * 60)
            
    def download_with_progress(self, url, path):
        """Download a file with progress indication and error handling"""
        if not url:
            self.log(f"WARNING: Skipping download for {path.name} - URL not available")
            return False
            
        try:
            self.log(f"Downloading {path.name} from {url}")
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test URL accessibility first
            try:
                import urllib.request
                request = urllib.request.Request(url, method='HEAD')
                response = urllib.request.urlopen(request, timeout=30)
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    self.log(f"File size: {size_mb:.1f} MB")
                else:
                    self.log("File size: Unknown")
            except Exception as e:
                self.log(f"WARNING: Could not get file info: {e}")
                # Continue with download anyway
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 50 == 0 or percent >= 100:  # Show progress every 50 blocks or at completion
                        downloaded_mb = (block_num * block_size) / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        self.log(f"  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            
            urllib.request.urlretrieve(url, path, reporthook=show_progress)
            
            # Verify download completed
            if path.exists() and path.stat().st_size > 0:
                file_size_mb = path.stat().st_size / (1024 * 1024)
                self.log(f"SUCCESS Downloaded {path.name} ({file_size_mb:.1f} MB)")
                return True
            else:
                self.log(f"FAILED Download incomplete: {path.name}")
                if path.exists():
                    path.unlink()
                return False
            
        except Exception as e:
            self.log(f"FAILED to download {path.name}: {e}")
            self.log(f"URL was: {url}")
            # Remove partial file if it exists
            if path.exists():
                try:
                    path.unlink()
                except:
                    pass
            return False
    
    def create_basic_workflow(self):
        """Create basic Flux Schnell FP8 workflow for fast generation"""
        return {
            "1": {
                "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Flux Schnell FP8"}
            },
            "2": {
                "inputs": {
                    "text": "A beautiful portrait with natural lighting, photorealistic, high detail",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted, cartoon, anime",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode", 
                "_meta": {"title": "Negative Prompt"}
            },
            "4": {
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Image Dimensions"}
            },
            "5": {
                "inputs": {
                    "seed": 42,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple", 
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "Generate Image"}
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "Decode Latent"}
            },
            "7": {
                "inputs": {"images": ["6", 0], "filename_prefix": "flux_basic"},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"}
            }
        }
            
    def create_flux_workflows(self, comfyui_dir):
        """Create optimized Flux workflows"""
        self.log("Creating Flux workflows...")
        
        # Create workflows in both ComfyUI and user directories (inside ComfyUI)
        workflows_dir = comfyui_dir / "workflows"
        user_workflows_dir = comfyui_dir / "user_workflows"
        
        workflows_dir.mkdir(exist_ok=True)
        user_workflows_dir.mkdir(exist_ok=True)
        
        # Basic Flux Schnell workflow
        schnell_workflow = {
            "1": {
                "inputs": {"ckpt_name": "flux1-schnell.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Flux Schnell"}
            },
            "2": {
                "inputs": {
                    "text": "A beautiful portrait of a person with photorealistic skin, natural lighting, professional photography, high detail",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted, cartoon, anime, painting, illustration",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "4": {
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Image Dimensions"}
            },
            "5": {
                "inputs": {
                    "seed": 42, "steps": 4, "cfg": 1.0,
                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "Generate"}
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "Decode"}
            },
            "7": {
                "inputs": {"images": ["6", 0], "filename_prefix": "flux_schnell"},
                "class_type": "SaveImage",
                "_meta": {"title": "Save"}
            }
        }
        
        # Save workflow files
        workflow_files = {
            "flux_schnell_basic.json": schnell_workflow,
            "flux_portrait.json": self.create_portrait_workflow(),
            "flux_landscape.json": self.create_landscape_workflow()
        }
        
        for filename, workflow in workflow_files.items():
            # Save to ComfyUI workflows directory
            workflow_path = workflows_dir / filename
            with open(workflow_path, "w") as f:
                json.dump(workflow, f, indent=2)
                
            # Also save to user workflows directory
            user_workflow_path = user_workflows_dir / filename
            with open(user_workflow_path, "w") as f:
                json.dump(workflow, f, indent=2)
                
        self.log(f"SUCCESS Created {len(workflow_files)} workflow files")
        
    def create_portrait_workflow(self):
        """Create a portrait-optimized workflow"""
        workflow = self.create_basic_workflow()
        
        # Modify for portrait optimization
        workflow["2"]["inputs"]["text"] = "Professional headshot portrait, beautiful eyes, natural skin texture, soft lighting, shallow depth of field, photorealistic, high detail, 8k"
        workflow["4"]["inputs"]["width"] = 832
        workflow["4"]["inputs"]["height"] = 1216
        workflow["5"]["inputs"]["steps"] = 4
        workflow["7"]["inputs"]["filename_prefix"] = "flux_portrait"
        
        # Add metadata
        workflow["_meta"] = {
            "title": "Flux Portrait Workflow",
            "description": "Optimized for portrait photography with 2:3 aspect ratio",
            "recommended_settings": {
                "steps": 4,
                "cfg": 1.0,
                "dimensions": "832x1216"
            }
        }
        
        return workflow
        
    def create_landscape_workflow(self):
        """Create a landscape-optimized workflow"""
        return {
            "1": {
                "inputs": {"ckpt_name": "flux1-schnell.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Flux"}
            },
            "2": {
                "inputs": {
                    "text": "Beautiful landscape photography, dramatic lighting, golden hour, detailed foreground and background, professional composition, 8k resolution",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Landscape Prompt"}
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted, overexposed, underexposed, noise",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative"}
            },
            "4": {
                "inputs": {"width": 1344, "height": 768, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Landscape Dimensions"}
            },
            "5": {
                "inputs": {
                    "seed": 456, "steps": 4, "cfg": 1.0,
                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "Generate"}
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "Decode"}
            },
            "7": {
                "inputs": {"images": ["6", 0], "filename_prefix": "flux_landscape"},
                "class_type": "SaveImage",
                "_meta": {"title": "Save"}
            }
        }
        
    def setup_model_linking(self, comfyui_dir):
        """Link user model directories to ComfyUI"""
        self.log("Setting up model directory linking...")
        
        # User directories are now inside ComfyUI
        user_models_dir = comfyui_dir / "user_models"
        user_loras_dir = comfyui_dir / "user_loras"
        comfyui_models_dir = comfyui_dir / "models"
        
        # Ensure ComfyUI model directories exist
        (comfyui_models_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (comfyui_models_dir / "loras").mkdir(parents=True, exist_ok=True)
        
        # Create links/copies from user directories to ComfyUI model directories
        try:
            if platform.system() == "Windows":
                # On Windows, try junction first, fall back to copying
                try:
                    # Remove existing junctions/directories if they exist
                    user_checkpoints_link = comfyui_models_dir / "user_checkpoints"
                    user_loras_link = comfyui_models_dir / "user_loras"
                    
                    for link in [user_checkpoints_link, user_loras_link]:
                        if link.exists():
                            if link.is_dir() and not link.is_symlink():
                                shutil.rmtree(link)
                            elif link.is_symlink():
                                link.unlink()
                    
                    self.run_command(f'mklink /J "{user_checkpoints_link}" "{user_models_dir}"')
                    self.run_command(f'mklink /J "{user_loras_link}" "{user_loras_dir}"')
                    self.log("SUCCESS Created junctions for user model directories")
                except:
                    self.log("Junction failed, models will be accessible directly from user directories")
            else:
                # On Unix systems, use symbolic links
                user_checkpoints_link = comfyui_models_dir / "user_checkpoints"
                user_loras_link = comfyui_models_dir / "user_loras"
                
                for link in [user_checkpoints_link, user_loras_link]:
                    if link.exists() or link.is_symlink():
                        if link.is_dir() and not link.is_symlink():
                            shutil.rmtree(link)
                        elif link.is_symlink():
                            link.unlink()
                
                user_checkpoints_link.symlink_to(user_models_dir.resolve())
                user_loras_link.symlink_to(user_loras_dir.resolve())
                self.log("SUCCESS Created symlinks for user model directories")
        except Exception as e:
            self.log(f"WARNING: Could not create model directory links: {e}")
            self.log("Models can still be placed directly in ComfyUI/models/ directories")
            
    def create_startup_scripts(self, comfyui_dir, python_cmd):
        """Create convenient startup scripts"""
        self.log("Creating startup scripts...")
        
        # Get the correct python command path
        if platform.system() == "Windows":
            venv_python = comfyui_dir / "venv" / "Scripts" / "python.exe"
            
            # Windows batch script
            batch_content = f'''@echo off
echo Starting ComfyUI with Flux models...
echo.
cd /d "{comfyui_dir}"
"{venv_python}" main.py --port 8188 --listen 0.0.0.0 --output-directory "./user_outputs"
pause
'''
            batch_file = self.install_dir / "start_comfyui.bat"
            with open(batch_file, "w") as f:
                f.write(batch_content)
                
            # High performance script
            performance_batch = f'''@echo off
echo Starting ComfyUI with high performance settings...
echo This uses more VRAM but generates faster
echo.
cd /d "{comfyui_dir}"
"{venv_python}" main.py --port 8188 --listen 0.0.0.0 --output-directory "./user_outputs" --highvram --dont-upcast-attention
pause
'''
            perf_file = self.install_dir / "start_comfyui_highperf.bat"
            with open(perf_file, "w") as f:
                f.write(performance_batch)
                
        else:
            venv_python = comfyui_dir / "venv" / "bin" / "python"
            
            # Unix shell script
            shell_content = f'''#!/bin/bash
echo "Starting ComfyUI with Flux models..."
cd "{comfyui_dir}"
"{venv_python}" main.py --port 8188 --listen 0.0.0.0 --output-directory "./user_outputs"
'''
            shell_file = self.install_dir / "start_comfyui.sh"
            with open(shell_file, "w") as f:
                f.write(shell_content)
            os.chmod(shell_file, 0o755)
            
        self.log("SUCCESS Created startup scripts")
        
    def create_user_guide(self):
        """Create a comprehensive user guide"""
        self.log("Creating user guide...")
        
        guide_content = f"""# Flux ComfyUI Installation Guide

## Post-Installation: Check for Download Failures

**IMPORTANT:** After installation completes, scroll up in the terminal output to look for a "DOWNLOAD FAILURES SUMMARY" section. If you see this section, some required models failed to download and need manual intervention.

### If You See Download Failures:
1. **Check the terminal output** for the "DOWNLOAD FAILURES SUMMARY" section
2. **Note which files failed** and their target directories
3. **Follow the manual download instructions** provided in the summary
4. **Use the alternative sources** listed if direct links don't work

### Manual Download Steps:
1. Download the failed files from the URLs shown in the failure summary
2. Place them in the correct directories as specified:
   - **Flux models**: `ComfyUI/models/checkpoints/` AND `ComfyUI/user_models/`
   - **Text encoders**: `ComfyUI/models/clip/`
   - **VAE files**: `ComfyUI/models/vae/`
   - **Upscalers**: `ComfyUI/models/upscale_models/`
3. **Restart ComfyUI** after adding files manually

### Alternative Download Sources:
- **Comfy-Org on Hugging Face**: https://huggingface.co/Comfy-Org
- **ComfyUI Examples**: https://comfyanonymous.github.io/ComfyUI_examples/flux/
- **Original Flux Repository**: https://huggingface.co/black-forest-labs

## Quick Start

### Windows:
1. Double-click `start_comfyui.bat` to start the server
2. Open your browser to http://localhost:8188
3. Load a workflow from the `user_workflows` folder

### Linux/Mac:
1. Run `./start_comfyui.sh` to start the server  
2. Open your browser to http://localhost:8188
3. Load a workflow from the `user_workflows` folder

## Directory Structure

### ComfyUI/
Main ComfyUI installation with all dependencies

### user_models/
Place your custom Flux models here (.safetensors files)
- flux1-schnell.safetensors (included)
- flux1-dev.safetensors (included if downloaded)

### user_loras/
Place additional LoRA files here for fine-tuning

### user_workflows/
Custom workflows are saved here:
- flux_schnell_basic.json - Basic Flux generation
- flux_portrait.json - Optimized for portraits
- flux_landscape.json - Optimized for landscapes

### user_outputs/
Generated images are automatically saved here

## Using Flux Models

### Flux Schnell:
- Fast generation (4 steps)
- Good quality
- Low VRAM usage
- CFG Scale: 1.0

### Flux Dev:
- Higher quality
- More steps (8-20)
- Higher VRAM usage
- CFG Scale: 1.0-3.5

## Performance Tips

### High Performance Mode:
Use `start_comfyui_highperf.bat` for faster generation if you have enough VRAM

### Low VRAM Mode:
If you get out of memory errors, edit the startup script and add:
- --lowvram (for 4-6GB VRAM)
- --novram (for <4GB VRAM)

## Workflow Loading

1. Start ComfyUI using the startup script
2. In the web interface, click "Load" 
3. Navigate to user_workflows folder
4. Select a .json workflow file
5. Modify the prompt and generate!

## Adding New Models

1. Download .safetensors files to user_models/
2. Restart ComfyUI
3. New models will appear in the CheckpointLoader dropdown

## Troubleshooting

### Download Failures During Installation:
If you see a "DOWNLOAD FAILURES SUMMARY" in your terminal output:
1. **Don't panic** - the installation can still work with manual downloads
2. **Save the terminal output** or take a screenshot of the failure summary
3. **Follow the manual download instructions** provided in the summary
4. **Alternative approach**: Re-run the installer to attempt downloads again

### Manual Model Download Process:
1. **Find the failed downloads** in the terminal output summary
2. **Download each file** using a web browser or download manager:
   ```
   # Example for Flux Schnell FP8:
   URL: https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors
   Save to: ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors
   Copy to: ComfyUI/user_models/flux1-schnell-fp8.safetensors
   ```
3. **Verify file integrity** - downloaded files should be several GB in size
4. **Restart ComfyUI** to detect the new models

### Network Issues:
If downloads consistently fail:
- **Check your internet connection** stability
- **Try using a VPN** if certain sites are blocked
- **Use a download manager** for large files (recommended for files >1GB)
- **Download during off-peak hours** for better speeds

### ComfyUI won't start:
- Check that Python virtual environment was created successfully
- Ensure all requirements were installed
- Try running the startup script from command line to see errors

### Out of memory errors:
- Use --lowvram or --novram flags in startup script
- Reduce image resolution in workflows
- Close other GPU-intensive applications

### Models not loading:
- Ensure model files are in the correct format (.safetensors)
- Check file permissions
- Restart ComfyUI after adding new models
- **Verify models are in the right directories** (check both ComfyUI/models/ and user_models/)

## Advanced Usage

### Custom Nodes:
Install additional nodes using ComfyUI Manager (included)

### API Usage:
ComfyUI runs on http://localhost:8188/api for programmatic access

### Batch Processing:
Modify workflows to increase batch_size for multiple images

## Support

For ComfyUI issues: https://github.com/comfyanonymous/ComfyUI
For Flux model issues: https://github.com/black-forest-labs/flux

Installation completed: {self.install_dir}
"""
        
        guide_file = self.install_dir / "USER_GUIDE.md"
        with open(guide_file, "w") as f:
            f.write(guide_content)
            
    def cleanup(self):
        """Clean up temporary files"""
        self.log("Cleaning up...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def install(self):
        """Main installation process"""
        try:
            self.log("Starting Flux ComfyUI installation...")
            self.log(f"Installation directory: {self.install_dir}")
            
            # Step 1: Check prerequisites
            self.check_prerequisites()
            
            # Step 2: Setup directory structure
            self.setup_directory_structure()
            
            # Step 3: Clone ComfyUI
            comfyui_dir = self.clone_comfyui()
            
            # Step 4: Setup user directories (now inside ComfyUI)
            self.setup_user_directories(comfyui_dir)
            
            # Step 5: Setup Python environment
            python_cmd, pip_cmd = self.setup_python_environment(comfyui_dir)
            
            # Step 6: Download models (unless skipped)
            if not getattr(self, 'skip_models', False):
                self.download_flux_models(comfyui_dir)
            else:
                self.log("Skipping model downloads (--skip-models flag)")
                # Initialize empty failed downloads for consistency
                self.failed_downloads = []
            
            # Step 7: Create workflows
            self.create_flux_workflows(comfyui_dir)
            
            # Step 8: Setup model linking
            self.setup_model_linking(comfyui_dir)
            
            # Step 9: Create startup scripts
            self.create_startup_scripts(comfyui_dir, python_cmd)
            
            # Step 10: Create user guide
            self.create_user_guide()
            
            # Step 11: Report any failed downloads
            self.report_failed_downloads()
            
            self.log("SUCCESS Installation completed successfully!")
            self.log(f"Installation directory: {self.install_dir}")
            
            if platform.system() == "Windows":
                self.log("To start ComfyUI: Double-click start_comfyui.bat")
            else:
                self.log("To start ComfyUI: ./start_comfyui.sh")
                
            self.log("Then open http://localhost:8188 in your browser")
            self.log("User directories are located inside ComfyUI folder:")
            self.log(f"  Models: {self.install_dir}/ComfyUI/user_models/")
            self.log(f"  Workflows: {self.install_dir}/ComfyUI/user_workflows/") 
            self.log(f"  Outputs: {self.install_dir}/ComfyUI/user_outputs/")
            
        except Exception as e:
            self.log(f"FAILED Installation failed: {str(e)}")
            self.log("Common solutions:")
            self.log("  - Ensure Python 3.8+ is installed")
            self.log("  - Check internet connection for downloads")
            self.log("  - Verify sufficient disk space (15GB+)")
            self.log("  - Install Git from https://git-scm.com")
            raise
        finally:
            self.cleanup()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Install ComfyUI with Flux models")
    parser.add_argument("--install-dir", default="flux-comfyui", help="Installation directory (default: flux-comfyui)")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads (faster for testing)")
    
    args = parser.parse_args()
    
    try:
        installer = FluxComfyUIInstaller(args.install_dir)
        installer.skip_models = args.skip_models
        installer.install()
        
    except KeyboardInterrupt:
        print("\nFAILED Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILED Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
