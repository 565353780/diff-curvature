cd ..
git clone https://github.com/NVlabs/nvdiffrast

pip install -U torch torchvision torchaudio

pip install -U trimesh matplotlib opencv-python einops

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

cd nvdiffrast
pip install .
