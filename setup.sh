cd ..
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/facebookresearch/pytorch3d.git

pip install -U torch torchvision torchaudio

pip install -U trimesh matplotlib opencv-python einops

# pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

pip install pyglet==1.5.13

cd nvdiffrast
pip install .

cd ../pytorch3d
pip install .
