cd ..
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/facebookresearch/pytorch3d.git

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install -U trimesh matplotlib opencv-python einops

# pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install pyglet==1.5.13

cd nvdiffrast
pip install .

cd ../pytorch3d
export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"
pip install .
