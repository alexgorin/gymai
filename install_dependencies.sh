sudo apt-get install -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g \
    zlib1g-dev swig g++ python3-venv python-dev python-dev-is-python3

for pip_package in wheel cffi cython lockfile
do
	pip install $pip_package
done

for pip_package in pandas tensorflow-gpu keras setuptools jupyterlab
do
	pip install $pip_package
done