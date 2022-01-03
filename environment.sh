conda create -y -n keti python=3.8
conda activate keti
#source activate keti
conda install scipy

####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install pyaudio
conda install -c conda-forge librosa


pip install PyYAML
pip install tensorboard