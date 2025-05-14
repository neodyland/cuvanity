# Tor V3 vanity address generator

Usage:
```
./vanity_torv3_cuda [-i] [-d N] pattern1 [pattern_2] [pattern_3] ... [pattern_n]
```

`-i` will display the keygen rate every 20 seconds in million addresses per second.  
`-d` use CUDA device with index N (counting from 0). This argument can be repeated multiple times with different N.

Example:
```
./vanity_torv3_cuda -i 4?xxxxx 533333 655555 777777 p99999
```
Capture generated keys simply by output redirection into a file :  
`$ ./vanity_torv3_cuda test | tee -a keys.txt`

You can then use the `genpubs.py` scripts under `util/` to generate all the tor secret files :  
 `$ python3 genpubs.py keys.txt`
 
A folder called `generated-<timestamp>` will be generated  

## Performance

This generator can check ~2 million keys/second on a single GeForce GTX 1080Ti.  
Multiple patterns don't slow down the search.  
Max pattern prefix search length is 32 characters to allow offset searching ie. `????wink??????test???`  
Anything beyond 12 characters will probably take a few hundred years...  
Only these characters are permitted in an address:  
```
abcdefghijklmnopqrstuvwxyz234567
```

## Build instructions

### Ubuntu 22.04

Run the following commands to build and install (assumes card is at least Pascal generation)
```
sudo apt update
sudo apt install cmake build-essential wget python3-pip ninja-build

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-12-4 cuda-drivers-550

echo "export PATH=\$PATH:/usr/local/cuda-12.4/bin" >> $HOME/.bashrc
source $HOME/.bashrc

cd vanity_torv3_cuda
pip install -r util/requirements.txt
mkdir build && cd build
cmake .. -G Ninja
ninja
```

