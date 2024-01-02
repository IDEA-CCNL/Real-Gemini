wget http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz
tar -zxvf pa_stable_v190600_20161030.tgz 
cd portaudio/
./configure && make && sudo make install 
pip install pyaudio
