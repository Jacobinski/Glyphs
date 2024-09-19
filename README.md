## Installation Notes (Windows; GPU)

Commands used to get PaddleOCR with GPU support running on Windows:
```
conda create --name subtitles python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda activate subtitles

conda install python=3.12
conda install paddlepaddle-gpu==3.0.0b1 paddlepaddle-cuda=12.3 -c paddle -c nvidia

pip install opencv-python
pip install srt
pip install paddlepaddle-gpu
pip install "paddleocr>=2.0.1"
pip install nltk
```

## Installation Notes (Mac M3 Pro)
Install the nightly version to avoid issues with a hanging PaddleOCR thread: https://github.com/PaddlePaddle/PaddleOCR/issues/11706
```
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html

pip install opencv-python
pip install srt
pip install "paddleocr>=2.0.1"
pip install nltk
pip install setuptools
```