# cd C:\WorkSpace\Practice2
echo "1 ----------------------------------------------------------------------"
python -m venv env
echo "2 ----------------------------------------------------------------------"
.\env\Scripts\activate
echo "3 ----------------------------------------------------------------------"
python -m pip install --upgrade pip
echo "4 ----------------------------------------------------------------------"
pip install opencv-python numpy tqdm tensorflow scikit-learn matplotlib pydot
echo "5 ----------------------------------------------------------------------"
echo "Download GraphVis: https://graphviz.org/download/ "
echo "and reboot"
