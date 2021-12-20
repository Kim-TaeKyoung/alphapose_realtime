from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.

file_id = '1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile(os.environ['ALPHAPOSE_DIR'] + 'detector/yolo/data/yolov3-spp.weights')

file_id = '1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile(os.environ['ALPHAPOSE_DIR'] + 'detector/tracker/data/JDE-1088x608-uncertainty')

os.rename(os.environ['ALPHAPOSE_DIR'] + 'detector/tracker/data/JDE-1088x608-uncertainty',\
          os.environ['ALPHAPOSE_DIR'] + 'detector/tracker/data/jde.1088x608.uncertainty.pt')

file_id = '1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile(os.environ['ALPHAPOSE_DIR'] + 'pretrained_models/fast_res50_256x192.pth')
