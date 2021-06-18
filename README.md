# NTUICG_final-image_morphing
Please go to https://python-poetry.org/docs/ to install poetry first.  
With poetry installed,  
- cd into the repo directory  
- Download https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks_GTX.dat.bz2 and extract it in image_morphing directory
- run 'poetry install'  
- run 'poetry run python image_morphing/\_\_init\_\_.py -a images/img4.jpg -b images/img5.jpg -o output.mp4'
