# Fire detection 

* Open a Conda prompt. Switch to the environment firedetection. 

<code>
conda create --name firedetection python=3.8 
conda activate firedetection

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

conda install -c conda-forge ultralytics

</code>

* We have scanned some homam videos and extracted frames out of them to get some training images of somewhat uncontrolled fires. Process the videos using the following command. It will create a frame every 50 frames and saves it to disk for use in labelling and training. 

<code>
python extract_homam_images.py "..\videos\homam1.mp4"

python extract_homam_images.py "..\videos\homam2.mp4"
</code>

* We now need to download some of the videos from youtube to get a good selection of types of fires we might encounter in a household. Using these diverse images can give us a good base for training the ML model.

<code>
python extract_homam_images.py "..\videos\kitchen1.mp4"

python extract_homam_images.py "..\videos\battery.mp4"

python extract_homam_images.py "..\videos\gas_cylinder.mp4"

python extract_homam_images.py "..\videos\gas_stove.mp4"

python extract_homam_images.py "..\videos\electric.mp4"

python extract_homam_images.py "..\videos\fireplace.mp4"

python extract_homam_images.py "..\videos\microwave.webm"

python extract_homam_images.py "..\videos\cigarate.mp4"

python extract_homam_images.py "..\videos\toaster.mp4"

python extract_homam_images.py "..\videos\electric_spark.mp4"

</code>

* Now we have to start labeling these images for use.

<code>
pip install pyqt5 lxml --upgrade
</code>