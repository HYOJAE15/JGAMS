Install JGAMS on Your Desktop
=============================

Follow these steps to install CHALK on your desktop:

0\. Create the conda venv & activate it
-----------------------------

Create the new venv:

```
conda create -n {venv} python=3.8 -y
```

Activate it:

```
conda activate {venv}
```

1\. Install JGAMS
-----------------------------

```
git clone --recurse-submodules https://github.com/HYOJAE15/JGAMS.git
```

2\. Install Required Packages
-----------------------------

Install the necessary packages by running the following command in your terminal or command prompt:

```
cd JGAMS
pip install -r requirements.txt
```

3\. Install PyTorch Compatible with Your Desktop Environment
------------------------------------------------------------

JGAMS has been tested with PyTorch 2.2.2 To install the appropriate version of PyTorch for your system, visit the PyTorch previous versions page:

[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

Here's an example installation command for CUDA 12.1:


```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

4\. Install MMSegmentation Using the OpenMIM Installer
------------------------------------------------------

To install MMSegmentation and its required dependencies, run the following commands:


```
pip install -U openmim
mim install mmcv 
mim install mmsegmentation
```

5\. Install the Segment Anything Model from the FAIR GitHub Repository
----------------------------------------------------------------------

Install the Segment Anything Model by running this command:


```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

6\. Install Grounding DINO
----------------------------------------------------------

Install Grounding DINO by running this command:

```
cd submodules/GroundingDINO
pip install -e .
```

7\. Setup JGAMS 
---------------

To setup JGAMS, run the following command:

```
python setup.py develop
```

8\. Download checkpoint
-------------------------------------------------------

Checkpoints for the JGAM function are available at the following links and should be placed in the `dnn/checkpoints` directory:

[Segment Anything Model (vit_h)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

[GroundingDINO-B](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth)

Once you've completed these steps, JGAMS should be installed and ready to use on your desktop.
