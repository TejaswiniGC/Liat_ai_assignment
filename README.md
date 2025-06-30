## Player Re-Identification in Sports Footage
`Assignment Task Chosen: Option 2: Re-Identification in a Single Feed (15sec_input_720p.mp4)`

This repository contains my attempt to implement a solution for the Liat.ai AI Intern assignment. My main goal was to accurately track players in a given video clip and ensure they keep a consistent ID, even if they temporarily move out of view and reappear.

Getting this project set up and running involved a lot of learning and troubleshooting, which was a really valuable experience in practical problem-solving. This README explains all the steps I took, including the challenges I faced and how I worked through them, so that someone else can easily reproduce my setup.

## 1. How to Set Up and Run the Code for getting it to work.
This section details the exact steps I followed and the debugging I did to make the player re-identification demo functional.

## 1.1 Prerequisites
  Before you start, please make sure you have these tools installed on your Windows computer:

  Miniconda or Anaconda: I used this to create and manage my Python environment, which helps keep all the project's specific libraries separate.

  Git: You'll need this to download (clone) this project's code.

## 1.2 Clone the Project Code
  First, open your PowerShell (or Anaconda Prompt). Navigate to the folder where you want to keep this project. Then, use this command to download the code:

  git clone https://github.com/ifzhang/ByteTrack.git
  cd ByteTrack

## 1.3 Create and Activate a Clean Python Environment
  Next, I created a new Conda environment with a specific Python version (3.10). This helps prevent conflicts with other Python projects you might have.

  conda create -n bytetrack_py310 python=3.10 -y
  conda activate bytetrack_py310

## 1.4 Install Core Libraries (My First Set of Challenges)
  Getting the main libraries like PyTorch and onnxruntime installed correctly was one of the first tricky parts. I made sure to install the CPU version of PyTorch since my computer doesn't have a NVIDIA GPU.

## Install PyTorch (CPU version) and TorchVision
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## Install onnxruntime (This installation caused some initial issues in earlier attempts,
# but installing it directly here seemed to work better)
  pip install onnxruntime

## 1.5 Install Additional Tracking-Specific Libraries
  Some other libraries needed for the tracking part of the project required specific installation methods. I found these errors during runtime and fixed them as I went:

## Install pycocotools (This fixed a 'ModuleNotFoundError: No module named 'pycocotools'')
  conda install -c conda-forge pycocotools -y

## Install cython_bbox (This fixed another 'ModuleNotFoundError: No module named 'cython_bbox'')
  pip install cython_bbox

## 1.6 Install Remaining Project Requirements
  After handling the more challenging libraries, I installed the rest of the project's requirements listed in the requirements.txt file:

  pip install -r requirements.txt --no-cache-dir

## 1.7 Install ByteTrack in "Editable" Mode
  To make sure Python could find all the project's files and that any small changes I made to the code would apply immediately, I installed the project in "editable" mode:

  pip install -e .

## 1.8 Download and Prepare the Player Tracking Model
  The assignment required a pre-trained model. I downloaded the bytetrack_x_mot17.pth.tar checkpoint because I learned it's well-suited for tracking people in videos like the ones from the MOT17 dataset, which seemed very relevant for football players.

  Download Command:

  pip install gdown
  gdown --id 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O bytetrack_x_mot17.pth.tar

  `(A note on model choice: While the assignment mentioned a YOLOv11 model, I chose this YOLOX-X based model as my research indicated it worked more directly with the ByteTrack algorithm's specific requirements for person tracking. Initially, I tried to use a model given to me i.e "best.pt". However, this caused several problems during the model loading phase and ultimately, it was incompatible with the ByteTrack framework as it was set up.)`

  Placement: After downloading, I made sure the bytetrack_x_mot17.pth.tar file was placed directly into the main ByteTrack project directory (e.g., C:\Users\tejas\Liat_Track\ByteTrack\).
 (Initial tests showed that the torch.load command was able to directly handle this .tar archive, so I didn't need to manually extract the .pth file first after solving other issues.)

## 1.9 Important Code Adjustments (My Debugging Diary)
  As I worked, I found that I needed to make a few small but very important changes to the ByteTrack code to make it compatible with my environment and the chosen model. These changes were guided by the error messages I received:

  Fixing Model Loading in tools/demo_track.py:

  The Problem: When I tried to load the model, PyTorch gave me errors like TypeError or RuntimeError complaining about how the model checkpoint (.pth.tar file) was structured compared to what the program expected. It seemed like the model file contained extra information beyond just the "weights" in a way the code didn't anticipate.

  My Solution: I edited C:\Users\tejas\Fresh_Track\ByteTrack\tools\demo_track.py.

  Around Line 349, I changed ckpt = torch.load(ckpt_file, map_location="cpu") to:

  ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

  (This told PyTorch to load the entire checkpoint, not just try to get "weights only", which was necessary for this specific model file.)

  Adjusting Class Count in exps/example/mot/yolox_x_mix_det.py:

  Fixing NumPy Deprecation in yolox/tracker/matching.py:

  The Problem: I encountered an AttributeError: module 'numpy' has no attribute 'float'. This happened because my NumPy version was newer and np.float was an old way of specifying a floating-point type that is no longer directly supported.

  My Solution: I edited C:\Users\tejas\Fresh_Track\ByteTrack\yolox\tracker\matching.py.

  Around Line 61, I changed ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float) to:

  ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)

  (This specified a standard 64-bit float type for the NumPy array, resolving the error.)

  Fixing Another NumPy Deprecation in yolox/tracker/byte_tracker.py:

  The Problem: A very similar AttributeError: module 'numpy' has no attribute 'float' appeared in another file.

  My Solution: I applied a similar fix by editing C:\Users\tejas\Fresh_Track\ByteTrack\yolox\tracker\byte_tracker.py.

  Around Line 18, I changed self._tlwh = np.asarray(tlwh, dtype=np.float) to:

  self._tlwh = np.asarray(tlwh, dtype=float)

  (This used the standard Python float type, which NumPy handles correctly.)

## 1.10 Running the Demo (Finally Seeing It Work!)
  After all the environment setups and careful code modifications, I could finally run the player re-identification demo.

  Setting the OpenMP Environment Variable: This step was important to prevent a common warning (OMP: Error #15) about conflicting  libraries. I ran this command in my PowerShell session before running the main script:

  $env:KMP_DUPLICATE_LIB_OK="TRUE"

  Executing the Demo Script: Then, from my ByteTrack directory with my Conda environment active, I ran the main command:

  python tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c bytetrack_x_mot17.pth.tar --path 15sec_input_720p.mp4 --save_result --device cpu --conf 0.3 --track_thresh 0.5 --min_box_area 5 --match_thresh 0.9 --track_buffer 90

 The 15sec_input_720p.mp4 video was placed in the main ByteTrack directory.

 The console output showed that the model loaded successfully and started detecting a good number of players in each frame with high confidence, which was a great sign!

## 1.11 Output Location
 Once the script finishes processing the video (it takes some time on CPU), the output video with bounding boxes and tracking IDs will be saved in a new folder, typically found here:
./YOLOX_outputs/yolox_x_mix_det/track_vis/[timestamp]/15sec_input_720p.mp4

`2. What Remains and How I Would Proceed (Since My Solution is Incomplete)`
  While I successfully got the system to detect players and perform basic tracking, fully consistent player re-identification, especially in complex scenarios, is still a challenge that needs more work beyond what I could achieve within the assignment's timeframe.

## 2.1 My Current Challenges with Re-Identification
  The main issues I observed with ID consistency are:

  ID Swaps When Players Cross: When players move very close to each other or cross paths, their assigned IDs sometimes get swapped.

  New IDs for Reappearing Players: If a player leaves the camera's view for a while and then comes back into the frame, they are usually assigned a completely new ID instead of their original one.

## 2.2 My Future Plans / How I Would Continue if I Had More Time
  If I had more time and resources, here's how I would try to improve the re-identification:

  Adding Appearance Information: The current tracking relies mostly on where a player is and how they move. For better re-identification, especially after long occlusions, the system needs to "remember" what each player looks like. I would investigate:

  Deep SORT: This is a popular method that adds a deep learning model to learn unique visual features (like a "face" or "jersey") for each player. This would help match players even when their movement or overlap is unclear.

  Specialized Re-ID Models: I would look into using dedicated models that are trained specifically to tell different people apart based on their appearance.

  Connecting Broken Tracks: I would also try to develop a way to look at short, disconnected tracks and figure out if they actually belong to the same player, then connect them.

  More Tuning of Settings: I would spend more time systematically testing different combinations of the tracking parameters (--track_thresh, --match_thresh, --track_buffer). I would try to find the perfect settings that make the IDs as stable as possible for this type of video.

  Using a GPU: My current setup runs on the computer's main processor (CPU), which is slow for this kind of task. If I had access to a graphics card (GPU), it would make the processing much faster. This would let me test changes and see results much more quickly, and potentially even handle higher quality video inputs.
  
=======
# Liat_ai_assignment
