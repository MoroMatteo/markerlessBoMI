# Abstract



# UBUNTU installation steps:

**TESTED with python>=3.7**

step 0 --> open a terminal and cd to the home folder:

``` 
    $ cd
``` 

step 1 --> [if not done yet] install pip and venv:

``` 
    $ sudo apt-get update
    $ sudo apt install python3-pip
    $ sudo apt install build-essential libssl-dev libffi-dev python3-dev
    $ sudo apt install python3-venv
``` 

step 2 --> create a virtual environment called BoMI:

``` 
    $ python3 -m venv BoMI
``` 

step 3 --> activate virtual enviroment:

``` 
    $ source BoMI/bin/activate
``` 

step 4 --> upgrade pip and install all the packages nedded for markerlessBoMI:

``` 
    $ pip install --upgrade pip
    $ pip install git+https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 5 --> istall tkinter:

``` 
    $ sudo apt install python3-tk
``` 

step 6 --> clone the github repository:

``` 
    $ git clone https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 7 --> cd in the correct folder and run main_reaching.py:

``` 
    $ cd markerlessBoMI_FaMa/
    $ python3 main_reaching.py
``` 

step 8 --> follow the steps in the GUI (see below after WINDOWS installation steps)

# MAC installation steps:

**TESTED with python>=3.7**

step 0 --> install python (if not already installed), pip and virtualenv

step 1 --> create a virtual environment called BoMI:

``` 
    $ python3 -m venv BoMI
``` 

step 2 --> activate virtual enviroment:

``` 
    $ source BoMI/bin/activate
``` 

step 3 --> upgrade pip and install all the packages nedded for markerlessBoMI:

``` 
    $ pip install --upgrade pip
    $ pip install git+https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 4 --> istall tkinter:

``` 
    $ pip install tk
``` 

step 5 --> clone the github repository:

``` 
    $ git clone https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 6 --> cd in the correct folder and run main_reaching.py:

``` 
    $ cd markerlessBoMI_FaMa/
    $ python3 main_reaching.py
``` 

step 7 --> follow the steps in the GUI (see below after WINDOWS installation steps)


# WINDOWS installation steps:

**TESTED with python>=3.7**

step 0 --> download Python3 at this link https://www.python.org/downloads/ 

step 1 --> enable long path (https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/) --> remember to launch regedit as root ("amministratore")

step 2 --> open a command window (terminal) as root ("amministratore") and type 

``` 
    $ cd
``` 

step 3 --> install pip and virtualenv

``` 
    $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $ python get-pip.py
    $ pip install virtualenv
``` 

step 4 --> create virtualenv (named BoMI) and activate it

``` 
    $ python3 -m venv BoMI
    $ BoMI\Scripts\activate
``` 

step 5 --> Upgrade pip and download all the following packages (in the terminal):

``` 
    $ pip install --upgrade pip
    $ pip install numpy
    $ pip install pandas
    $ pip install matplotlib
    $ pip install pygame
    $ pip install pyautogui
    $ pip install tensorflow
    $ pip install mediapipe
    $ pip install scipy
    $ pip install sklearn
``` 

step 6 --> download Visual Studio Code (https://code.visualstudio.com/download)

Step 7 --> Download the repository from the current github page and open it as a project in Visual Studio Code

step 8 --> Set the correct python interpreter (the one of the virtual environment created - BoMI\Scripts\python) in Visual Studio Code from the bottom left corner: left click on the bottom left corner and follow the instruction searching for BoMI\Scripts\python.

step 9 --> eventually [not always] there is the possibility that it is necessary to do the steps described here https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads

step 10 --> On Visual Studio Code run the file main_reaching.py

# Graphical User Interface

If you run the script main_reaching.py, you should see something like this:

![BoMI_0](https://user-images.githubusercontent.com/75327840/142367411-f78e3f09-481a-4edd-95c4-f8a23778b0e2.png)

*Figure 1. Markerless BoMI GUI.

**Step 1: Select Joints**. As you can notice, only the button **Select Joints** is available. You should select the anatomical joints you want to use to control the cursor (click on the corresponding checkbox, try for example **Nose** and **Shouders**), and, then, press **Select Joints**:

![BoMI_1](https://user-images.githubusercontent.com/75327840/142368575-1152f407-1e94-474d-acee-292905ff7db8.png)

*Figure 2. Select the anatomical points you want to use to control the cursor and then press Select Joints.

After this step, also the others buttons should be visible. To complete the task, you have to follow the following steps **in order**.

**Step 2: Calibration**. During the calibration step, you should move the anatomical points you selected during the previous step facing your webcam. Make sure only you are visible from the webcam with nobody else (only one person should be visible). Press the button **Calibration**: a new window will pop up (see Figure 3). Press **ok** and start moving your joints facing the webcam for 30 seconds (a countdown as in Figure 4 will appear). **During the calibration try to cover with your movements all the space seen by your webcam**. The calibration is an important step and it is possible that you have to do it multiple times.

![BoMI_3](https://user-images.githubusercontent.com/75327840/142370385-1e202942-34ce-4f54-8f0b-eb73aa30f8cc.png)

Figure 3. Window that will pop up after pressing the button **Calibration**. Press **ok** and start moving your selected joints.

![BoMI_4](https://user-images.githubusercontent.com/75327840/142370626-20db02ec-17ff-48bf-bf3b-af29bd5de480.png)

Figure 4. Countdown during calibration. You should move for 30 seconds.
