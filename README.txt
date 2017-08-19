/**********************************************************************************************************
*
* Building a Filipino Colloquialism Machine Translator (FCMT) using Sequence-to-Sequence Model
* by: Nicco Nocon and Nyssa Kho, 2017
* Institution: De La Salle University - Manila
* E-mail: noconoccin@gmail.com and nyssa_kho@dlsu.edu.ph
*
* Colloquialism in the Philippines has been prominently used in day-to-day conversations. Its vast 
* emergence is evident especially on social media platforms, but poses issues in terms of understandability 
* to certain groups. For this research, a machine translator using sequence-to-sequence model has been 
* implemented to fill in that gap. The translator covers Filipino Textspeak or Shortcuts, Swardspeak or 
* Gay-lingo, Conyo, and Datkilab – implemented on Tensorflow library. Implementing in Tensorflow achieved 
* 85.88 BLEU score when evaluated to the training data and 14.67 to the test data.
*
**********************************************************************************************************/

I. Under the Package:
-----------------------------------------------------------------------------------------------------------
	\__init__.py		|
	\data_utils.py 		|
	\seq2seq_model.py 	|
	\translate.py		- Tensorflow Libraries for Sequence-to-Sequence Model
	\decode.py			- API for Translation
	\main.py			- Sample Code for using decode.py API
	\fcmt.py			- Interactive Loop Program implementing Filipino Colloquialism Machine Translator
	data\ 				- Train.* and Test.* files
	data\train 			- Directory for adding the training model

II. System Requirements:
-----------------------------------------------------------------------------------------------------------
	Python
	Tensorflow v 1.0.1

III. Instructions for installing Python on Windows:
-----------------------------------------------------------------------------------------------------------
	1) Download Python 3.6 @ https://www.python.org/downloads/
	2) Launch the downloaded exe file
	3) Follow the Installation Wizard
	3) Open your Terminal (e.g. Command Prompt)
	4) Type 'python -V'  (without quotes) and press enter to see the python version installed.
	5) Type 'python' to enter python shell
	6) Type 'exit()' to exit python shell

IV. Instructions for installing Tensorflow on Windows:
-----------------------------------------------------------------------------------------------------------
	1) Install Tensorflow via native pip. Follow instructions at 
			https://www.tensorflow.org/install/install_windows
	2) Downgrade the Tensorflow version to v 1.0.1 by typing
			pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl
	3) Don't forget to validate your installation. See instructions at 
			https://www.tensorflow.org/install/install_windows

V. Building the FCMT Model:
-----------------------------------------------------------------------------------------------------------
	1) Open your terminal (e.g. Command Prompt)
	2) Navigate to the downloaded FCMT folder by typing:
	cd [directory]/FilipinoColloquialismMachineTranslator
	(ex. cd C:\Users\Username\Python_Files\Translator_Project\FilipinoColloquialismMachineTranslator)
	3) Type 'python translate.py' to initiate model training. Global steps built for FCMT is 8200, but you 
	can exceed the given value to experience better performance. Also, feel free to modify our translator 
	(see Section VIII for more details) and if FCMT's performance increased, let us know about it so we can 
	update our software.
	4) Training the model is stopped (Ctrl+C) manually at any step, recording sequences in every 200 steps. 
	The trained model will be stored at data\train directory.
	* Alternative: download the full package on our Google Drive - https://goo.gl/nRCeV4
		
VI. How to use:
-----------------------------------------------------------------------------------------------------------
	1) Open your terminal (e.g. Command Prompt)
	2) Navigate to the downloaded FCMT folder by typing:
	cd [directory]/FilipinoColloquialismMachineTranslator
	(ex. cd C:\Users\Username\Python_Files\Translator_Project\FilipinoColloquialismMachineTranslator)
	3) Type 'python fcmt.py' to start the interactive loop
	4) A '>' prompt will appear. You may now enter any colloquial text for translation
	5) Press Ctrl+C to exit the interactive loop

VII. Integrate FCMT with your Python Project:
-----------------------------------------------------------------------------------------------------------
Type the following in your python code (see sample on main.py):

	import decode # imports the decode (inside decode.py) functionality to your python file

	colloqtext = "ano , bro ? valk later ? g ?"  # colloquial text here
	translation = decode.decode(colloqtext)  # code for using the decode functionality.
	print(translation) # translation is the translated version of the input text

VIII. Play around with the configuration:
-----------------------------------------------------------------------------------------------------------
 1. Increase texts in parallel corpus by modifying train.enc and test.enc (for colloquial text - source 
 language) and train.dec and test.dec (for Filipino text - target language). Parallel corpus can be found 
 at FilipinoColloquialismMachineTranslator/data/
 2. Modify the learning rate, number of layers, batch size, layer size and vocab size by editing the 
 translate.py (training), fcmt (interactive loop) and decode.py (api) 
 
 * Line numbers 50-75: tf.app.flags.DEFINE_xxxx()
 * has to be same values for train and execution of program
 * Run training using 'python translate.py'
 * It is recommended to create a folder for every new model to train.
 
IX. Reference:
-----------------------------------------------------------------------------------------------------------
- Nocon, N. and Kho, N. (2017). Building a Filipino Colloquialism Translator using Sequence-to-Sequence 
  Model. De la Salle University, Manila. Under Consideration in IALP 2017.
