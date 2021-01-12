This folder contains the code to generate the training results for the different models:
	- vanilla Unet 
	- vanilla Unet w/ deep supervision
	- Unet with vanilla Attention Gate (ie no edge_block)
	- Unet with Edge Attention Gate (with edge block: residual conv on low-level feature maps)

The models_DS.py is necessary for training/testing.
Models without DS should be trained using models.py and tested with the same models.py,
otherwise the edge_conv() parameters are loaded onto the state.dict() of the model, which 
weren't there during training: ERRORS

Same works with the DS models - train it on models_DS.py, but then run predict.py and have:

from models_DS.py import Unet_DS, Att_Unet, Edge_Att_Unet

so that there is no confusion between edge_conv() parameters.


-------------------------
Steps for getting results
-------------------------

Step 1: get the model predictions.

predict.py:
	- once the model has been trained, the main folder (ISBI_data_code) will contain the model 		  checkpoints for every 10 epochs (in model_checkpoint) and the training log with the total 		  avg. loss/epoch and training times (in train_log) for the model you trained. 
	- copy these two folders (model_checkpoint and train_log) and paste them in the directory for 		  that model and associated loss function (e.g. Unet_experiments -> Dice_loss -> {here}).
	- go to predict.py and change the following:
		- root_dir: change this to the path to where you saved the model_checkpoint and 		  train_log folders (e.g. './Unet_experiments/Dice_loss').
		- model name: change this to the name of the model we want to get predictions for by 			   going to the variable "net" (e.g. Unet).
	- with these changes done, open the terminal and type "python predict.py". This will send all 		  the images (inputs, cyto_edges, nuclei) and performance results to the same place where 		  model_checkpoint and train_log are.

Step 2: overlay predicted/GT edges on top of input images.

show_results.py:
	- now that the predictions are done and stored in their respective class folders, go to 	  show_results.py and change the root_dir to the same as predict.py (ie where everything 		  about that model has been stored)
	- all the code does is read in each .png file, find the edge coordinates (both gt and pred), 		  and plot them on top of the original images: cytoplasms (yellow), nuclei (red).
	- open terminal and type "python show_results.py"
	- this is all that you need to do: the code will do the rest!
	

Result: the edge_results_images folder will contain the gt_edges and pred_edges for each image in the test set. Classes are colour coordinated.
 
Now repeat for all models, and all respective loss functions to get all the edge_results and performance results you need.


FOR DEEP SUPERVISION MODELS:

There are a few other changes to make to the code for these versions of the models.
Because DS requires edge maps from intermediate layers, 
outputs = model(inputs) has x1_edge, x2_edge, outputs = model(inputs). 

This means that in predict.py, we need to ignore the edge outputs which are only required during training for learning the edges better. Therefore, to fix this, simply do 
" _, _, outputs = net(test_img) " in predict.py. 

This has already been done, so just uncomment the line under "For deep supervision models" and comment the line under "Regular outputs" .

Date: 05.03.2020.

===========================
