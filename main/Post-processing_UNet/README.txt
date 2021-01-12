This folder contains the code to generate the training/testing results for the post-processing 
network. 

Initial idea: to teach a basic encoder-decoder (Unet) to connect broken edges in a 
cytoplasm edge map. 

Method: 
    - train the network on the 10_cells_edge_data (500 examples from 0-50% overlap)
        - 400 train, 100 test
    - this will give us 100 predicted edge maps from the test examples, together with their GTs
    - I'll then use these 100 predicted/GT edge maps to train ANOTHER Unet, this time the input will
      be the predicted edge map, the output will be another edge map.
    - loss then computed by comparing the new predicted edge map with the GT one
    - network slowly learns to reconstruct the broken edge map using Dice loss.
    
Another way (probably better):
    - pre-train a Unet (1) on the 10_cells_edge data: this becomes a data generating network.
    - use the 1st network (1) to generate training data for a second network (2): this will be trained on 
      the generated edge maps, and become the correction network (2) for repairing broken edges.
    - now that we have a correction network (2), use the segmentation results from our original U-Net (i.e. 
      edge maps generated from 90 test data) as inputs into our newly trained correction network (2) for connecting
      the edges and improving the segmentation results.
      
      
Date: 16.03.2020.