For the download link and details of the LoRa dataset, see https://github.com/gxhen/LoRa_RFFI



To train a model, run `train.py`. 

Please modify the parameters (`dataset_path`, `saved_model_path`, `model_name`, `run_for`) in the `main` function as needed.



To evaluate the trained model, run `eval.py`.

The parameters in the `main` function may need to be modified as needed. 

The LoRa dataset includes many test subsets. Please see https://github.com/gxhen/LoRa_RFFI for details. The path and name of the test dataset need to be modified for different scenarios. The device index range should also be modified according to the dataset. For example, the test set named *dataset_rogue.h5* includes devices indexed 41-45, then the index range in the code should be modified as [40, 45] (the loaded labels start from 0 but not 1 to adapt to deep learning, i.e., device 1 is labeled 0 and device 2 is labeled 1 and so forth.). 



Note that since Pytorch requires that the convolutional layer cannot use stride values greater than 1 when using padding, we do not adopt `padding='same'` for the first layer of LoRaNet, which is a slight difference from the original LoRaNet. 



