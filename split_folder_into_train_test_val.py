"""
Code for splitting folder into train, test, and val.
 
pip install split-folders
"""
import splitfolders  # or import split_folders

input_folder = 'data/'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output="data2", seed=42, ratio=(.80, .1, .1), group_prefix=None) # default values

# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
#splitfolders.fixed("input_folder", output="data2", seed=1337, fixed=(100, 100), oversample=False, group_prefix=None) # default values


"""
For waterashed semantic segmentation the folder structure needs to look like below


new_dataok/
    train_images/
                train/
                    img1, img2, img3, ......
    
    train_masks/
                train/
                    msk1, msk, msk3, ......
                    
    val_images/
                val/
                    img1, img2, img3, ......                

    val_masks/
                val/
                    msk1, msk, msk3, ......
      
    test_images/
                test/
                    img1, img2, img3, ......    
                    
    test_masks/
                test/
                    msk1, msk, msk3, ......
      
                
"""
