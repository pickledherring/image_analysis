# This is a class project for Image Analysis on cancer classification

main.ipynb and main.py are essentially the same, but main.py's batch_process function allows for saving histogram outputs. Otherwise, I apologize if there are other discrepancies that were not updated to make the two equal. I recommend using main.py for batch processes and main.ipynb for graphs of histograms and processing individual images.

For an easier interface, there is run_me.py. Follow the intructions therein, save the file if necessary, and run it using python 3 at a terminal at the root level of this folder, for example `python3 run_me.py`.

You will also need glob, matplotlib, and PIL, which can be installed by running `pip3 install -r requirements.txt` in a terminal window, assuming you already have python 3.

eval.py is there to time every function and will take hours to run. This is partially because it opens and saves every image for each operation. For timing on just one transformation, you can subtract the time to run "grayscale alone" from the time of your transformation. Note that this will be very imprecise.

At the end of main.py/.ipynb is code to extract features from images and classify them based on their file path name. There is also code to draw bounding boxes on images, if desired. There is some data extracted from a selection of the images in cells.csv.