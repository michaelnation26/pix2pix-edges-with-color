# pix2pix-edges-with-color

Extension of the [pix2pix (edges2shoes) implementation](https://github.com/phillipi/pix2pix).

In the original implementation, the input image is black and white, only containing the edges of the shoe. After the GAN model is trained, there is no way to control the color of the output image.

In this project, color is added to the input image during training. Resulting in a generator model where the color in the output image corresponds to the color in the input image.

## Training Data

Only the `Sneakers and Athletic Shoes` images from the [UT Zappos50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) dataset were used, resulting in 12,860 training examples.

Canny edge detection was used to extract the edges of the shoes. The edges are 1 pixel in width.

350 points were randomly sampled from all non-white pixels in the original image. For each point, the RGB values of the surrounding pixels are averaged, resulting in a 3x3 pixels square. The 350 color points are overlayed onto the black and white edges image.

![preprocessing steps](images/training_data_preprocess_steps.png)


### THE CODE FOR THIS PROECT IS COMPLETE. THE WRITEUP FOR THIS PROJECT IS STILL IN PROGRESS.
