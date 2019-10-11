# pix2pix-edges-with-color

Extension of the [pix2pix (edges2shoes) implementation](https://github.com/phillipi/pix2pix).

In the original implementation, the input image is black and white, only containing the edges of the shoe. After the GAN model is trained, there is no way to control the color of the output image.

In this project, color is added to the input image during training. Resulting in a generator model where the color in the output image corresponds to the color in the input image.

## Dataset

Only the `Sneakers and Athletic Shoes` images from the [UT Zappos50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) dataset were used, resulting in 12,860 training examples. All images were resized from 136x136 to 256x256.

Canny edge detection was used to extract the edges of the shoes. The edges are 1 pixel in width.

350 points were randomly sampled from all non-white pixels in the original image. For each point, the RGB values of the surrounding pixels are averaged, resulting in a 3x3 pixels square. The 350 color points are overlayed onto the black and white edges image.

![preprocessing steps](images/training_data_preprocess_steps.png)

## GAN Model

Jason Browniee's [Keras implementation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) of the Pix2Pix GAN was used.

## Experimentation

### Number of Color Points

Choosing the number of color points to randomly sample for the input image had a large impact on how the generated output image would look.

If the number of points is too small, there will be much more white space in the input image. Resulting in a model that is more likely to produce color in the output image in sections that were meant to be white. This is not ideal. If the toe cap of a shoe is white in the input, it should remain white in the output.

If the number of points is too large, the validation/test input images will also require a large number of color points/lines in the input or the output will have a lot of white patches. Ultimately, the user will manually draw the black shoe edges and add color points/lines. The user experience would not be great if the user is forced to fill in large regions with color. Ideally, the user would only have to add a few color points/lines to each region and the model will know to fill in the entire region with that color.

### THE CODE FOR THIS PROECT IS COMPLETE. THE WRITEUP FOR THIS PROJECT IS STILL IN PROGRESS.
