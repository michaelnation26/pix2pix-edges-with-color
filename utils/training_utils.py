import os

from matplotlib import pyplot


def save_results(gen_model, d_model, data_generator, epoch_num, output_dir, save_models):
    output_imgs_dir = os.path.join(output_dir, 'images')
    output_models_dir = os.path.join(output_dir, 'models')
    os.makedirs(output_imgs_dir, exist_ok=True)
    os.makedirs(output_models_dir, exist_ok=True)

    for idx, (imgs_source, imgs_target_real, _, _) in enumerate(data_generator):
        imgs_target_fake = gen_model.predict(imgs_source)
        n_examples = len(imgs_source)

        # scale all pixels from [-1,1] to [0,1]
        imgs_source = (imgs_source + 1) / 2.0
        imgs_target_real = (imgs_target_real + 1) / 2.0
        imgs_target_fake = (imgs_target_fake + 1) / 2.0

        # plot source images
        for i in range(n_examples):
            pyplot.subplot(3, n_examples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(imgs_source[i])

        # plot generated target image
        for i in range(n_examples):
            pyplot.subplot(3, n_examples, 1 + n_examples + i)
            pyplot.axis('off')
            pyplot.imshow(imgs_target_fake[i])

        # plot real target image
        for i in range(n_examples):
            pyplot.subplot(3, n_examples, 1 + n_examples*2 + i)
            pyplot.axis('off')
            pyplot.imshow(imgs_target_real[i])

        # save plot to file
        img_output_filename = f'plot_{epoch_num:05d}_{idx}.png'
        filepath = os.path.join(output_imgs_dir, img_output_filename)
        pyplot.savefig(filepath, dpi=400)
        pyplot.close()

    if save_models:
        gen_model_output_filename = os.path.join(output_models_dir, f'{epoch_num:05d}_gen_model.h5')
        d_model_output_filename = os.path.join(output_models_dir, f'{epoch_num:05d}_d_model.h5')
        gen_model.save(gen_model_output_filename)
        d_model.save(d_model_output_filename)
