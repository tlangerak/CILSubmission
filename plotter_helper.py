import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import os
from skimage import io

def evaluation_side_by_side_plot(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs[0].permute((1, 2, 0)).numpy())
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs, cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth[0].view((400, 400)), cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)


def evaluation_side_by_side_plot_np(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs)
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs, cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth, cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = 255 * predicted_img
    color_mask[:, :, 1] = 165 * predicted_img

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    img9 = img_float_to_uint8(predicted_img)
    alpha = Image.fromarray(255 - img9, "L")

    blended_overlay = Image.blend(background, overlay, 0.7)
    new_img = Image.composite(background, blended_overlay, alpha)
    return new_img


def overlay_side_by_side(img, ground_truth, prediction, save=False, save_name="plot.png"):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(make_img_overlay(img, ground_truth))
    ax1.set_title("Ground truth")
    ax2.imshow(make_img_overlay(img, prediction))
    ax2.set_title("Prediction")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)

if __name__ == "__main__":
    PREDICT_TEST = False
    PREDICT_TRAINING = False
    CREATE_GROUND_TRUTH_OVERLAY = False
    if PREDICT_TEST:
        for i in range(1, 224):
            filename = "predictions_test/scaled/test_prediction_" + str(i) + ".png"
            filename_im = "test/test_" + str(i) + ".png"
            if not os.path.isfile(filename):
                continue
            print("Loading image {}".format(filename))

            # Only prediction
            prediction = Image.open(filename)
            prediction = prediction.convert('L')
            prediction = np.asarray(prediction)
            image = io.imread(filename_im)
            overlay_image = make_img_overlay(image, prediction)
            overlay_image.save("predictions_test/scaled/overlay_" + str(i) + ".png")
    if PREDICT_TRAINING:
        for i in range(1, 101):
            filename_predictions = "predictions_training/augmented/test_prediction_" + str(i) + ".png"
            filename_images = "train/images/satImage_" + str(i).zfill(3) + ".png"
            filename_ground_truth = "train/target/satImage_" + str(i).zfill(3) + ".png"
            print("Loading image {}".format(filename_images))

            prediction = Image.open(filename_predictions)
            prediction = prediction.convert("L")
            prediction = np.asarray(prediction)
            image = io.imread(filename_images)
            ground_truth = io.imread(filename_ground_truth)

            side_by_side = overlay_side_by_side(image, ground_truth, prediction, save=True,
                                                save_name="predictions_training/side_by_side_" + str(i) + ".png")

    if CREATE_GROUND_TRUTH_OVERLAY:
        folder = "data/original/train/"
        for i in range(1,101):
            filename_images = folder + "input/satImage_" + str(i).zfill(3) + ".png"
            filename_ground_truth = folder + "target/satImage_" + str(i).zfill(3) + ".png"

            image = Image.open(filename_images)
            raw_image = np.array(image)
            ground_truth = np.array(Image.open(filename_ground_truth))


            overlay_image = make_img_overlay(raw_image, ground_truth)

            new_im = Image.new('RGBA', (800, 400))
            new_im.paste(image.convert("RGBA"), box=(0,0))
            new_im.paste(overlay_image, box=(400,0))
            new_im.save(folder + "overlay/overlay_" + str(i) + ".png")
