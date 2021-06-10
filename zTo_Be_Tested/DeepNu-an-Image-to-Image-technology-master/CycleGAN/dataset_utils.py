import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import glob
import os
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE

predefined_cyclegan_task_name_list = ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo",
                                      "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps",
                                      "cityscapes", "facades", "iphone2dslr_flower", ]


def load_cyclegan_image_dataset_by_task_name(task_name):
    """Data from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    View sample images here, https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology/tree/master/CycleGAN
    Processing code to view here, https://www.tensorflow.org/datasets/datasets#cycle_gan"""
    cycle_gan_dataset_name_list = ["cycle_gan/apple2orange", "cycle_gan/summer2winter_yosemite",
                                   "cycle_gan/horse2zebra", "cycle_gan/monet2photo",
                                   "cycle_gan/cezanne2photo", "cycle_gan/ukiyoe2photo",
                                   "cycle_gan/vangogh2photo", "cycle_gan/maps",
                                   "cycle_gan/cityscapes", "cycle_gan/facades",
                                   "cycle_gan/iphone2dslr_flower", ]

    task_name = "cycle_gan/" + task_name
    if task_name not in cycle_gan_dataset_name_list:
        print("Not include this task!")
        print(f"You can choose task from {cycle_gan_dataset_name_list}")
        raise ValueError("not include this task!")

    # download data
    dataset, metadata = tfds.load(task_name, with_info=True, as_supervised=True)

    trainA_dataset, trainB_dataset = dataset['trainA'], dataset['trainB']
    testA_dataset, testB_dataset = dataset['testA'], dataset['testB']

    return trainA_dataset, trainB_dataset, testA_dataset, testB_dataset


def load_cyclegan_image_dataset_from_data_folder(data_dir):
    """There is a need for a data folder, the data file contains four subfolders
     trainA, trainB, testA, testB. The four subfolders respectively store the
     source image set used for training, the target image set used for training,
     the source image set used for the test, and the target image set used for the test."""

    def get_image_path(data_dir, image_type):
        image_data_dir = os.path.join(data_dir, image_type)
        filenames = glob.glob(os.path.join(image_data_dir, "*.jpg"))
        return filenames

    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img, "Z"

    trainA_image_path = get_image_path(data_dir, "trainA")
    trainB_image_path = get_image_path(data_dir, "trainB")
    testA_image_path = get_image_path(data_dir, "testA")
    testB_image_path = get_image_path(data_dir, "testB")

    print(f"trainA_image_path numbers: {len(trainA_image_path)}")
    print(f"trainB_image_path numbers: {len(trainB_image_path)}")
    print(f"testA_image_path numbers: {len(testA_image_path)}")
    print(f"testB_image_path numbers: {len(testB_image_path)}")

    trainA_image_path_dataset = tf.data.Dataset.from_tensor_slices(trainA_image_path)
    trainA_dataset = trainA_image_path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    trainB_image_path_dataset = tf.data.Dataset.from_tensor_slices(trainB_image_path)
    trainB_dataset = trainB_image_path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    testA_image_path_dataset = tf.data.Dataset.from_tensor_slices(testA_image_path)
    testA_dataset = testA_image_path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    testB_image_path_dataset = tf.data.Dataset.from_tensor_slices(testB_image_path)
    testB_dataset = testB_image_path_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return trainA_dataset, trainB_dataset, testA_dataset, testB_dataset


def download_and_processing_cyclegan_dataset(data_dir_or_predefined_task_name=None,
                                             BATCH_SIZE=1, BUFFER_SIZE=1000,
                                             IMG_WIDTH=256, IMG_HEIGHT=256):
    """
    :param data_dir: Folder paths that provide your own data, check load_cyclegan_image_dataset_from_data_folder function.
    :param task_name: For tasks with processed data, you can check cycle_gan_dataset_name_list,
     or go to https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology/tree/master/CycleGAN for details.
    :return: trainA_dataset, trainB_dataset, testA_dataset, testB_dataset
    """

    def random_crop(image):
        cropped_image = tf.image.random_crop(
            image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
        return cropped_image

    # normalizing the images to [-1, 1]
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def random_jitter(image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)
        # random mirroring
        image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)
        return image

    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    # prepare data, task_name and data_dir only need to provide one of them
    if data_dir_or_predefined_task_name in predefined_cyclegan_task_name_list:
        trainA_dataset, trainB_dataset, testA_dataset, testB_dataset = load_cyclegan_image_dataset_by_task_name(
            data_dir_or_predefined_task_name)
        print("prepare data from task_name")
    elif os.path.exists(data_dir_or_predefined_task_name):
        trainA_dataset, trainB_dataset, testA_dataset, testB_dataset = load_cyclegan_image_dataset_from_data_folder(
            data_dir_or_predefined_task_name)
        print("prepare data from data_dir")
    else:
        raise ValueError("Task_name error and data_dir does not exist!")

    # processing data
    trainA_dataset = trainA_dataset.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    trainB_dataset = trainB_dataset.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    testA_dataset = testA_dataset.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    testB_dataset = testB_dataset.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    return trainA_dataset, trainB_dataset, testA_dataset, testB_dataset


def show_dataset(sampleA, sampleB, numder=0, store_sample_image_path="sample_image",
                 sampleA_name="sampleA", sampleB_name="sampleB"):
    if not os.path.exists(store_sample_image_path):
        os.mkdir(store_sample_image_path)
    plt.title(sampleA_name)
    plt.imshow(sampleA[0] * 0.5 + 0.5)
    save_path = os.path.join(store_sample_image_path, f"{str(numder)}_{sampleA_name}.png")
    plt.savefig(save_path)

    plt.title(sampleB_name)
    plt.imshow(sampleB[0] * 0.5 + 0.5)
    save_path = os.path.join(store_sample_image_path, f"{str(numder)}_{sampleB_name}.png")
    plt.savefig(save_path)


def download_all_predefined_tasks_data():
    for task_name in predefined_cyclegan_task_name_list:
        BATCH_SIZE = 1
        trainA_dataset, trainB_dataset, testA_dataset, testB_dataset = download_and_processing_cyclegan_dataset(
            task_name, BATCH_SIZE)
        store_sample_image_path = f"sample_image_{task_name}"
        i = 0
        for sampleA, sampleB in zip(trainA_dataset.take(3), trainB_dataset.take(3)):
            show_dataset(sampleA, sampleB, numder=i, sampleA_name="A", sampleB_name="B",
                         store_sample_image_path=store_sample_image_path)
            i += 1


def check_one_dataset_info(data_dir_or_predefined_task_name, store_sample_image_path='test'):
    trainA_dataset, trainB_dataset, _, _ = download_and_processing_cyclegan_dataset(
        data_dir_or_predefined_task_name, BATCH_SIZE=10)
    i = 0
    for sampleA, sampleB in zip(trainA_dataset.take(3), trainB_dataset.take(3)):
        show_dataset(sampleA, sampleB, numder=i, store_sample_image_path=store_sample_image_path)
        i += 1


if __name__ == "__main__":
    print("You can choose a task_name from predefined_cyclegan_task_name_list!")
    print(predefined_cyclegan_task_name_list)
    # task_name and data_dir only need to provide one of them
    #data_dir_or_predefined_task_name = "/home/b418a/.keras/datasets/apple2orange"
    data_dir_or_predefined_task_name = "apple2orange"
    store_sample_image_path = "test_path_data"

    if len(sys.argv) == 2:
        data_dir_or_predefined_task_name = sys.argv[1]
    print(f"You choose data_dir_or_predefined_task_name is {data_dir_or_predefined_task_name}")
    check_one_dataset_info(data_dir_or_predefined_task_name=data_dir_or_predefined_task_name, store_sample_image_path=store_sample_image_path)
