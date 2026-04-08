import argparse
import time
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


parser = argparse.ArgumentParser(description='TensorFlow Tiny-ImageNet Training (UVM)')
parser.add_argument('-data', metavar='DIR', help='path to Tiny-ImageNet dataset')
parser.add_argument('--epochs', default=90, type=int, help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--pretrained', action='store_true', help='use ImageNet pre-trained backbone')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
args = parser.parse_args()


# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device_name = f'/GPU:{args.gpu}'

# UVM Config
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.5  # UVM oversubscription
session = InteractiveSession(config=config)

IMG_SIZE = 224
NUM_CLASSES = 200 # tiny imagenet
AUTOTUNE = tf.data.AUTOTUNE



# Tiny-ImageNet class mapping
def load_tiny_imagenet_classes(data_root):
    """
    Tiny ImageNet uses wnids.txt for 200 class ids.
    """
    wnids_path = os.path.join(data_root, 'wnids.txt')
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f"wnids.txt not found: {wnids_path}")

    with open(wnids_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]

    if len(class_names) != NUM_CLASSES:
        print(f"[Warning] wnids count = {len(class_names)} (expected {NUM_CLASSES})")

    class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    return class_names, class_to_index


CLASS_NAMES, CLASS_TO_INDEX = load_tiny_imagenet_classes(args.data)



# Image preprocessing
def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image - mean) / std
    return image


def preprocess_train(image, label):
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = normalize_image(image)
    return image, label


def preprocess_val(image, label):
    image = tf.image.resize(image, [256, 256])
    image = tf.image.central_crop(image, central_fraction=IMG_SIZE / 256.0)
    image = normalize_image(image)
    return image, label


def decode_and_resize(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image.set_shape([None, None, 3])
    return image


# Dataset builders
def build_train_dataset(data_root):
    """
    Tiny ImageNet train structure:
      train/
        n01443537/
          images/
            *.JPEG
    """
    image_paths = []
    labels = []

    train_root = Path(data_root) / 'train'
    if not train_root.exists():
        raise FileNotFoundError(f"train directory not found: {train_root}")

    for class_name in CLASS_NAMES:
        class_dir = train_root / class_name / 'images'
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob('*.JPEG'):
            image_paths.append(str(img_path))
            labels.append(CLASS_TO_INDEX[class_name])

    if len(image_paths) == 0:
        raise RuntimeError("No training images found in Tiny ImageNet train directory.")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load(path, label):
        image = decode_and_resize(path)
        return preprocess_train(image, label)

    ds = ds.shuffle(buffer_size=min(len(image_paths), 10000), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(args.batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_val_dataset_from_class_dirs(val_root):
    """
    If val directory is already reorganized as:
      val/
        class_x/
          *.JPEG
    """
    image_paths = []
    labels = []

    for class_name in CLASS_NAMES:
        class_dir = Path(val_root) / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.rglob('*.JPEG'):
            image_paths.append(str(img_path))
            labels.append(CLASS_TO_INDEX[class_name])

    if len(image_paths) == 0:
        return None

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load(path, label):
        image = decode_and_resize(path)
        return preprocess_val(image, label)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(args.batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_val_dataset_from_annotations(data_root):
    """
    Tiny ImageNet original val structure:
      val/
        images/
          *.JPEG
        val_annotations.txt

    val_annotations.txt format:
      <img_name>\t<class_name>\t<x1>\t<y1>\t<x2>\t<y2>
    """
    val_root = Path(data_root) / 'val'
    ann_file = val_root / 'val_annotations.txt'
    images_dir = val_root / 'images'

    if not ann_file.exists() or not images_dir.exists():
        return None

    image_paths = []
    labels = []

    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            img_name, class_name = parts[0], parts[1]
            if class_name not in CLASS_TO_INDEX:
                continue

            img_path = images_dir / img_name
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(CLASS_TO_INDEX[class_name])

    if len(image_paths) == 0:
        return None

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load(path, label):
        image = decode_and_resize(path)
        return preprocess_val(image, label)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(args.batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_val_dataset(data_root):
    val_root = Path(data_root) / 'val'

    if not val_root.exists():
        raise FileNotFoundError(f"val directory not found: {val_root}")

    # 1) already reorganized class-folder structure
    ds = build_val_dataset_from_class_dirs(val_root)
    if ds is not None:
        print("=> validation dataset loaded from class directories")
        return ds

    # 2) original Tiny ImageNet val/images + val_annotations.txt
    ds = build_val_dataset_from_annotations(data_root)
    if ds is not None:
        print("=> validation dataset loaded from val_annotations.txt")
        return ds

    raise RuntimeError("Could not build validation dataset from Tiny ImageNet val directory.")


train_dataset = build_train_dataset(args.data)
val_dataset = build_val_dataset(args.data)



# ResNet101 model
with tf.device(device_name):
    print(f"Using device: {device_name}")
    if args.pretrained:
        print("=> using pre-trained ResNet101 backbone")
        backbone = tf.keras.applications.ResNet101(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )
        outputs = tf.keras.layers.Dense(NUM_CLASSES)(backbone.output)
        model = tf.keras.Model(inputs=backbone.input, outputs=outputs)
    else:
        print("=> creating ResNet101 from scratch")
        model = tf.keras.applications.ResNet101(
            weights=None,
            include_top=True,
            classes=NUM_CLASSES,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )



# Checkpoint restore
checkpoint = tf.train.Checkpoint(model=model)

if args.resume:
    if os.path.exists(args.resume):
        checkpoint.restore(args.resume)
        print(f"=> loaded checkpoint '{args.resume}'")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")



# Loss / Optimizer
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def get_lr(epoch):
    """Initial LR decayed by 10 every 30 epochs"""
    return args.lr * (0.1 ** (epoch // 30))


optimizer = tf.keras.optimizers.SGD(
    learning_rate=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd
)



# AverageMeter
class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0



# Train / Validate
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = criterion(labels, output)
        loss += tf.add_n(model.losses) if model.losses else 0.0
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, output


@tf.function
def val_step(images, labels):
    output = model(images, training=False)
    loss = criterion(labels, output)
    return loss, output


def accuracy_topk(output, target, topk=(1,)):
    results = []
    for k in topk:
        top_k = tf.math.top_k(output, k=k).indices
        target_expand = tf.expand_dims(target, 1)
        correct = tf.reduce_any(tf.equal(top_k, target_expand), axis=1)
        results.append(tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0)
    return results


def train_one_epoch(epoch):
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    optimizer.learning_rate.assign(get_lr(epoch))

    end = time.time()
    for i, (images, labels) in enumerate(train_dataset):
        loss, output = train_step(images, labels)

        acc1, acc5 = accuracy_topk(output, labels, topk=(1, 5))
        n = images.shape[0] if images.shape[0] is not None else int(tf.shape(images)[0])

        losses.update(float(loss), n)
        top1.update(float(acc1), n)
        top5.update(float(acc5), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"Epoch [{epoch}][{i}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                f"Loss {losses.val:.4e} ({losses.avg:.4e})  "
                f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})  "
                f"Acc@5 {top5.val:.2f} ({top5.avg:.2f})"
            )


def validate(epoch):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for i, (images, labels) in enumerate(val_dataset):
        loss, output = val_step(images, labels)
        acc1, acc5 = accuracy_topk(output, labels, topk=(1, 5))
        n = images.shape[0] if images.shape[0] is not None else int(tf.shape(images)[0])

        losses.update(float(loss), n)
        top1.update(float(acc1), n)
        top5.update(float(acc5), n)

    print(
        f"[Val Epoch {epoch}] "
        f"Loss {losses.avg:.4e}  "
        f"Acc@1 {top1.avg:.3f}  "
        f"Acc@5 {top5.avg:.3f}"
    )
    return top1.avg



# Main loop
best_acc1 = 0.0
init_tickcount = time.perf_counter()

for epoch in range(args.start_epoch, args.epochs):
    train_one_epoch(epoch)

    # validation 필요 없으면 아래 두 줄 중 validate 주석 처리 가능
    acc1 = validate(epoch)

    if acc1 > best_acc1:
        best_acc1 = acc1
        checkpoint.write(f'model_best_epoch{epoch}')
        print(f"=> saved best model (epoch {epoch}, acc@1 {best_acc1:.3f})")

    checkpoint.write(f'checkpoint_epoch{epoch}')

print("Training done.")

elapsed_us = (time.perf_counter() - init_tickcount) * 1_000_000
print(f"elapsed_time(us): {int(elapsed_us)}")

session.close()