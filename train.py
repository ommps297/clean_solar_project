#!/usr/bin/env python3
"""
train.py - Clean training script for Solar Panels Dust Detection (transfer learning)

Expected dataset layout (place your images here):
data/
  train/
    clean/
    dusty/
  val/
    clean/
    dusty/

If you don't have a validation set, set validation_split in the ImageDataGenerator call.

How to run (after creating conda env and installing requirements):
python train.py --epochs 10 --batch_size 32 --img_size 224 --save_model models/mobilenetv2.h5

The script will:
- load images with ImageDataGenerator
- build a MobileNetV2-based classifier (transfer learning)
- train and save the model and training history (history.json)
- plot training curves to training_plot.png
"""

import os
import argparse
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

def build_model(input_shape=(224,224,3), lr=1e-4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze feature extractor initially
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=preds)
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, out_path):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['accuracy'], label='train_acc')
    plt.plot(history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val') if args.use_val else None

    if not os.path.isdir(train_dir):
        raise SystemExit(f"Training directory not found: {train_dir}\\nPlease prepare your dataset as described in the README inside the project folder.")

    # Image generators
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,
                                       validation_split=0.15 if not args.use_val else 0.0)
    val_datagen = ImageDataGenerator(rescale=1./255)

    target_size = (args.img_size, args.img_size)
    if args.use_val:
        train_gen = train_datagen.flow_from_directory(train_dir, target_size=target_size,
                                                      batch_size=args.batch_size, class_mode='binary', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_dir, target_size=target_size,
                                                  batch_size=args.batch_size, class_mode='binary', shuffle=False)
    else:
        train_gen = train_datagen.flow_from_directory(train_dir, target_size=target_size,
                                                      batch_size=args.batch_size, class_mode='binary', subset='training', shuffle=True)
        val_gen = train_datagen.flow_from_directory(train_dir, target_size=target_size,
                                                    batch_size=args.batch_size, class_mode='binary', subset='validation', shuffle=False)

    model = build_model(input_shape=(args.img_size, args.img_size, 3), lr=args.lr)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    callbacks = [
        ModelCheckpoint(args.save_model, save_best_only=True, monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]

    history = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks)
    # Save history
    hist = {k: [float(x) for x in v] for k,v in history.history.items()}
    with open('history.json', 'w') as f:
        json.dump(hist, f)

    plot_history(hist, 'training_plot.png')
    print("Training complete. Model saved to", args.save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Root data folder containing train/ (and optionally val/)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_model', type=str, default='models/mobilenetv2.h5')
    parser.add_argument('--use_val', action='store_true', help='If true, use data/val as validation directory. Otherwise split train into train/val.')
    args = parser.parse_args()
    main(args)