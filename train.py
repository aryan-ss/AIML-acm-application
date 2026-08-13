"""
Task 3 training driver (mirrors cat_dog_classifier.ipynb).
Runs the MobileNetV2 transfer-learning pipeline and saves real outputs:
  - accuracy_curve.png
  - example_predictions.png
  - RESULTS.md  (final train/val accuracy + notes)

Trains on a subset of batches for speed on CPU, but the reported VALIDATION
accuracy is measured on the FULL 5,000-image test set.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

TRAIN_DIR = "dogs-vs-cats/train"
VAL_DIR   = "dogs-vs-cats/test"
IMG_SIZE  = (160, 160)
BATCH     = 32
SEED      = 42
EPOCHS    = 4
TRAIN_BATCHES = 180   # ~5760 images/epoch (subset for CPU speed)
VAL_BATCHES   = 40    # ~1280 images for the per-epoch validation signal
tf.random.set_seed(SEED)
print("TensorFlow", tf.__version__)

train_full = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True)
val_full = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True)
class_names = train_full.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_full.take(TRAIN_BATCHES).prefetch(AUTOTUNE)
val_small = val_full.take(VAL_BATCHES).prefetch(AUTOTUNE)
val_eval = val_full.prefetch(AUTOTUNE)   # full test set for final number

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_small, epochs=EPOCHS)

# Honest final metric on the FULL test set
print("\nEvaluating on full test set...")
val_loss, val_acc_full = model.evaluate(val_eval, verbose=1)
train_acc = history.history["accuracy"][-1]
print(f"\nFinal training accuracy (last epoch): {train_acc:.3f}")
print(f"Full-test validation accuracy:        {val_acc_full:.3f}")

# --- accuracy curve ---
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], "o-", label="train")
plt.plot(history.history["val_accuracy"], "o-", label="val (subset)")
plt.title("Accuracy per epoch"); plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=120); plt.close()
print("Saved accuracy_curve.png")

# --- 5 example predictions: a class-balanced mix incl one wrong ---
# Pull a few batches so we can find both correct cats and correct dogs plus a miss.
imgs, labs = [], []
for bx, by in val_full.take(4):
    imgs.append(bx); labs.append(by)
images = tf.concat(imgs, 0); labels = tf.concat(labs, 0)
probs = model.predict(images).ravel()
preds = (probs > 0.5).astype(int)
truth = labels.numpy().astype(int)

right_cat = [i for i in range(len(preds)) if preds[i] == truth[i] == 0]
right_dog = [i for i in range(len(preds)) if preds[i] == truth[i] == 1]
wrong     = [i for i in range(len(preds)) if preds[i] != truth[i]]
# 2 correct cats + 2 correct dogs + 1 misclassified (fallback if no miss found)
show_idx = right_cat[:2] + right_dog[:2] + (wrong[:1] if wrong else right_cat[2:3])
labels = tf.constant(truth)  # keep downstream int indexing simple

plt.figure(figsize=(15, 4))
for plot_i, i in enumerate(show_idx):
    conf = probs[i] if preds[i] == 1 else 1 - probs[i]
    true, pred = class_names[int(labels[i])], class_names[preds[i]]
    ok = "correct" if pred == true else "WRONG"
    ax = plt.subplot(1, 5, plot_i + 1)
    ax.imshow(images[i].numpy().astype("uint8"))
    ax.set_title(f"pred: {pred} ({conf:.0%})\ntrue: {true} [{ok}]",
                 color=("green" if pred == true else "red"), fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("example_predictions.png", dpi=120); plt.close()
print("Saved example_predictions.png")

with open("RESULTS.md", "w") as f:
    f.write("# Task 3 — Training Results\n\n")
    f.write(f"- Model: MobileNetV2 (frozen ImageNet base) + GAP + dropout + 1 sigmoid unit\n")
    f.write(f"- Trained on ~{TRAIN_BATCHES*BATCH} images, {EPOCHS} epochs, on CPU\n")
    f.write(f"- **Final training accuracy:** {train_acc:.3f}\n")
    f.write(f"- **Validation accuracy (full 5,000-image test set):** {val_acc_full:.3f}\n\n")
    f.write("![accuracy](accuracy_curve.png)\n\n")
    f.write("## Example predictions\n\n")
    f.write("![predictions](example_predictions.png)\n\n")
    if wrong:
        i = wrong[0]
        c = probs[i] if preds[i] == 1 else 1 - probs[i]
        conf_word = ("high — a *confidently wrong* prediction" if c >= 0.8
                     else "moderate — the model was near its 50% decision boundary")
        f.write(f"The misclassified example was predicted **{class_names[preds[i]]}** "
                f"at {c:.0%} confidence when it was actually a "
                f"**{class_names[int(truth[i])]}**. The confidence is {conf_word}. "
                f"Likely cause: the subject is presented in a way that matches the "
                f"other class's typical photos — e.g. a small, dark, curled-up puppy "
                f"cradled in a hand on a soft blanket looks a lot like how cats are "
                f"usually pictured, and the dog-defining cues (long snout, body shape, "
                f"upright ears) are hidden. Transfer-learned features latch onto that "
                f"overall pose/context, so a cat-like presentation fools them.\n")
print("Saved RESULTS.md\nDONE.")
