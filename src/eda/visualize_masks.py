import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- RLE decode ---
def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


# --- combine classes ---
def create_mask(df, image_id, shape):
    sub_df = df[df['id'] == image_id]

    mask = np.zeros(shape, dtype=np.uint8)

    for _, row in sub_df.iterrows():
        decoded = rle_decode(row['segmentation'], shape)

        if row['class'] == 'stomach':
            mask[decoded == 1] = 1
        elif row['class'] == 'small_bowel':
            mask[decoded == 1] = 2
        elif row['class'] == 'large_bowel':
            mask[decoded == 1] = 3

    return mask


# --- visualization ---
def show_overlay(image, mask, save_path=None):
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()