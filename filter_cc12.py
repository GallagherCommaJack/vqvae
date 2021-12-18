from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

img_root_dir = Path("/data1/DALLE-datasets/general/cc12")


def try_open_image(ix):
    fname = img_root_dir / "images" / f"{ix}.jpg"
    try:
        img = Image.open(fname)
        if img.mode == "RGB":
            img.convert("L")
        else:
            img.convert("RGB")
        return ix
    except:
        return None


print("reading ixs")

with open(img_root_dir / "ixs.txt", "r") as f:
    ixs = [line[:-1] for line in tqdm(f)]

ixs_out = []

print("starting filter processes")
pool = Pool(processes=128)

print("filtering images")


skipped = 0
bar = tqdm(ixs)  # , mininterval=1.0, miniters=50_000)
# bar.set_description(f"skipped: {skipped}")
for ix in pool.imap(try_open_image, bar):
    if ix is None:
        skipped += 1
    else:
        ixs_out.append(ix)

    bar.set_description(
        f"accepted: {len(ixs_out):06}, skipped: {skipped:06}", refresh=False
    )

print(f"done filtering, dropped {skipped}")
print("writing file")

with open("ixs_filtered.txt", "w") as f:
    for ix in tqdm(ixs_out):
        print(ix, file=f)

