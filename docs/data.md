# Data contract

BiTrackSwin-Seg v1 does not distribute any images, labels, manifests, or derived metadata. Obtain and use the data only under terms that permit your intended use.

## Required layout

```text
data/
├── images/
│   └── <sample_id>.tif
├── labels/
│   └── <sample_id>.tif  (or another supported raster/image format)
└── splits/
    ├── v1_train.txt
    └── v1_val.txt
```

Each manifest contains one relative sample identifier per line, without a filename extension. For example, `tiles/000001` resolves to `images/tiles/000001.tif` and the matching label with a supported extension. The original training entry point supports `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.bmp` for images and labels; GeoTIFF is the tested input format.

## Image contract

- Raster format: readable by Rasterio/GDAL; GeoTIFF is the tested format.
- Shape: five channels in band order `Red`, `Green`, `Blue`, `RedEdge`, `NIR`.
- Tile size: 256 x 256 pixels for the published recipe.
- NoData: any pixel with a band value below `-999.0` is excluded from statistics and treated as ignored during training.

## Label contract

The released task is binary semantic segmentation:

| Stored value | Meaning | Training ID |
| ---: | --- | ---: |
| 0 | background | 0 |
| 1 | plastic-film residue / foreground | 1 |
| 255 | ignore | 255 |

## Fixed split requirement

For a result to be compared to the v1 model-zoo row, it must use the versioned `v1_train.txt` and `v1_val.txt` manifests unchanged. The original epoch-95 experiment selected 24,717 samples from 41,196 source samples and then made an 80/20 split (19,773 / 4,944). Its exact manifests were not retained in the experiment directory.

Consequently, a future reconstructed manifest must carry a new identifier and checksum. It must not be described as the original fixed v1 split unless its membership is recovered from an authoritative record.
