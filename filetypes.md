# Filetypes

## `State` contents

The `state` class contains the following:

- The `cluster` object in its entirety, which contains the current simulation properties for all codes.
  - This also contains the `particles` (`state.cluster.particles`) subclass, which contains the properties and positions of all stars in the simulation
- `yields`, a table containing the time dependent yields of each isotope in each system
  - `yields.local`: the local model wind yields (assumes a 1 parsec range limit)
  - `yields.global`: the global model wind yields (unlimited range, inverse square falloff)
- The `metadata` class, which includes the initialisation arguments, time of initialisation, and various simulation settings, to aid in replication.
  - `metadata` also includes:
    - `simtime`, the total time elapsed for the simulation

## Compression
Data is compressed using the `lzma` library compression scheme, which offers a good mix of compression and performance, whilst this is a non-standard package, LZMA derivative schema are significantly more efficient than DEFLATE derived schema such as `zip`. Stream compression is used, which rules out frame compression methods such as `lz4`, `brotli` and `zstd`

### Comparisons with other compression algorithms

#### Compression `state` files

A typical 10,000 star system is initialised and saved with different compression algorithms, the time is recorded. Data is saved to an NVME SSD with a sequential write speed of `~3.2GiB/s`, and repeated 3 times.

| Schema         | Time elapsed (s) | Slowdown | Filesize  | Compression Ratio |
|----------------|------------------|----------|-----------|-------------------|
| No compression | `0.167`          | `1.00`   | `4.1 MiB` | `1:1`             |
| `gzip`         | `0.319`          | `1.91`   | `1.2 MiB` | `3.42:1`          |
| `bz2`          | `0.604`          | `3.62`   | `1.2 MiB` | `3.42:1`          |
| `lzma`         | `0.771`          | `4.62`   | `1.1 MiB` | `3.72:1`          |

#### Compression of `yields` file

A 1,000 star system is initialised and run to 10Myr with ~1,200 subsequent checkpoints, the yields file is read to memory, converted to a `Yields` object, re-serialised and written to disk with varying compression methods. Data is saved to an NVME SSD with a sequential write speed of `~3.2GiB/s`, and repeated 3 times.

| Method    | Time      | Final Size  | Compression Ratio | Throughput     |
|-----------|-----------|-------------|-------------------|----------------|
| `None`    | `0.14 s`  | `34.41 MiB` | `1.00:1`          | `238.68 MiB/s` |
| `LZ4`     | `0.13 s`  | `11.28 MiB` | `3.05:1`          | `270.77 MiB/s` |
| `ZSTD`    | `0.19 s`  | `7.42 MiB`  | `4.64:1`          | `176.64 MiB/s` |
| `DEFLATE` | `0.98 s`  | `8.43 MiB`  | `4.08:1`          | `35.00 MiB/s`  |
| `GZIP`    | `1.38 s`  | `8.37 MiB`  | `4.11:1`          | `24.98 MiB/s`  |
| `BZIP2`   | `4.47 s`  | `8.36 MiB`  | `4.12:1`          | `7.69 MiB/s`   |
| `LZMA`    | `4.96 s`  | `6.58 MiB`  | `5.23:1`          | `6.93 MiB/s`   |
| `Brotli`  | `33.43 s` | `6.47 MiB`  | `5.32:1`          | `1.03 MiB/s`   |

As can be seen the best performing algorithm in terms of compression ratio and compression throughput is `LZMA`, however as the python implementation is block rather than stream based, there is a small memory overhead as data must be compressed first before being written to disk. This is fairly minimal, and should not consitute a greater overhead than serialising the `Yields` object anyway.