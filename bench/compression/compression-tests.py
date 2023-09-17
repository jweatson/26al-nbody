# Quick script to test compression 

import sys
sys.path.append("../")
from al26_nbody import Yields
import zstandard as zstd
import gzip
import lzma
import lz4.frame
import bz2
import zlib
import brotli
import ubjson
import time
import os

def zstd_compress(file,threads=-1,level=8):
  c = zstd.ZstdCompressor(threads=threads,level=level)
  compressed_data = c.compress(file)
  return compressed_data

def decompress(compressed_data):
  c = zstd.ZstdDecompressor()
  decompressed_data = c.decompress(compressed_data)
  return decompressed_data

# Just a quick function to speed some things up
def timeit(process,filename,t1):
  t2 = time.time()
  sz = os.path.getsize(filename)
  print("{}, {:.5f} s, {} bytes".format(process,t2-t1,sz))

# Read in file
yields = Yields("test")
yields.plate("test-yields.ubj.zst")
marinade = {}
for attr, value in yields.__dict__.items():
  marinade[attr] = value

# Begin tests 
t1 = time.time()
with open("testfile","wb") as f:
  ubjson.dump(marinade,f)
timeit("dump_no_compress","testfile",t1)

t1 = time.time()
with open("testfile","wb") as f:
  marinate = zstd_compress(ubjson.dumpb(marinade))
  f.write(marinate)
timeit("dump_zstd","testfile",t1)

t1 = time.time()
with lzma.open("testfile","wb") as f:
  ubjson.dump(marinade,f)
timeit("dump_lzma","testfile",t1)

t1 = time.time()
with gzip.open("testfile","wb") as f:
  ubjson.dump(marinade,f)
timeit("dump_gzip","testfile",t1)

t1 = time.time()
with open("testfile","wb") as f:
  ff = ubjson.dumpb(marinade)
  fr = lz4.frame.compress(ff)
  f.write(fr)
timeit("dump_lz4","testfile",t1)

t1 = time.time()
with bz2.open("testfile","wb") as f:
  ubjson.dump(marinade,f)
timeit("dump_bz2","testfile",t1)

t1 = time.time()
with open("testfile","wb") as f:
  ff = ubjson.dumpb(marinade)
  fr = zlib.compress(ff)
  f.write(fr)
timeit("dump_deflate","testfile",t1)

t1 = time.time()
with open("testfile","wb") as f:
  ff = ubjson.dumpb(marinade)
  fr = brotli.compress(ff)
  f.write(fr)
timeit("dump_brotli","testfile",t1)
