"""Explicit download/verification of the official frozen ResNet18 weights."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

SHA256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def ensure(path,download=False):
    path=Path(path)
    if path.exists():
        if digest(path)!=SHA256:raise ValueError('Existing weights have wrong SHA256; file not overwritten')
        return path.resolve()
    if not download:raise FileNotFoundError('Weights missing; pass --download to fetch the official asset explicitly')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='resnet18-',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as output,urllib.request.urlopen(URL,timeout=60) as source:
            while True:
                block=source.read(1024*1024)
                if not block:break
                output.write(block)
        if digest(name)!=SHA256:raise ValueError('Downloaded weights SHA256 mismatch')
        # Same-directory hard link publishes a complete file atomically and never overwrites.
        os.link(name,path)
        return path.resolve()
    finally:Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',default='models/local/resnet18-f37072fd.pth')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    try:print(str(ensure(args.path,args.download))+' SHA256='+SHA256)
    except (OSError,ValueError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__=='__main__':main()
