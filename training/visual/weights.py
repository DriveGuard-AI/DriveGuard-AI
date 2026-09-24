"""Explicit download/verification of the official frozen ResNet18 weights."""
import argparse
import hashlib
import http.client
import math
import os
from pathlib import Path
import tempfile
import sys
import time
import urllib.error
import urllib.request

SHA256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def ensure(path,download=False,*,timeout=300,attempts=3,retry_delay=2):
    if (not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0
            or type(attempts)!=int or attempts<1
            or not isinstance(retry_delay,(int,float)) or not math.isfinite(retry_delay) or retry_delay<0):
        raise ValueError('timeout/attempts must be positive and retry_delay nonnegative')
    path=Path(path)
    if path.exists():
        if digest(path)!=SHA256:raise ValueError('Existing weights have wrong SHA256; file not overwritten')
        return path.resolve()
    if not download:raise FileNotFoundError('Weights missing; pass --download to fetch the official asset explicitly')
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,attempts+1):
        print(f'Downloading ResNet18 weights: attempt {attempt}/{attempts}, socket timeout {timeout}s',file=sys.stderr,flush=True)
        fd,name=tempfile.mkstemp(prefix='resnet18-',suffix='.partial',dir=path.parent)
        try:
            try:
                with os.fdopen(fd,'wb') as output,urllib.request.urlopen(URL,timeout=timeout) as source:
                    received=0;next_report=8*1024*1024
                    while True:
                        block=source.read(1024*1024)
                        if not block:break
                        output.write(block);received+=len(block)
                        if received>=next_report:
                            print(f'Downloaded {received/(1024*1024):.1f} MiB',file=sys.stderr,flush=True)
                            next_report+=8*1024*1024
            except (TimeoutError,ConnectionError,urllib.error.URLError,http.client.IncompleteRead) as exc:
                if attempt==attempts:
                    raise OSError(f'Weight download failed after {attempts} attempts: {exc}. '
                                  f'Retry later or copy the official weights offline to {path}; SHA256={SHA256}') from exc
                print(f'Network download failed: {exc}; retrying from the beginning.',file=sys.stderr,flush=True)
                time.sleep(min(retry_delay,60))
                continue
            if digest(name)!=SHA256:raise ValueError('Downloaded weights SHA256 mismatch')
            # Same-directory hard link publishes a complete file atomically and never overwrites.
            os.link(name,path)
            return path.resolve()
        finally:Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',default='models/local/resnet18-f37072fd.pth')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--timeout',type=float,default=300,help='Socket timeout in seconds (not total download duration)')
    parser.add_argument('--attempts',type=int,default=3,help='Total download attempts; network failures restart download')
    args=parser.parse_args(argv)
    try:print(str(ensure(args.path,args.download,timeout=args.timeout,attempts=args.attempts))+' SHA256='+SHA256)
    except (OSError,ValueError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__=='__main__':main()
