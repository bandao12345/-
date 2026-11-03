"""Restore public, checksum-pinned OCR weights. This never reads account credentials."""
import hashlib
import argparse
import json
from pathlib import Path
from urllib.request import urlopen, Request

BASE_DIR=Path(__file__).resolve().parent


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',choices=('v5','v6'),default='v6')
    version=parser.parse_args().version
    model_dir=BASE_DIR/'models'/('ocr-v6' if version=='v6' else 'ocr')
    manifest=json.loads((model_dir/"manifest.json").read_text(encoding="utf-8-sig"))
    for entry in (manifest.get("files") or manifest["models"]):
        filename=entry.get("file") or entry["filename"]
        if Path(filename).name!=filename or not filename.endswith(".onnx"):
            raise ValueError("Invalid model filename")
        target=model_dir/filename
        if target.is_file() and digest(target)==entry["sha256"]:
            print("Verified:",filename)
            continue
        temporary=model_dir/(filename+".download")
        print("Downloading public model:",filename)
        request=Request(entry["url"],headers={"User-Agent":"INKLAB-model-setup/2.0"})
        with urlopen(request,timeout=60) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda:response.read(256*1024),b""):
                stream.write(chunk)
        if digest(temporary)!=entry["sha256"]:
            raise ValueError("Checksum mismatch; model not installed: "+filename)
        temporary.replace(target)
        print("Installed verified model:",filename)
    print("All OCR models are ready.")


if __name__=="__main__":
    main()

