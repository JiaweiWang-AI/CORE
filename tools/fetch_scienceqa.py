"""Download pinned ScienceQA inputs and optional verified MM-CoT captions."""
import argparse
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.net import download_file, file_hash

COMMIT = '2cbf8318e07b9ece895bb2ae605e71e38d623264'
BASE = 'https://raw.githubusercontent.com/lupantech/ScienceQA/' + COMMIT + '/'
PROBLEMS_SHA = '4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7'
HASHES = {'problems.json': PROBLEMS_SHA,
          'pid_splits.json': '85c7820b9f136e572579a01dab2c77a7f8140b59211e421fa5182491d45ee3a7',
          'LICENSE-DATA': 'fae845447443179ecda7a2cdaf6f305e92fe8fc1b7e0c5a8f39a2e4d5de9bfce'}
CAPTION_URL = ('https://raw.githubusercontent.com/amazon-science/mm-cot/'
               '8dd4ac02b94f21347973491f6e6b828502d23f9d/data/instruct_captions.json')
CAPTION_SHA = '6de97c2a8d8fa015ddde3fd23f357f1adeec4ce1909a8a4a0794eecab73791e3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--captions', action='store_true', help='Also fetch the verified MM-CoT InstructBLIP captions.')
    parser.add_argument('--images', nargs='+', choices=['train', 'val', 'test'],
                        help='Also download/extract official image archives; use train test for two-shot test runs.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError('Download datasets outside the CORE folder.')
    out.mkdir(parents=True, exist_ok=True)
    for source, name in [('data/scienceqa/problems.json', 'problems.json'),
                         ('data/scienceqa/pid_splits.json', 'pid_splits.json'),
                         ('LICENSE-DATA', 'LICENSE-DATA')]:
        target = out/name
        download_file(BASE + source, target, HASHES[name])
        print(name, file_hash(target), flush=True)
    if args.captions:
        target = download_file(CAPTION_URL, out/'instruct_captions.json', CAPTION_SHA)
        print(target.name, file_hash(target), flush=True)
    for split in dict.fromkeys(args.images or []):
        archive = download_file('https://scienceqa.s3.us-west-1.amazonaws.com/images/' + split + '.zip',
                                out/'archives'/f'{split}.zip')
        dest = out/'images'
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (dest/member.filename).resolve()
                if dest.resolve() not in target.parents:
                    raise ValueError('Unsafe archive entry: ' + member.filename)
            zipped.extractall(dest)
        print('Extracted', split, 'archive SHA256:', file_hash(archive), flush=True)


if __name__ == '__main__':
    main()
