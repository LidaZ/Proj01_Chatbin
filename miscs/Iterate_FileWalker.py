from pathlib import Path



mydir = Path("path/to/my/dir")
for file in mydir.glob('**/*.mp4'):
    print(file.name)
    # do your stuff