import sys, subprocess
print('Python executable:', sys.executable)
res = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
print(res.stdout.strip())

pkgs=["fastapi","uvicorn","jinja2","Pillow","python-multipart","mediapipe","numpy","requests","playwright","torch","torchvision","pytorch-lightning","kornia","opencv-contrib-python","scikit-image","scikit-learn","wandb"]
for p in pkgs:
    print('\nChecking', p)
    r = subprocess.run([sys.executable, '-m', 'pip', 'show', p], capture_output=True, text=True)
    if r.returncode!=0 or not r.stdout.strip():
        print('  NOT INSTALLED')
        continue
    loc = None
    ver = None
    for line in r.stdout.splitlines():
        if line.startswith('Location:'):
            loc = line.split(':',1)[1].strip()
        if line.startswith('Version:'):
            ver = line.split(':',1)[1].strip()
    print(f'  Version: {ver or "?"}')
    print(f'  Location: {loc or "?"}')
