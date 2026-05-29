import subprocess
import sys
import os

def main():
    yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))
    data_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data.yaml'))
    weights = os.path.join(yolov5_dir, 'yolov5s.pt')
    train_script = os.path.join(yolov5_dir, 'train.py')

    cmd = [
        sys.executable, train_script,
        '--cache', 'ram',
        '--img', '640',
        '--batch', '16',
        '--epochs', '100',
        '--data', data_yaml,
        '--cfg', os.path.join(yolov5_dir, 'models', 'yolov5s.yaml'),
        '--weights', weights,
        '--name', 'crowd_yolov5s',
        '--project', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs')),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()