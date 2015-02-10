import os
import urllib


target_dir = "data"
target_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

if not os.path.exists(target_path):
    print "Downloading..."
    urllib.urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "data/cifar-10-python.tar.gz")

print "Extracting..."
os.system("tar xzvf data/cifar-10-python.tar.gz -C data")

print "done."