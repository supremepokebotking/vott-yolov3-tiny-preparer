import glob, os
import random
import json
from PIL import Image

EXPORT_PATH = os.environ.get("EXPORT_PATH", "./vott-csv-export")
USE_GUARANTEED_IMAGES = bool(int(os.environ.get('USE_GUARANTEED_IMAGES', 1)))
SHOW_DEBUG_TEXT = bool(int(os.environ.get('SHOW_DEBUG_TEXT', 0)))

import csv
import math
vott_data = []

all_csvs = glob.glob("%s/*.csv" % EXPORT_PATH)
export_csv = all_csvs[0]
for file in all_csvs:
    print(file)

guaranteed_testing_file = None
all_guaranteed_testing_files = glob.glob("%s/*testing-export.txt" % EXPORT_PATH)

guaranteed_training_file = None
all_guaranteed_training_files = glob.glob("%s/*training-export.txt" % EXPORT_PATH)

if len(all_guaranteed_testing_files) > 0:
    guaranteed_testing_file = all_guaranteed_testing_files[0]

if len(all_guaranteed_training_files) > 0:
    guaranteed_training_file = all_guaranteed_training_files[0]

guaranteed_testing_file_names = []
guaranteed_training_file_names = []


for export_csv in all_csvs:
    print('inspecting csv:', export_csv)
    #with open('PokeTest1-export.csv') as csv_file:
    with open(export_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            if len(row) < 7:
                # if image dimensions dont exist, manually add them
                image = Image.open("%s/%s" % (EXPORT_PATH, row[0]))

                width, height = image.size

                row.append(width)
                row.append(height)

            vott_data.append(row)

if all_guaranteed_testing_files is not None:
    for guaranteed_testing_file in all_guaranteed_testing_files:
        print('inspecting testing csv:', guaranteed_testing_file)
        with open(guaranteed_testing_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                guaranteed_testing_file_names.append(row[0])

if all_guaranteed_training_files is not None:
    for guaranteed_training_file in all_guaranteed_training_files:
        print('inspecting training csv:', guaranteed_training_file)
        with open(guaranteed_training_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                guaranteed_training_file_names.append(row[0])

print("guaranteed_testing_file_names", guaranteed_testing_file_names)
print("guaranteed_training_file_names", guaranteed_training_file_names)

base_max_width = 10000.0
base_max_height = 10000.0

def convert_labels(path, x1, y1, x2, y2, asset_width, asset_height):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    size = (asset_height, asset_width)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
full_codenames = []
full_codenames_2 = []
candy = []
for data in vott_data:
    asset_width = base_max_width
    asset_height = base_max_height
    filename = data[0]

    if len(data) > 7:
        asset_width = float(data[6])
        asset_height = float(data[7])

    xmin = float(max(0, float(data[1])))
    ymin = float(max(0, float(data[2])))
    xmax = float(min(float(data[3]), asset_width))
    ymax = float(min(float(data[4]), asset_height))

    codename = data[5]
    if codename not in full_codenames:
        full_codenames.append(codename)
    code = full_codenames.index(codename)
    if codename not in candy:
        candy.append(codename)
        full_codenames_2.append((code, codename))
#    continue

    filename_noext, ext = os.path.splitext(filename)
    filename_noslash = filename_noext.split('/')[-1]
    textfile_out = '%s.txt' % (filename_noext)


    xCenter = (xmin + xmax)/2 / asset_width
    yCenter = (ymin + ymax)/2 / asset_height
    width = (xmax - xmin) / asset_width
    height = (ymax - ymin) / asset_height

    (x,y,w,h) = convert_labels(None, xmin, ymin, xmax, ymax, asset_width, asset_height)

    output = '%d %.6f %.6f %.6f %.6f' % (code, xCenter, yCenter, width, height)
#    print(textfile_out)
    output2 = '%d %.6f %.6f %.6f %.6f' % (code, x, y, w, h)
    if not math.isclose(xCenter, x) or not math.isclose(yCenter, y) or not math.isclose(width, w) or not math.isclose(height, h):
        print(output)
        print(output2)
        print()
#    continue
    file_save_path = "%s/%s" % (EXPORT_PATH, textfile_out)
    if SHOW_DEBUG_TEXT:
        print(textfile_out, ': ', output)
    f=open(file_save_path, "a+")
    f.write("%s\n" % (output))
    f.close()

print(full_codenames)
for name in full_codenames:
    print(name)

print('sorted codenames')
full_codenames_2.sort(key=lambda name: name[0], reverse=False)
print(full_codenames_2)
for name in full_codenames_2:
    print(name[1])

############### Config Writing ######################

NUM_CLASSES = len(full_codenames)
FILTERS = 3*(NUM_CLASSES+5)
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "5000"))

obj_save_path = "%s/%s" % (EXPORT_PATH, "obj.names")
with open(obj_save_path, "w") as obj_file:
    for name in full_codenames:
        obj_file.write("%s\n" % (name))

obj_save_path = "%s/%s" % (EXPORT_PATH, "obj.data")
with open(obj_save_path, "w") as obj_file:
    obj_file.write("classes= %d\n"% NUM_CLASSES)
    obj_file.write("train  = train.txt\n")
    obj_file.write("valid  = test.txt\n")
    obj_file.write("names = obj.names\n")
    obj_file.write("backup = backup\n")

original_cfg_lines = []
with open('yolov3-tiny.cfg') as original_cfg_file:
    original_cfg_lines = original_cfg_file.readlines()


tiny_cfg_save_path = "%s/%s" % (EXPORT_PATH, "yolov3-tiny.cfg")
with open(tiny_cfg_save_path, "w") as tiny_cfg_file:
    for line in original_cfg_lines:
        line = line.replace('classes=80', ('classes=%d'%NUM_CLASSES))
        line = line.replace('filters=255', ('filters=%d'%FILTERS))
        line = line.replace('max_batches = 500200', ('max_batches = %d'%MAX_BATCHES))
        tiny_cfg_file.write(line)


########################TEST TRAIN SPLIT ###########

# Percentage of images to be used for the test set
percentage_test = int(os.environ.get('TEST_PERCENTAGE', 10));

# Create and/or truncate train.txt and test.txt
file_train = open('%s/train.txt'%EXPORT_PATH, 'w')
file_test = open('%s/test.txt'%EXPORT_PATH, 'w')

# Populate train.txt and test.txt
counter = 1

index_test = round(100 // percentage_test)
#print(glob.iglob(os.path.join(EXPORT_PATH, "*.jpg")))
extensions = ("*.png","*.jpg","*.jpeg",)
all_files = []
for extension in extensions:
    all_files.extend(glob.iglob(os.path.join(EXPORT_PATH,extension)))

print(all_files[:10])
random.shuffle(all_files)
print()
print(all_files[:10])
test_images = 0
train_images = 0

logged_test_images = []

#for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
for pathAndFilename in all_files:

    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if USE_GUARANTEED_IMAGES:
        title_and_ext = '%s%s' % (title, ext)
        title_and_ext = title_and_ext.strip()
#        print('aaa:',title_and_ext)
        if title_and_ext in guaranteed_testing_file_names or 'testing_' in title_and_ext:
            counter = 1
            file_test.write(title + ext + "\n")
            test_images += 1
            logged_test_images.append(title)
            print('used test image')
            continue

        if title_and_ext in guaranteed_training_file_names or 'training_' in title_and_ext:
            file_train.write(title + ext + "\n")
            train_images += 1
            print('used training image')
            continue

    if counter == index_test:
        counter = 1
        file_test.write(title + ext + "\n")
        test_images += 1
        logged_test_images.append(title)
    else:
        file_train.write(title + ext + "\n")
        train_images += 1
        counter = counter + 1
file_test.close()
file_train.close()
print('Number of training images: %d' % train_images)
print('Number of testing images: %d' % test_images)
print('All images used for testing:\n%s' % json.dumps(logged_test_images))
