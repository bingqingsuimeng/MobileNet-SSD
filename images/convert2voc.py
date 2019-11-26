import cv2
import shutil
import os
import random

from xml.dom.minidom import Document


def writexml(filename, saveimg, bboxes, xmlpath):
    """
    write to xml style of VOC dataset
    :param filename: xml filename
    :param saveimg: the image data with shape [H, W, C]
    :param bboxes: bounding boxes 
    :param xmlpath: xml file save path
    :return: None
    """

    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    flikerid = doc.createElement('flikerid')
    flikerid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flikerid)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinhom'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

    f = open(xmlpath, 'w')
    f.write(doc.toprettyxml(indent=' '))
    f.close()


# wider face dataset folder path
#rootdir = "E:/dataset/wider_face"
rootdir = "/mnt/learn/MobileNet-SSD/images"

def convertimgset(img_set):
    #遍历数据集内的文件名字
    imgdir = rootdir + "/" + img_set
    files = os.listdir(imgdir)

    #创建生成标签的VOC格式的文件夹内的文件
    voc_imageset_main_path = rootdir + "/ImageSets/Main"
    fwrite = open(voc_imageset_main_path + '/' + img_set + ".txt", 'w')

    gtfilepath = rootdir + "/label"
    for filename in files:
        lable_name_new = "new_" + os.path.splitext(filename)[0] + ".xml"
        lable_name = os.path.splitext(filename)[0] + ".xml"
        lable_f_new = gtfilepath + "/" + lable_name_new
        #lable_f = gtfilepath + "/" + lable_name

        imgpath = imgdir + "/" + filename
        img = cv2.imread(imgpath)
        cv2.imwrite("{}/JPEGImages/{}".format(rootdir, filename), img)
        fwrite.write(filename.split('.')[0] + '\n')

        shutil.copyfile(lable_f_new, rootdir + "/Annotations/" + lable_name)

    # index = 0
    #
    # gtfilepath = rootdir + "/label"
    # with open(gtfilepath, 'r') as gtfiles:
    #     while index < 3200:  # True
    #         filename = gtfiles.readline()[:-1]
    #         if filename == "":
    #             continue
    #         imgpath = imgdir + "/" + filename
    #         # print(imgpath)
    #         img = cv2.imread(imgpath)
    #
    #         if not img.data:
    #             break
    #         numbbox = int(gtfiles.readline())
    #
    #         bboxes = []
    #         for i in range(numbbox):
    #             line = gtfiles.readline()
    #             lines = line.split()
    #             lines = lines[0: 4]
    #             bbox = (int(lines[0]), int(lines[1]), int(lines[0]) + int(lines[2]), int(lines[1]) + int(lines[3]))
    #             bboxes.append(bbox)
    #
    #         filename = filename.replace("/", "_")
    #
    #         if len(bboxes) == 0:
    #             print("no face")
    #             continue
    #
    #         cv2.imwrite("{}/JPEGImages/{}".format(rootdir, filename), img)
    #         fwrite.write(filename.split('.')[0] + '\n')
    #
    #         xmlpath = '{}/Annotations/{}.xml'.format(rootdir, filename.split('.')[0])
    #         writexml(filename, img, bboxes, xmlpath)
    #         if index % 100 == 0:
    #             print("success NO." + str(index))
    #         index += 1
    # print(img_set + " total: " + str(index))
    fwrite.close()

def divide_dataset(traindir, valdir, scale):
    print("divide dataset %s with scale %f" % (traindir, scale))
    if not os.path.exists(valdir):
        os.mkdir(valdir)
    files = os.listdir(traindir)
    random.shuffle(files)
    datalen = int(len(files) * 0.8)
    for f in files[datalen:]:
        shutil.move(traindir + '/' + f, valdir + '/')

def create_voc_style_folders():
    #创建生成标签的VOC格式的文件夹ImageSets
    print("create folder %s" % (rootdir + "/ImageSets"))
    voc_imageset_path = rootdir + "/ImageSets"
    if not os.path.exists(voc_imageset_path):
        os.mkdir(voc_imageset_path)

    print("create folder %s" % (rootdir + "/ImageSets/Main"))
    voc_imageset_main_path = rootdir + "/ImageSets/Main"
    if not os.path.exists(voc_imageset_main_path):
        os.mkdir(voc_imageset_main_path)

    # 创建生成标签的VOC格式的文件夹JPEGImages
    print("create folder %s" % (rootdir + "/JPEGImages"))
    voc_jpegimages_path = rootdir + "/JPEGImages"
    if not os.path.exists(voc_jpegimages_path):
        os.mkdir(voc_jpegimages_path)

    # 创建生成标签的VOC格式的文件夹Annotations
    print("create folder %s" % (rootdir + "/Annotations"))
    voc_annotations_path = rootdir + "/Annotations"
    if not os.path.exists(voc_annotations_path):
        os.mkdir(voc_annotations_path)

if __name__=="__main__":
    #1create_voc_style_folders
    create_voc_style_folders()

    #2divide_dataset
    #divide_dataset(rootdir+"/train", rootdir + "/val", 0.8)

    #3convertimgset
    img_sets = ['train', 'val']
    for img_set in img_sets:
        print("handling " + img_set)
        convertimgset(img_set)
    #
    shutil.move(rootdir + "/ImageSets/Main/" + "train.txt", rootdir + "/ImageSets/Main/" + "trainval.txt")
    shutil.move(rootdir + "/ImageSets/Main/" + "val.txt", rootdir + "/ImageSets/Main/" + "test.txt")
