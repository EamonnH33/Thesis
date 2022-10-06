'''

# youtube video and github repo ver useful - note that one change required in lib folder -
I fixed the issue by changing a row in libs/labelDialog.py

row:
layout.addWidget(bb, alignment=Qt.AlignmentFlag.AlignLeft

Changed as

layout.addWidget(bb, alignment=Qt.AlignLeft)
# activate labelimg
# python labelImg.py C:\Users\eamonn.herlihy\PycharmProjects\Thesis\FR_images_3 C:\Users\eamonn.herlihy\PycharmProjects\Thesis\labelImg-master\Class.txt

Need to create FR images as the normal images
then determine categories and draw bounding boxes for large set to fit our own object detection algo
the result using the opensource labelImg software is xmd file
need to figure out how we go from many xmd files to an object detection algorithm
When all this is done we can redo step 2 and 3 (scripts) that have been completed here using naive YOLO

'''