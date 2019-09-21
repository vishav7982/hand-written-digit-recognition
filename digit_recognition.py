import cv2
import numpy as np
# training data to our algorithm
digits = cv2.imread("digits.png",cv2.IMREAD_GRAYSCALE)
#test data to our algorithm
test_digits = cv2.imread("test_digits.png",cv2.IMREAD_GRAYSCALE)

# list of testing data
test_digits = np.vsplit(test_digits,50)
test_cells = []
for d in test_digits:
    d= d.flatten()
    test_cells.append(d)

#coverting test data int np array coz knn works on np arrays not normal arrays
test_cells = np.array(test_cells,dtype =np.float32)


rows = np.vsplit(digits,50);  #diving the image into 50 rows as image is 50 rows * 50 columns

# empty cells list (training data) each cell is one digit
cells = []
for row in rows:
    row_cells = np.hsplit(row,50)   # each cell  is  extracted from each row using horizontal split
    for cell in row_cells:
        # cell is an image of just one digit. IT corresponds to an array of RGB intensity of colours of that digit
        cell = cell.flatten()
        cells.append(cell)

cells = np.array(cells,dtype = np.float32)
k =np.arange(10)
cells_label = np.repeat(k,250)
# K-nearest neighbour algorirthm

knn = cv2.ml.KNearest_create()

knn.train(cells,cv2.ml.ROW_SAMPLE,cells_label) # trainging our algo to tell it that first 250 are 0 next are ones and so on

#now we will test our algo

ret,result,neighbours,distance= knn.findNearest(test_cells,k=7) #k = closest distance you want

print(result)