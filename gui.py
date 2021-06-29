import cv2
import time
import numpy as np
import streamlit as st
import math
from time import sleep

from PIL import Image




st.title("Personality Predicition through Handwriting   \n")
st.write("  ")
st.markdown("Our project is based on getting handwriting of an individual and predicting main personality traits of the individual")
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0
def prediction(image):
    # please don't worry about these variables now
    ANCHOR_POINT = 6000
    MIDZONE_THRESHOLD = 15000
    MIN_HANDWRITING_HEIGHT_PIXEL = 20



 
    def bilateralFilter(image, d):
        image = cv2.bilateralFilter(image,d,50,50)
        return image

    def medianFilter(image, d):
        image = cv2.medianBlur(image,d)
        return image


    def threshold(image, t):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,image = cv2.threshold(image,t,255,cv2.THRESH_BINARY_INV)
        return image

    
    def dilate(image, kernalSize):
        kernel = np.ones(kernalSize, np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        return image
        
    
    def erode(image, kernalSize):
        kernel = np.ones(kernalSize, np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        return image
        
    
    def straighten(image):

        global BASELINE_ANGLE
        
        angle = 0.0
        angle_sum = 0.0
        contour_count = 0
      
        
        # apply bilateral filter
        filtered = bilateralFilter(image, 3)
        #cv2.imshow('filtered',filtered)

        # convert to grayscale and binarize the image by INVERTED binary thresholding
        thresh = threshold(filtered, 120)
        #cv2.imshow('thresh',thresh)
        
        # dilate the handwritten lines in image with a suitable kernel for contour operation
        dilated = dilate(thresh, (5 ,100))
        #cv2.imshow('dilated',dilated)
        
        ctrs,hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, ctr in enumerate(ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            
            # We can be sure the contour is not a line if height > width or height is < 20 pixels. Here 20 is arbitrary.
            if h>w or h<MIN_HANDWRITING_HEIGHT_PIXEL:
                continue
            
            # We extract the region of interest/contour to be straightened.
            roi = image[y:y+h, x:x+w]
            #rows, cols = ctr.shape[:2]
            
            # If the length of the line is less than one third of the document width, especially for the last line,
            # ignore because it may yeild inacurate baseline angle which subsequently affects proceeding features.
            
            if w < image.shape[1]/2 :
                roi = 255
                image[y:y+h, x:x+w] = roi
                continue

            # minAreaRect is necessary for straightening
            rect = cv2.minAreaRect(ctr)
            center = rect[0]
            angle = rect[2]
            #print "original: "+str(i)+" "+str(angle)
            # I actually gave a thought to this but hard to remember anyway!
            if angle < -45.0:
                angle += 90.0;
            #print "+90 "+str(i)+" "+str(angle)
                
            rot = cv2.getRotationMatrix2D(((x+w)/2,(y+h)/2), angle, 1)
            #extract = cv2.warpAffine(roi, rot, (w,h), borderMode=cv2.BORDER_TRANSPARENT)
            extract = cv2.warpAffine(roi, rot, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            #cv2.imshow('warpAffine:'+str(i),extract)

            # image is overwritten with the straightened contour
            image[y:y+h, x:x+w] = extract
        
            #print angle
            angle_sum += angle
            contour_count += 1
      
        #cv2.imshow('contours', display)
        
        # mean angle of the contours (not lines) is found
        mean_angle = angle_sum / contour_count
        BASELINE_ANGLE = mean_angle
        print ("Average baseline angle: "+str(mean_angle))
        return image

    def horizontalProjection(img):
        # Return a list containing the sum of the pixels in each row
        (h, w) = img.shape[:2]
        sumRows = []
        for j in range(h):
            row = img[j:j+1, 0:w] # y1:y2, x1:x2
            sumRows.append(np.sum(row))
        return sumRows
        
 
    def verticalProjection(img):
        # Return a list containing the sum of the pixels in each column
        (h, w) = img.shape[:2]
        sumCols = []
        for j in range(w):
            col = img[0:h, j:j+1] # y1:y2, x1:x2
            sumCols.append(np.sum(col))
        return sumCols
        
   
    def extractLines(img):

        global LETTER_SIZE
        global LINE_SPACING
        global TOP_MARGIN
        
        # apply bilateral filter
        filtered = bilateralFilter(img, 5)
        
        # convert to grayscale and binarize the image by INVERTED binary thresholding
        # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
        thresh = threshold(filtered, 160)
        #cv2.imshow('thresh', lthresh)

        # extract a python list containing values of the horizontal projection of the image into 'hp'
        hpList = horizontalProjection(thresh)

        # Extracting 'Top Margin' feature.
        topMarginCount = 0
        for sum in hpList:
            # sum can be strictly 0 as well. Anyway we take 0 and 255.
            if(sum<=255):
                topMarginCount += 1
            else:
                break
                
        #print "(Top margin row count: "+str(topMarginCount)+")"

        # FIRST we extract the straightened contours from the image by looking at occurance of 0's in the horizontal projection.
        lineTop = 0
        lineBottom = 0
        spaceTop = 0
        spaceBottom = 0
        indexCount = 0
        setLineTop = True
        setSpaceTop = True
        includeNextSpace = True
        space_zero = [] # stores the amount of space between lines
        lines = [] # a 2D list storing the vertical start index and end index of each contour
        
        # we are scanning the whole horizontal projection now
        for i, sum in enumerate(hpList):
            # sum being 0 means blank space
            if(sum==0):
                if(setSpaceTop):
                    spaceTop = indexCount
                    setSpaceTop = False # spaceTop will be set once for each start of a space between lines
                indexCount += 1
                spaceBottom = indexCount
                if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                    if(hpList[i+1]==0): # if the next horizontal projectin is 0, keep on counting, it's still in blank space
                        continue
                # we are using this condition if the previous contour is very thin and possibly not a line
                if(includeNextSpace):
                    space_zero.append(spaceBottom-spaceTop)
                else:
                    if (len(space_zero)==0):
                        previous = 0
                    else:
                        previous = space_zero.pop()
                    space_zero.append(previous + spaceBottom-lineTop)
                setSpaceTop = True # next time we encounter 0, it's begining of another space so we set new spaceTop
            
            # sum greater than 0 means contour
            if(sum>0):
                if(setLineTop):
                    lineTop = indexCount
                    setLineTop = False # lineTop will be set once for each start of a new line/contour
                indexCount += 1
                lineBottom = indexCount
                if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                    if(hpList[i+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in contour
                        continue
                        
                    # if the line/contour is too thin <10 pixels (arbitrary) in height, we ignore it.
                    # Also, we add the space following this and this contour itself to the previous space to form a bigger space: spaceBottom-lineTop.
                    if(lineBottom-lineTop<20):
                        includeNextSpace = False
                        setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
                        continue
                includeNextSpace = True # the line/contour is accepted, new space following it will be accepted
                
                # append the top and bottom horizontal indices of the line/contour in 'lines'
                lines.append([lineTop, lineBottom])
                setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
        
     
        
        # SECOND we extract the very individual lines from the lines/contours we extracted above.
        fineLines = [] # a 2D list storing the horizontal start index and end index of each individual line
        for i, line in enumerate(lines):
        
            anchor = line[0] # 'anchor' will locate the horizontal indices where horizontal projection is > ANCHOR_POINT for uphill or < ANCHOR_POINT for downhill(ANCHOR_POINT is arbitrary yet suitable!)
            anchorPoints = [] # python list where the indices obtained by 'anchor' will be stored
            upHill = True # it implies that we expect to find the start of an individual line (vertically), climbing up the histogram
            downHill = False # it implies that we expect to find the end of an individual line (vertically), climbing down the histogram
            segment = hpList[line[0]:line[1]] # we put the region of interest of the horizontal projection of each contour here
            
            for j, sum in enumerate(segment):
                if(upHill):
                    if(sum<ANCHOR_POINT):
                        anchor += 1
                        continue
                    anchorPoints.append(anchor)
                    upHill = False
                    downHill = True
                if(downHill):
                    if(sum>ANCHOR_POINT):
                        anchor += 1
                        continue
                    anchorPoints.append(anchor)
                    downHill = False
                    upHill = True
                    
            #print anchorPoints
            
            # we can ignore the contour here
            if(len(anchorPoints)<2):
                continue
            
         
            # len(anchorPoints) > 3 meaning contour composed of multiple lines
            lineTop = line[0]
            for x in range(1, len(anchorPoints)-1, 2):
                # 'lineMid' is the horizontal index where the segmentation will be done
                lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
                lineBottom = lineMid
                # line having height of pixels <20 is considered defects, so we just ignore it
                # this is a weakness of the algorithm to extract lines (anchor value is ANCHOR_POINT, see for different values!)
                if(lineBottom-lineTop < 20):
                    continue
                fineLines.append([lineTop, lineBottom])
                lineTop = lineBottom
            if(line[1]-lineTop < 20):
                continue
            fineLines.append([lineTop, line[1]])
        
        # LINE SPACING and LETTER SIZE will be extracted here
        # We will count the total number of pixel rows containing upper and lower zones of the lines and add the space_zero/runs of 0's(excluding first and last of the list ) to it.
        fineLines = np.array(fineLines, dtype=np.int)# We will count the total number of pixel rows containing midzones of the lines for letter size.
        # For this, we set an arbitrary (yet suitable!) threshold MIDZONE_THRESHOLD = 15000 in horizontal projection to identify the midzone containing rows.
        # These two total numbers will be divided by number of lines (having at least one row>MIDZONE_THRESHOLD) to find average line spacing and average letter size.
        space_nonzero_row_count = 0
        midzone_row_count = 0
        lines_having_midzone_count = 0

        flag = False
    






        for i, line in enumerate(fineLines):
            segment = hpList[line[0]:line[1]]
            for j, sum in enumerate(segment):
                if(sum<MIDZONE_THRESHOLD):
                    space_nonzero_row_count += 1
                else:
                    midzone_row_count += 1
                    flag = True
                    
            # This line has contributed at least one count of pixel row of midzone
            if(flag):
                lines_having_midzone_count += 1
                flag = False
        
        # error prevention ^-^
        if(lines_having_midzone_count == 0): lines_having_midzone_count = 1
        
        
        total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1]) 
        #excluding first and last entries: Top and Bottom margins
        # the number of spaces is 1 less than number of lines but total_space_row_count contains the top and bottom spaces of the line
        average_line_spacing = float(total_space_row_count) / lines_having_midzone_count 
        average_letter_size = float(midzone_row_count) / lines_having_midzone_count
        # letter size is actually height of the letter and we are not considering width
        LETTER_SIZE = average_letter_size
        # error prevention ^-^
        if(average_letter_size == 0): average_letter_size = 1
        # We can't just take the average_line_spacing as a feature directly. We must take the average_line_spacing relative to average_letter_size.
        # Let's take the ratio of average_line_spacing to average_letter_size as the LINE SPACING, which is perspective to average_letter_size.
        relative_line_spacing = average_line_spacing / average_letter_size
        LINE_SPACING = relative_line_spacing
   
        relative_top_margin = float(topMarginCount) / average_letter_size
        TOP_MARGIN = relative_top_margin
        
      
     
        return fineLines
        

    def extractWords(image, lines):

        global LETTER_SIZE
        global WORD_SPACING
        
        # apply bilateral filter
        filtered = bilateralFilter(image, 5)
        
        # convert to grayscale and binarize the image by INVERTED binary thresholding
        thresh = threshold(filtered, 180)
        #cv2.imshow('thresh', wthresh)
        
        # Width of the whole document is found once.
        width = thresh.shape[1]
        space_zero = [] # stores the amount of space between words
        words = [] # a 2D list storing the coordinates of each word: y1, y2, x1, x2
        
        # Isolated words or components will be extacted from each line by looking at occurance of 0's in its vertical projection.
        for i, line in enumerate(lines):
            extract = thresh[line[0]:line[1], 0:width] # y1:y2, x1:x2
            vp = verticalProjection(extract)
            #print i
            #print vp
            
            wordStart = 0
            wordEnd = 0
            spaceStart = 0
            spaceEnd = 0
            indexCount = 0
            setWordStart = True
            setSpaceStart = True
            includeNextSpace = True
            spaces = []
            
            # we are scanning the vertical projection
            for j, sum in enumerate(vp):
                # sum being 0 means blank space
                if(sum==0):
                    if(setSpaceStart):
                        spaceStart = indexCount
                        setSpaceStart = False # spaceStart will be set once for each start of a space between lines
                    indexCount += 1
                    spaceEnd = indexCount
                    if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                        if(vp[j+1]==0): # if the next vertical projectin is 0, keep on counting, it's still in blank space
                            continue

                    # we ignore spaces which is smaller than half the average letter size
                    if((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                        spaces.append(spaceEnd-spaceStart)
                        
                    setSpaceStart = True # next time we encounter 0, it's begining of another space so we set new spaceStart
                
                # sum greater than 0 means word/component
                if(sum>0):
                    if(setWordStart):
                        wordStart = indexCount
                        setWordStart = False # wordStart will be set once for each start of a new word/component
                    indexCount += 1
                    wordEnd = indexCount
                    if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                        if(vp[j+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in non-space zone
                            continue
                    
                    # append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                    # we ignore the ones which has height smaller than half the average letter size
                    # this will remove full stops and commas as an individual component
                    count = 0
                    for k in range(line[1]-line[0]):
                        row = thresh[line[0]+k:line[0]+k+1, wordStart:wordEnd] # y1:y2, x1:x2
                        if(np.sum(row)):
                            count += 1
                    if(count > int(LETTER_SIZE/2)):
                        words.append([line[0], line[1], wordStart, wordEnd])
                        
                    setWordStart = True # next time we encounter value > 0, it's begining of another word/component so we set new wordStart
            
            space_zero.extend(spaces[1:-1])
        
        #print space_zero
        space_columns = np.sum(space_zero)
        space_count = len(space_zero)
        if(space_count == 0):
            space_count = 1
        average_word_spacing = float(space_columns) / space_count
        print(LETTER_SIZE)
    
        relative_word_spacing = average_word_spacing / LETTER_SIZE
        WORD_SPACING = relative_word_spacing
        #print "Average word spacing: "+str(average_word_spacing)
        #print ("Average word spacing relative to average letter size: "+str(relative_word_spacing))
        
        return words
            
    
    def extractSlant(img, words):
        
        global SLANT_ANGLE
  
        # We are checking for 9 different values of angle
        theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
        #theta = [-0.785398, -0.523599, -0.436332, -0.349066, -0.261799, -0.174533, -0.0872665, 0, 0.0872665, 0.174533, 0.261799, 0.349066, 0.436332, 0.523599, 0.785398]

        # Corresponding index of the biggest value in s_function will be the index of the most likely angle in 'theta'
        s_function = [0.0] * 9
        count_ = [0]*9
        
        # apply bilateral filter
        filtered = bilateralFilter(img, 5)
        
        # convert to grayscale and binarize the image by INVERTED binary thresholding
        # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
        thresh = threshold(filtered, 180)
        #cv2.imshow('thresh', lthresh)
        
        # loop for each value of angle in theta
        for i, angle in enumerate(theta):
            s_temp = 0.0 # overall sum of the functions of all the columns of all the words!
            count = 0 # just counting the number of columns considered to contain a vertical stroke and thus contributing to s_temp
            
            #loop for each word
            for j, word in enumerate(words):
                original = thresh[word[0]:word[1], word[2]:word[3]] # y1:y2, x1:x2

                height = word[1]-word[0]
                width = word[3]-word[2]
                
                # the distance in pixel we will shift for affine transformation
                # it's divided by 2 because the uppermost point and the lowermost points are being equally shifted in opposite directions
                shift = (math.tan(angle) * height) / 2
                
                # the amount of extra space we need to add to the original image to preserve information
                # yes, this is adding more number of columns but the effect of this will be negligible
                pad_length = abs(int(shift))
                
                # create a new image that can perfectly hold the transformed and thus widened image
                blank_image = np.zeros((height,width+pad_length*2,3), np.uint8)
                new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                new_image[:, pad_length:width+pad_length] = original
                
                # points to consider for affine transformation
                (height, width) = new_image.shape[:2]
                x1 = width/2
                y1 = 0
                x2 = width/4
                y2 = height
                x3 = 3*width/4
                y3 = height
        
                pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
                pts2 = np.float32([[x1+shift,y1],[x2-shift,y2],[x3-shift,y3]])
                M = cv2.getAffineTransform(pts1,pts2)
                deslanted = cv2.warpAffine(new_image,M,(width,height))
                
                # find the vertical projection on the transformed image
                vp = verticalProjection(deslanted)
                
                # loop for each value of vertical projection, which is for each column in the word image
                for k, sum in enumerate(vp):
                    # the columns is empty
                    if(sum == 0):
                        continue
                    
                    # this is the number of foreground pixels in the column being considered
                    num_fgpixel = sum / 255

                    # if number of foreground pixels is less than onethird of total pixels, it is not a vertical stroke so we can ignore
                    if(num_fgpixel < int(height/3)):
                        continue
                    
                    # the column itself is extracted, and flattened for easy operation
                    column = deslanted[0:height, k:k+1]
                    column = column.flatten()
                    
                    # now we are going to find the distance between topmost pixel and bottom-most pixel
                    # l counts the number of empty pixels from top until and upto a foreground pixel is discovered
                    for l, pixel in enumerate(column):
                        if(pixel==0):
                            continue
                        break
                    # m counts the number of empty pixels from bottom until and upto a foreground pixel is discovered
                    for m, pixel in enumerate(column[::-1]):
                        if(pixel==0):
                            continue
                        break
                    
                    # the distance is found as delta_y, I just followed the naming convention in the research paper I followed
                    delta_y = height - (l+m)
                
                    # please refer the research paper for more details of this function, anyway it's nothing tricky
                    h_sq = (float(num_fgpixel)/delta_y)**2
                    
                    # I am multiplying by a factor of num_fgpixel/height to the above function to yeild better result
                    # this will also somewhat negate the effect of adding more columns and different column counts in the transformed image of the same word
                    h_wted = (h_sq * num_fgpixel) / height

                
                    # add up the values from all the loops of ALL the columns of ALL the words in the image
                    s_temp += h_wted
                    
                    count += 1
                
              
                    
            s_function[i] = s_temp
            count_[i] = count
        
        # finding the largest value and corresponding index
        max_value = 0.0
        max_index = 4
        for index, value in enumerate(s_function):
            #print str(index)+" "+str(value)+" "+str(count_[index])
            if(value > max_value):
                max_value = value
                max_index = index
                
        # We will add another value 9 manually to indicate irregular slant behaviour.
        # This will be seen as value 4 (no slant) but 2 corresponding angles of opposite sign will have very close values.
        if(max_index == 0):
            angle = 45
            result =  " : Extremely right slanted"
        elif(max_index == 1):
            angle = 30
            result = " : Above average right slanted"
        elif(max_index == 2):
            angle = 15
            result = " : Average right slanted"
        elif(max_index == 3):
            angle = 5
            result = " : A little right slanted"
        elif(max_index == 5):
            angle = -5
            result = " : A little left slanted"
        elif(max_index == 6):
            angle = -15
            result = " : Average left slanted"
        elif(max_index == 7):
            angle = -30
            result = " : Above average left slanted"
        elif(max_index == 8):
            angle = -45
            result = " : Extremely left slanted"
        elif(max_index == 4):
            p = s_function[4] / s_function[3]
            q = s_function[4] / s_function[5]
            #print 'p='+str(p)+' q='+str(q)
            # the constants here are abritrary but I think suits the best
            if((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
                angle = 0
                result = " : No slant"
            elif((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
                angle = 0
                result = " : No slant"
            else:
                max_index = 9
                angle = 180
                result =  " : Irregular slant behaviour"
            
            
        

            
        
        SLANT_ANGLE = angle
        print ("Slant angle(degree): "+str(SLANT_ANGLE)+result)
        return

   
    def barometer(image):

        global PEN_PRESSURE

        # it's extremely necessary to convert to grayscale first
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # inverting the image pixel by pixel individually. This costs the maximum time and processing in the entire process!
        h, w = image.shape[:]
        inverted = image
        for x in range(h):
            for y in range(w):
                inverted[x][y] = 255 - image[x][y]
        
        #cv2.imshow('inverted', inverted)
        
        # bilateral filtering
        filtered = bilateralFilter(inverted, 3)
        
        # binary thresholding. Here we use 'threshold to zero' which is crucial for what we want.
        # If src(x,y) is lower than threshold=100, the new pixel value will be set to 0, else it will be left untouched!
        ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
        #cv2.imshow('thresh', thresh)
        
        # add up all the non-zero pixel values in the image and divide by the number of them to find the average pixel value in the whole image
        total_intensity = 0
        pixel_count = 0
        for x in range(h):
            for y in range(w):
                if(thresh[x][y] > 0):
                    total_intensity += thresh[x][y]
                    pixel_count += 1
                    
        average_intensity = float(total_intensity) / pixel_count
        PEN_PRESSURE = average_intensity
        #print total_intensity
        #print pixel_count
        # print ("Average pen pressure: "+str(average_intensity))

        return


    # Extract pen pressure. It's such a cool function name!
    barometer(image)

    # apply contour operation to straighten the contours which may be a single line or composed of multiple lines
    # the returned image is straightened version of the original image without filtration and binarization
    straightened = straighten(image)

        
    # extract lines of handwritten text from the image using the horizontal projection
    # it returns a 2D list of the vertical starting and ending index/pixel row location of each line in the handwriting
    lineIndices = extractLines(straightened)
    #print lineIndices
    #print

    # extract words from each line using vertical projection
    # it returns a 4D list of the vertical starting and ending indices and horizontal starting and ending indices (in that order) of each word in the handwriting
    wordCoordinates = extractWords(straightened, lineIndices)

        
    #print wordCoordinates
    #print len(wordCoordinates)
    #for i, item in enumerate(wordCoordinates):
    #	cv2.imshow('item '+str(i), straightened[item[0]:item[1], item[2]:item[3]])
        
    # extract average slant angle of all the words containing a long vertical stroke
    extractSlant(straightened, wordCoordinates)







uploaded_file = st.file_uploader("Choose a image file")

st.success('To Continue Please Click Predict Button')
sideb = st.sidebar



if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    imageuploaded = Image.open(uploaded_file)

    # Now do something with the image! For example, let's display it:
    st.image(uploaded_file, width=None)
  

    if st.button('predict'):
    
        
   
        prediction(opencv_image)
        def determine_baseline_angle(raw_baseline_angle):
            comment = ""
            # falling
            if(raw_baseline_angle >= 0.2):
                baseline_angle = 0
                comment = "DESCENDING"
                # rising
            elif(raw_baseline_angle <= -0.3):
                baseline_angle = 1
                comment = "ASCENDING"
            # straight
            else:
                baseline_angle = 2
                comment = "STRAIGHT"
                
                
            return baseline_angle, comment

        def determine_top_margin(raw_top_margin):
            comment = ""
            # medium and bigger
            if(raw_top_margin >= 1.7):
                top_margin = 0
                comment = "MEDIUM OR BIGGER"
                # narrow
            else:
                top_margin = 1
                comment = "NARROW"
                    
            
            return top_margin, comment

        def determine_letter_size(raw_letter_size):
            comment = ""
            # big
            if(raw_letter_size >= 18.0):
                letter_size = 0
                comment = "BIG"
                # small
            elif(raw_letter_size < 13.0):
                letter_size = 1
                comment = "SMALL"
            # medium
            else:
                letter_size = 2
                comment = "MEDIUM"
                    

            return letter_size, comment

        def determine_line_spacing(raw_line_spacing):
            comment = ""
            # big
            if(raw_line_spacing >= 3.5):
                line_spacing = 0
                comment = "BIG"
                # small
            elif(raw_line_spacing < 2.0):
                line_spacing = 1
                comment = "SMALL"
                # medium
            else:
                line_spacing = 2
                comment = "MEDIUM"
                    
            
            return line_spacing, comment

        def determine_word_spacing(raw_word_spacing):
            comment = ""
            # big
            if(raw_word_spacing > 2.0):
                word_spacing = 0
                comment = "BIG"
                # small
            elif(raw_word_spacing < 1.2):
                word_spacing = 1
                comment = "SMALL"
                # medium
            else:
                word_spacing = 2
                comment = "MEDIUM"


                    
            return word_spacing, comment

        def determine_pen_pressure(raw_pen_pressure):
            comment = ""
            # heavy
            if(raw_pen_pressure > 180.0):
                pen_pressure = 0
                comment = "HEAVY"
                # light
            elif(raw_pen_pressure < 151.0):
                pen_pressure = 1
                comment = "LIGHT"
                # medium
            else:
                pen_pressure = 2
                comment = "MEDIUM"
                    
            
            return pen_pressure, comment

        def determine_slant_angle(raw_slant_angle):
            comment = ""
            # extremely reclined
            if(raw_slant_angle == -45.0 or raw_slant_angle == -30.0):
                slant_angle = 0
                comment = "EXTREMELY RECLINED"
                # a little reclined or moderately reclined
            elif(raw_slant_angle == -15.0 or raw_slant_angle == -5.0 ):
                slant_angle = 1
                comment = "A LITTLE OR MODERATELY RECLINED"
                # a little inclined
            elif(raw_slant_angle == 5.0 or raw_slant_angle == 15.0 ):
                slant_angle = 2
                comment = "A LITTLE INCLINED"
                # moderately inclined
            elif(raw_slant_angle == 30.0 ):
                slant_angle = 3
                comment = "MODERATELY INCLINED"
                # extremely inclined
            elif(raw_slant_angle == 45.0 ):
                slant_angle = 4
                comment = "EXTREMELY INCLINED"
                # straight
            elif(raw_slant_angle == 0.0 ):
                slant_angle = 5
                comment = "STRAIGHT"
                # irregular
                #elif(raw_slant_angle == 180 ):
            else:
                slant_angle = 6
                comment = "IRREGULAR"


            return slant_angle, comment


        # trait_1 = emotional stability | 1 = stable, 0 = not stable
        def determine_trait_1(baseline_angle, slant_angle):
            if (slant_angle == 0 or slant_angle == 4 or slant_angle == 6 or baseline_angle == 0):
                return 0
            else:
                return 1
        # trait_2 = mental energy or will power | 1 = high or average, 0 = low
        def determine_trait_2(letter_size, pen_pressure):
            if ((pen_pressure == 0 or pen_pressure == 2) or (letter_size == 1 or letter_size == 2)):
                return 1
            else:
                return 0
        # trait_3 = modesty | 1 = observed, 0 = not observed (not necessarily the opposite)
        def determine_trait_3(top_margin, letter_size):
            if (top_margin == 0 or  letter_size == 1):
                return 1
            else:
                return 0
        # trait_4 = personal harmony and flexibility | 1 = harmonious, 0 = non harmonious
        def determine_trait_4(line_spacing, word_spacing):
            if (line_spacing == 2 and word_spacing == 2):
                return 1
            else:
                return 0
        # trait_5 = lack of discipline | 1 = observed, 0 = not observed (not necessarily the opposite)
        def determine_trait_5(top_margin, slant_angle):
            if (top_margin == 1 and slant_angle == 6):
                return 1
            else:
                return 0
            
        # trait_6 = poor concentration power | 1 = observed, 0 = not observed (not necessarily the opposite)
        def determine_trait_6(letter_size, line_spacing):
            if (letter_size == 0 and line_spacing == 1):
                return 1
            else:
                return 0
        # trait_7 = non communicativeness | 1 = observed, 0 = not observed (not necessarily the opposite)
        def determine_trait_7(letter_size, word_spacing):
            if (letter_size == 1 and word_spacing == 0):
                return 1
            else:
                return 0
        # trait_8 = social isolation | 1 = observed, 0 = not observed (not necessarily the opposite)
        def determine_trait_8(line_spacing, word_spacing):
            if (word_spacing == 0 or line_spacing == 0):
                return 1
            else:
                return 0








        baseline_angle, comment1 = determine_baseline_angle(BASELINE_ANGLE)
        top_margin, comment2 = determine_top_margin(TOP_MARGIN)
        letter_size, comment3 = determine_letter_size(LETTER_SIZE)
        line_spacing, comment4 = determine_line_spacing(LINE_SPACING)
        word_spacing, comment5 = determine_word_spacing(WORD_SPACING)
        pen_pressure, comment6 = determine_pen_pressure(PEN_PRESSURE)
        slant_angle, comment7 = determine_slant_angle(SLANT_ANGLE)

        # trait_1 = emotional stability | 1 = stable, 0 = not stable
        trait_1 = determine_trait_1(baseline_angle, slant_angle)
        if(trait_1 == 1):
            trait1_comment="The Person is Emotionally Stable"
        else:
            trait1_comment="The Person is Emotionally Unstable"
        

        # trait_2 = mental energy or will power | 1 = high or average, 0 = low
        trait_2 = determine_trait_2(letter_size, pen_pressure)
        if(trait_2 == 1):
            trait2_comment="The Person has High/Average Mental Energy or Will power"
        else:
            trait2_comment="The Person has Low Mental Energy or Will power"
        
        # trait_3 = modesty | 1 = observed, 0 = not observed (not necessarily the opposite)
        trait_3 = determine_trait_3(top_margin, letter_size)
        if(trait_3 == 1):
            trait3_comment="The Person is Modest (To some extent)"
        else:
            trait3_comment="The Person is Immodest (not necessarily the opposite)"
        # trait_4 = personal harmony and flexibility | 1 = harmonious, 0 = non harmonious
        trait_4 = determine_trait_4(line_spacing, word_spacing)
        if(trait_4 == 1):
            trait4_comment="The Person has a level of Personal Harmony and Flexibility"
        else:
            trait4_comment="The Person doesn't have Personal Harmony and Flexibility(To some extent)"
        
        
        # trait_5 = lack of discipline | 1 = observed, 0 = not observed (not necessarily the opposite)
        trait_5 = determine_trait_5(top_margin, slant_angle)
        if(trait_5 == 1):
            trait5_comment="The person has a lack of discipline"
        else:
            trait5_comment="The person is discipline"
        
        # trait_6 = poor concentration power | 1 = observed, 0 = not observed (not necessarily the opposite)
        trait_6 = determine_trait_6(letter_size, line_spacing)
        if(trait_6 == 1):
            trait6_comment="The person has poor concentration power(lack of focus)"
        else:
            trait6_comment="The person has good power of concentration"
        # trait_7 = non communicativeness | 1 = observed, 0 = not observed (not necessarily the opposite)
        trait_7 = determine_trait_7(letter_size, word_spacing)
        if(trait_7 == 1):
            trait7_comment="The person is Non Communicativeness)"
        else:
            trait7_comment="The person is Non Communicativeness(to some extent)"
        
        
        # trait_8 = social isolation | 1 = observed, 0 = not observed (not necessarily the opposite)
        trait_8 = determine_trait_8(line_spacing, word_spacing)
        if(trait_8 == 1):
            trait8_comment="The person is Socially Isolated"
        else:
            trait8_comment="The person is outgoing and socially interactive"
        
        print(trait_1)
        print(trait_2)
        print(trait_3)
        print(trait_4)
        print(trait_5)
        print(trait_6)
        print(trait_7)
        print(trait_8)
        
    


        

        


        st.write('baseline angle is:', comment1, BASELINE_ANGLE)
        st.write('Top Margin is:', comment2, TOP_MARGIN)
        st.write('Letter Size is:', comment3, LETTER_SIZE)
        st.write('Line Spacing is:', comment4, LINE_SPACING)
        st.write('Word Spacing is:', comment5, WORD_SPACING)
        st.write('Slant Angle is:', comment7, SLANT_ANGLE)
        st.write('Pen Pressure is:', comment6, PEN_PRESSURE)


        st.write(trait1_comment)
        st.write(trait2_comment)
        st.write(trait3_comment)
        st.write(trait4_comment)
        st.write(trait5_comment)
        st.write(trait6_comment)
        st.write(trait7_comment)
        st.write(trait8_comment)








