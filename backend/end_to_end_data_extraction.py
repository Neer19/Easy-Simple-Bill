import csv
import cv2
from sklearn.cluster import KMeans
import requests
import re
import sys
import pandas as pd
from paddleocr import PaddleOCR
ocr = PaddleOCR()


# Header Extraction
def extract_header(h):
    header = ocr.ocr(h)
    header = sorted(header, key=lambda x: x[0][0][1])
    temp = [0]
    for i in range(len(header) - 1):
        if abs(header[i][0][0][1] - header[i+1][0][0][1]) > 15:
            temp.append(i+1)
    temp.append(len(header))
    t = []
    for i in range(len(temp) - 1):
        t.append(header[temp[i]:temp[i+1]])
        
    for i in range(len(t)):
        t[i] = sorted(t[i], key=lambda x: x[0][0][0])

    header_txt = []
    for i in range(len(t)):
        header_txt.append(' '.join([x[1][0] for x in t[i]]))

    header_data = {}
    
    # Patient Name
    patient_name = ""

    for i in header_txt:
        try:
            x = re.search('(?:patient name|pt. name|m/s|name)',
                          i, re.IGNORECASE).span()
            patient_name += i[x[1]:]
            break
        except:
            pass

    patient_name = patient_name.replace(':', ' ')
    patient_name = patient_name.strip()

    # Patient Address
    patient_address = ""

    for i in header_txt:
        try:
            x = re.search('(?:address|add)', i, re.IGNORECASE).span()
            patient_address += i[x[1]:]
            break
        except:
            pass
    patient_address = patient_address.replace(':', ' ')
    patient_address = patient_address.strip()

    # Date
    date = ""

    for i in header_txt:
        try:
            x = re.search('date', i, re.IGNORECASE).span()
            date += i[x[1]:]
            break
        except:
            pass
    date = date.replace('.', ' ')
    date = date.replace(':', ' ')
    date = date.strip()
    date = date.split(' ')[0]
    try:
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
    except:
        date = ""

    # Patient Mobile Number
    text = ' '.join(header_txt)
    text = text.split(".")
    text = ' '.join(text)
    text = text.split(",")
    text = ' '.join(text)
    text = text.split(" ")
    numbers = []
    for i in text:
        if 10 <= len(i) <= 12:
            for j in i:
                if not j.isdigit():
                    break
            else:
                numbers.append(int(i))

    # Invoice Number
    invoice_no = ""

    for i in range(len(header_txt)):
        header_txt[i] = header_txt[i].replace(':-', ' ')
        header_txt[i] = header_txt[i].replace('.', ' ')
        header_txt[i] = header_txt[i].replace(':', ' ')
        header_txt[i] = header_txt[i].replace('-:', ' ')
        try:
            x = re.search('(?:invoice no.|inv.no.|inv. no.|invoiceno|g.rcpt. no.|g. rcpt. no.|estimate no.|s.return no.|p.return no.|bill no|billno)',
                          header_txt[i], re.IGNORECASE).span()
            s = header_txt[i][x[1]:]
            s = s.split(' ')
            s = [i for i in s if i != '']
            invoice_no += s[0]
            break
        except:
            pass

    header_data['patient_name'] = patient_name
    header_data['patient_address'] = patient_address
    header_data['date'] = date
    header_data['invoice_no'] = invoice_no
    header_data['patient_mobile_no'] = numbers

    return header_data


# Table Extraction
def extract_table1(t):
    text = ocr.ocr(t)

    temp = [0]
    for i in range(len(text) - 1):
        if abs(text[i][0][0][1] - text[i+1][0][0][1]) > 15:
            temp.append(i+1)
    temp.append(len(text))
    t = []
    for i in range(len(temp) - 1):
        t.append(text[temp[i]:temp[i+1]])

    for i in range(len(t)):   
        t[i] = sorted(t[i], key=lambda x: x[0][0][0])

    header = [i[1][0] for i in t[0]]

    x = []
    for i in range(len(t)):
        for j in range(len(t[i])):
            x.append((t[i][j][1][0], t[i][j][0][0][0], i))

    coords = []
    for i in range(len(t)):
        for j in range(len(t[i])):
            x1, y1, x2, y2 = t[i][j][0][0][0], t[i][j][0][0][1], t[i][j][0][2][0], t[i][j][0][2][1]
            coords.append(((x1 + x2) // 2, i))

    clusters = KMeans(n_clusters=len(header), init='k-means++', max_iter=300, n_init=10, random_state=0)

    clusters.fit(coords)

    cols = []
    for j in range(len(header)):
        cluster = []
        for i in range(len(clusters.labels_)):
            if clusters.labels_[i] == j:
                cluster.append(x[i])
        cols.append(cluster)

    for i in range(len(cols)):
        for j in range(len(t)):
            try:
                n = cols[i][j][2]
                if n != j:
                    cols[i].insert(j, ('BLANK', cols[i][0][1], j))
            except:
                cols[i].insert(j, ('BLANK', cols[i][0][1], j))

    c = list(map(list, zip(*cols)))
    c = [sorted(i, key=lambda x: x[1]) for i in c]

    for i in range(len(c)):
            count_blank = 0
            for j in range(len(c[i])):
                if c[i][j][0] == 'BLANK':
                    count_blank += 1
            if count_blank > len(c[i]) // 2:
                c[i] = []

    c = [i for i in c if i != []]

    table_final = []

    for i in range(len(c)):
        table_final.append([c[i][j][0] for j in range(len(c[i]))])

    table_ = table_final[1:]

    df = pd.DataFrame(table_, columns=header)
    return df
    
def extract_table2(t):    
    t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    t = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    t = cv2.medianBlur(t, 3)
    t = cv2.bitwise_not(t)
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    t = 255 - t
    t = cv2.resize(t, (0, 0), fx=0.8, fy=0.5)  
    img = t
    # define border color
    lower = (0, 100, 110)
    upper = (0, 100, 150)
    # threshold on border color
    mask = cv2.inRange(img, lower, upper)
    # dilate threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    # recolor border to white
    img[mask == 255] = (255, 255, 255)
    # convert img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # otsu threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] 
    # apply morphology open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 40))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph
    # find contours and bounding boxes
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if w > 80 and h > 20:
            cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bboxes.append((x, y, w, h))
    bboxes = sorted(bboxes, key=lambda x: x[0])
    # crop all bounding boxes
    columns = []
    for bbox in bboxes:
        x, y, w, h = bbox
        columns.append(img[y:y + h + 20, x:x + w + 20])
    cs = []

    for column in columns:
        text = ocr.ocr(column)
        cs.append([[text[i][1][0], text[i][0][0][0], text[i][0][0][1]] for i in range(len(text))])
    t = len(cs[-1])
    for i in range(len(cs) - 1):
        for j in range(t):
            try:
                if abs(cs[-1][j][2] - cs[i][j][2]) > 30:
                    cs[i].insert(j, ['BLANK', cs[i][0][1], cs[-1][j][2]])
            except:
                cs[i].insert(j, ['BLANK', cs[i][0][1], cs[-1][j][2]])
    # transpose cs
    cs = list(map(list, zip(*cs)))
    for i in range(len(cs)):
        count_blank = 0
        for j in range(len(cs[i])):
            if cs[i][j][0] == 'BLANK':
                count_blank += 1
        if count_blank > len(cs[i]) // 2:
            cs[i] = []

    cs = [i for i in cs if i != []]

    table_final = []

    for i in range(len(cs)):
        table_final.append([cs[i][j][0] for j in range(len(cs[i]))])

    header = table_final[0]
    table_final = table_final[1:]

    df = pd.DataFrame(table_final, columns=header)
    return df
    

# Footer Extraction
def extract_footer(f):    
    footer = ocr.ocr(f)

    temp = [0]
    for i in range(len(footer) - 1):
        if abs(footer[i][0][0][1] - footer[i+1][0][0][1]) > 15:
            temp.append(i+1)
    temp.append(len(footer))
    t = []
    for i in range(len(temp) - 1):
        t.append(footer[temp[i]:temp[i+1]])
        t[i] = sorted(t[i], key=lambda x: x[0][0][0])

    # last apperance of 'total'
    index = 0
    for i in range(len(t)):
        for j in range(len(t[i])):
            if 'total' in t[i][j][1][0].lower():
                index = i
                x1 = int(t[i][j][0][0][0]) - 150
                y2 = int(t[i][j][0][2][1]) + 140
                break
        if index != 0:
            break
    if index == 0:
        for i in range(len(t)):
            for j in range(len(t[i])):
                if 'please' in t[i][j][1][0].lower():
                    index = i
                    x1 = int(t[i][j][0][0][0]) - 150
                    y2 = int(t[i][j][0][2][1])
                    break
            if index != 0:
                break

    for i in range(len(t)):
        for j in range(len(t[i])):
            if t[i][j][0][0][0] + 150 < x1 or t[i][j][0][2][1] - 140 > y2:
                t[i][j] = []

    for i in range(len(t)):
        t[i] = [i for i in t[i] if i != []]
        

    footer_txt = []

    for i in range(len(t)):
        footer_txt.append(' '.join([j[1][0] for j in t[i]]))

    footer = footer_txt

    footer = [i.strip() for i in footer]

    for i in range(len(footer)):
        footer[i] = footer[i].replace(':', '')
        footer[i] = footer[i].replace(',', '')

    footer = [i.strip() for i in footer]

    # remove empty strings
    footer = [i for i in footer if i != '']
    footer = [i for i in footer if i != ' ']

    try:
        status = False
        x = footer[-2].split(' ')
        for i in x:
            try:
                x = float(i)
                status = True
                break
            except:
                pass
        if not status:
            footer[-2] += " " + footer[-1]
            footer.pop(-1)
    except:
        pass

    # make key value pairs
    v = []
    k = []

    for i in footer:
        v.append(i.split(' ')[-1])
        k.append(' '.join(i.split(' ')[:-1]))

    for i in range(len(v)):
        try:
            v[i] = float(v[i])
        except:
            k[i] = ""
            v[i] = ""

    for i in range(len(k)):
        try:
            x = float(k[i][0])
            k[i] = ""
            v[i] = ""
        except:
            pass

    d = dict(zip(k, v))

    # remove empty strings
    footer = {k: v for k, v in d.items() if k != ''}
    return footer


# main function
def extractcsv(image_path, switch):
     # 1 for clustering based table extraction, 2 for whitespace based table extraction
    # image_path = sys.argv[1]
    # switch = int(sys.argv[2])
    image = cv2.imread(image_path)
    retailer_id = image_path.split('/')[-1].split('-')[0]

    y2 = requests.get(f"https://idpadmin.raseet.com/panel/api/configs/{retailer_id}").json()['y1'] - 20
    h = image[:y2, 200:]
    h = cv2.resize(h, (0, 0), fx=2, fy=2)
    header = extract_header(h)
    df1 = pd.DataFrame(header)
    df1.to_csv('final.csv', index=False)


    y1 = requests.get(f"https://idpadmin.raseet.com/panel/api/configs/{retailer_id}").json()['y1']
    y2 = requests.get(f"https://idpadmin.raseet.com/panel/api/configs/{retailer_id}").json()['y2']
    t = image[y1:y2, 110:image.shape[1] - 110]
    if switch == 1:
        table = extract_table1(t)
    else:
        table = extract_table2(t)
    table.to_csv('table.csv', index=False)

    


    y1 = requests.get(f"https://idpadmin.raseet.com/panel/api/configs/{retailer_id}").json()['y2']
    x1 = 200
    f = image[y1:, x1:]
    footer = extract_footer(f)
    print("\n\n", header, "\n\n", table, "\n\n", footer)
    df3 = pd.DataFrame(footer, index=[0])
    print("Footer converted to dataframe")

    df3.to_csv('footer.csv', index=False)

    print("Footer converted to csv")

    with open('final.csv', 'a',newline='') as f1:
        writer = csv.writer(f1)
        with open('table.csv', 'r') as f2:
            reader1 = csv.reader(f2)
            for row in reader1:
                writer.writerow(row)
        
        with open('footer.csv', 'r') as f3:
            reader2 = csv.reader(f3)
            for row in reader2:
                writer.writerow(row)

if __name__=="__main__":
    extractcsv()
