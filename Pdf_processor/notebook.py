import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import tensorflow as tf
from paddleocr import PaddleOCR, draw_ocr
import layoutparser as lp
from fuzzywuzzy import fuzz
import fitz
import json
from transformers import T5Tokenizer, TFT5ForConditionalGeneration 
import re
import PyPDF2
from pdf2image import convert_from_path
from PyPDF2 import PdfWriter, PdfReader



def intersection(box_1, box_2):
  return [box_2[0], box_1[1],box_2[2], box_1[3]]


#**********************************************************************

def iou(box_1, box_2):

  x_1 = max(box_1[0], box_2[0])
  y_1 = max(box_1[1], box_2[1])
  x_2 = min(box_1[2], box_2[2])
  y_2 = min(box_1[3], box_2[3])

  inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
  if inter == 0:
      return 0

  box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
  box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

  return inter / float(box_1_area + box_2_area - inter)

#**********************************************************************
 
# Fonction pour trouver une page spécifique dans un PDF

def Find_Page(pdf_path, search_word, not_in_page_phrase):
    # Initialize a variable to track if the word is found
    found_search_word = False
    found_not_in_page_phrase = False
    specific_page_number = None

    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Iterate over each page in the PDF file
        for page_num, page in enumerate(pdf_reader.pages):
            # Get the text from the current page
            page_text = page.extract_text()
            
            # Check if the "not in page" phrase is NOT present in the page text
            if not_in_page_phrase.lower() not in page_text.lower():
                found_not_in_page_phrase = True

            # Check if the search word is present in the page text
            if found_not_in_page_phrase and (search_word.lower() in page_text.lower()):
                found_search_word = True

            # If both conditions are met, set the page number and break out of the loop
            if found_search_word and found_not_in_page_phrase:
                specific_page_number = page_num + 1  # Page numbers start from 0
                break

    if specific_page_number is not None:
        return specific_page_number
    else:
        print(f"Did not find '{search_word}' in the PDF")

#**********************************************************************

# Fonction pour extraire les sections du PDF

def split_sections(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)

        bilan_start_page = Find_Page(pdf_path, "bilan", "état de variation")
        etat_resultat_start_page = Find_Page(pdf_path, "etat de resultat", "bilan")
        etat_variation_start_page = Find_Page(pdf_path, "etat de variation", "bilan")
        notes_start_page = etat_variation_start_page + 1

        bilan_writer = PdfWriter()
        etat_resultat_writer = PdfWriter()
        etat_variation_writer = PdfWriter()
        notes_writer = PdfWriter()

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            if page_num >= bilan_start_page and (not etat_resultat_start_page or page_num < etat_resultat_start_page):
                bilan_writer.add_page(page)
            elif etat_resultat_start_page and page_num >= etat_resultat_start_page and (not etat_variation_start_page or page_num < etat_variation_start_page):
                etat_resultat_writer.add_page(page)
            elif etat_variation_start_page and page_num >= etat_variation_start_page and (not notes_start_page or page_num < notes_start_page):
                etat_variation_writer.add_page(page)
            elif notes_start_page and page_num >= notes_start_page:
                notes_writer.add_page(page)

        with open("bilan.pdf", "wb") as output_file:
            bilan_writer.write(output_file)
        with open("etat_resultat.pdf", "wb") as output_file:
            etat_resultat_writer.write(output_file)
        with open("etat_variation.pdf", "wb") as output_file:
            etat_variation_writer.write(output_file)
        with open("notes.pdf", "wb") as output_file:
            notes_writer.write(output_file)

    print("PDFs extracted and saved successfully.")

#**********************************************************************
#Main function to extract text from PDF

def extract_text_from_pdf(pdf_path):
    split_sections(pdf_path)
    images = convert_from_path("bilan.pdf")
    for i in range(len(images)):
        images[i].save('pages/page'+str(i)+'.jpg', 'JPEG')
    image = cv2.imread('pages/page0.jpg')
    image = image[..., ::-1]

    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                    threshold=0.5,
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                    enforce_cpu=False,
                                    enable_mkldnn=True)

    layout = model.detect(image)

    x_1=0
    y_1=0
    x_2=0
    y_2=0

    for l in layout:
        if l.type == 'Table' or l.type == 'Figure':
            x_1 = int(l.block.x_1)
            y_1 = int(l.block.y_1)
            x_2 = int(l.block.x_2)
            y_2 = int(l.block.y_2)
            break

    im = cv2.imread('pages/page0.jpg')       
    cv2.imwrite('ext_im.jpg', image[y_1:y_2,x_1:x_2])

    im = Image.open("ext_im.jpg")
    enhancer = ImageEnhance.Brightness(im)

    factor = 1.5
    im_output = enhancer.enhance(factor)
    im_output.save('ext_im-2.jpg')

    ocr = PaddleOCR(lang='en')
    image_path = 'ext_im-2.jpg'
    image_cv = cv2.imread(image_path)
    image_height = image_cv.shape[0]
    image_width = image_cv.shape[1]
    output = ocr.ocr(image_path)[0]

    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    image_boxes = image_cv.copy()

    for box,text in zip(boxes,texts):
        cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
        cv2.putText(image_boxes, text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

    cv2.imwrite('detections.jpg', image_boxes)

    im = image_cv.copy()

    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0,int(box[0][0])
        y_h, y_v = int(box[0][1]),0
        width_h,width_v = image_width, int(box[2][0]-box[0][0])
        height_h,height_v = int(box[2][1]-box[0][1]),image_height

        horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

        cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
        cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)

    cv2.imwrite('horiz_vert.jpg',im)

    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size = 1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    horiz_lines = np.sort(np.array(horiz_out))

    im_nms = image_cv.copy()

    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)

    cv2.imwrite('im_nms.jpg',im_nms)

    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size = 1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None

    )

    vert_lines = np.sort(np.array(vert_out))

    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)

    cv2.imwrite('im_nms.jpg',im_nms)

    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

    unordered_boxes = []

    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])

    ordered_boxes = np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                if(iou(resultant,the_box)>0.1):
                    out_array[i][j] = texts[b]
    out_array=np.array(out_array)
    print(out_array)
    pd.DataFrame(out_array).to_csv('sample.csv')

#**********************************************************************


# Fonction pour calculer la similarité entre deux colonnes
def compute_similarity(col1, col2):
    return fuzz.partial_ratio(str(col1), str(col2))

#**********************************************************************

# Fonction pour fusionner les colonnes
def merge_columns():
    # Charger le fichier CSV
    df = pd.read_csv('sample.csv', index_col=0)

    # Seuil de similarité pour la fusion des colonnes
    seuil_similarity = 60

    # Fusionner les colonnes similaires
    colonnes_fusionnees = [df.iloc[:, 0]]

    for i in range(1, len(df.columns)):
        colonne_actuelle = df.iloc[:, i]
        fusionnee = False
        for j, colonne_fusionnee in enumerate(colonnes_fusionnees):
            sim = compute_similarity(colonne_actuelle, colonne_fusionnee)
            if sim > seuil_similarity:
                # Fusionner la colonne actuelle avec la colonne fusionnée
                colonnes_fusionnees[j] = colonne_fusionnee.combine_first(colonne_actuelle)
                fusionnee = True
                break
        if not fusionnee:
            colonnes_fusionnees.append(colonne_actuelle)
    
    # Créer un nouveau DataFrame avec les colonnes fusionnées
    df_fusionne = pd.concat(colonnes_fusionnees, axis=1)
    # Enregistrer le DataFrame fusionné au format CSV
    df_fusionne.to_csv('sample_fusionne.csv')

#**********************************************************************


# Fonction pour convertir le fichier CSV en fichier JSON
def convert_to_json():
    # Read Excel file into a DataFrame
    df = pd.read_csv('sample_fusionne.csv', index_col=0)

    json_data = {}
    keys = df.iloc[0].tolist()
    for index, row in df.iloc[1:].iterrows():
        data = {}
        for i, val in enumerate(row):
            if pd.notna(val):
                data[keys[i]] = val
        json_data[str(index)] = data

    # Save JSON data to a file
    with open('bilan.json', 'w') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


#**********************************************************************

# Fonction pour extraire les données du bilan en text
def text_pdf(pdf):
    text=""
    with open(pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
                text +=page.extract_text()
        return text
    

#**********************************************************************

# Fonction pour savoir des pages specifiques

def extract_specific_pages(pdf_path):
    # Initialize a variable to track if the word is found
    found = False
    specific_page_number = None

    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Iterate over each page in the PDF file
        for page_num, page in enumerate(pdf_reader.pages):
            # Get the text from the current page
            page_text = page.extract_text()

            # Check if "BILAN" is present in the text
            if "bilan" in page_text.lower() and "31/" in page_text:
                found = True
                specific_page_number = page_num + 1  # Page numbers start from 0
                break

    if found:
        print(f"Found 'Etat de variation' on page {specific_page_number}")
    else:
        print("Did not find 'Etat de variation' in the PDF")
    
#**********************************************************************

# Fonction pour manipuler les notes

def handle_notes(pdf_path):
    output_path = "output.pdf"
    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Find the page number of the page to delete
        page_to_delete = None
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if "etat de variation" in page_text.lower() or "  NOTES AUX ETATS FINANCIERS" in page_text.upper() :
                page_to_delete = page_num
                break


        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()

        # Add pages after the page to delete to the PDF writer
        for page_num, page in enumerate(pdf_reader.pages):
            if page_to_delete is not None and page_num > page_to_delete:
                pdf_writer.add_page(page)



        # Write the output PDF file
        with open(output_path, "wb") as output_pdf_file:
            pdf_writer.write(output_pdf_file)

    if page_to_delete is not None:
        print(f"Deleted page {page_to_delete + 1} and all pages before it")
    else:
        print("No page to delete found")
#**********************************************************************


def notes_json():
    # Assuming 'text' contains the extracted text from the PDF
    text = text_pdf('output.pdf')
    titles, segments = extract_titles_and_segments(text)

    # Create a dictionary with titles as keys and segments as values
    data = {}
    for title, segment in zip(titles, segments):
        data[title] = segment

    # Save the data to a JSON file
    with open('notes.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)       


#**********************************************************************

# Fonction pour extraire les titres et les segments du texte

def extract_titles_and_segments(text):
    titles = []
    segments = []
    current_title = None
    current_segment = ""
    lines = text.split("\n")
    for line in lines:
        # Check if the line matches the title pattern
        if re.match(r'^\d+(\.|\-)\s*[a-zA-Z]*', line):
            # If we already have a current_title, save the previous segment
            if current_title is not None:
                segments.append(current_segment.strip())
            # Update the current_title and start a new segment
            current_title = line.strip()
            titles.append(current_title)
            current_segment = ""
        else:
            # Append the line to the current segment
            current_segment += line + "\n"

    # Append the last segment
    if current_title is not None:
        segments.append(current_segment.strip())

    return titles, segments

#**********************************************************************

# Fonction pour matcher les NOTES avec le bilan

def match_notes_with_balance_sheet():
    # Charger le modèle et le tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

    # Charger les données de 'bilan.json'
    with open('bilan.json', 'r') as bilan_file:
        bilan_data = json.load(bilan_file)

    # Charger les données de 'notes.json'
    with open('notes.json', 'r') as notes_file:
        notes_data = json.load(notes_file)

    # Parcourir les éléments de bilan_data et remplacer les valeurs de la clé 'NOTES'
    for key, value in bilan_data.items():
        if 'NOTES' in value:
            note_key = value['NOTES']

            # Utiliser une comparaison souple pour rechercher la clé correspondante dans notes_data
            found = False
            for note_title, note_content in notes_data.items():
                if note_key.lower() in note_title.lower():
                    value['NOTES'] = note_content

                    # Prétraitement du texte pour le résumé
                    input_text = "summarize: " + value['NOTES']
                    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=150, truncation=True)

                    # Génération du résumé
                    summary_ids = model.generate(inputs, max_length=150, min_length=10, length_penalty=10.0, num_beams=4, early_stopping=True)

                    # Décodage du résumé et mise à jour du champ 'NOTES'
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    value['NOTES'] = summary

                    found = True
                    break

            # Si aucune correspondance n'est trouvée, remplacer par une chaîne vide
            if not found:
                value['NOTES'] = ""

    # Sauvegarder les données mises à jour dans 'bilan.json'
    with open('bilan.json', 'w') as bilan_file:
        json.dump(bilan_data, bilan_file, indent=4)