from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .notebook import *
import os
from .utils import connect_to_mongodb
from django.http import JsonResponse
from django.shortcuts import redirect
from django.http import HttpResponse
from .chatbot import get_response


def upload_pdf(request):
    if request.method == 'POST' and request.FILES['pdf_file']:
        pdf_file = request.FILES['pdf_file']
        fs = FileSystemStorage()
        filename = fs.save(pdf_file.name, pdf_file)

        # Process the uploaded PDF file
        pdf_path = os.path.join(settings.MEDIA_ROOT, filename)
        extract_text_from_pdf(pdf_path)
        merge_columns()
        convert_to_json()
        handle_notes(pdf_path)
        notes_json()
        match_notes_with_balance_sheet()
        
        # Remove the .pdf extension from the filename
        filename_without_extension = os.path.splitext(filename)[0]

         # Save the JSON file with the name of the PDF file in MongoDB
        client = connect_to_mongodb()
        db = client['BFI']
        collection = db['data']
        with open('bilan.json', 'r') as file:
            json_data = json.load(file)
            json_data['_id'] = filename_without_extension
            collection.insert_one(json_data)
        client.close()  # Close the MongoDB connection
        return redirect('show_pdf_json')  # Redirect to the 'pdfs' URL

    return render(request, 'upload.html')





def show_pdf_list(request):
    client = connect_to_mongodb()
    db = client['BFI']
    collection = db['data']

    # Convert the documents to a list for easier processing
    documents = collection.find()

    return render(request, 'result.html', {'documents': documents})


def show_home(request):
    client = connect_to_mongodb()
    db = client['BFI']
    collection = db['data']

    # Convert the documents to a list for easier processing
    documents = collection.find()

    return render(request, 'index-2.html', {'documents': documents})


def download_json(request, pdf_filename):
    client = connect_to_mongodb()
    db = client['BFI']
    collection = db['data']

    document = collection.find_one({'_id': pdf_filename})
    if document:
        return JsonResponse(document, json_dumps_params={'indent': 4})
    else:
        return JsonResponse({'error': 'Document not found'}, status=404)
    

def delete_document(request, document_id):
    client = connect_to_mongodb()
    db = client['BFI']
    collection = db['data']

    # Delete the document from the database
    result = collection.delete_one({'_id': document_id})

    client.close()  # Close the MongoDB connection

    if result.deleted_count > 0:
        return HttpResponse(status=204)  # No content (successful deletion)
    else:
        return HttpResponse(status=404)  # Document not found
    



def chatbot_view(request):
    if request.method == 'GET':
        user_input = request.GET.get('msg')
        response = get_response(user_input)
        return JsonResponse({'response': response})
    else:
        return JsonResponse({'error': 'Invalid request method'})