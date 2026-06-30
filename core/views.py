from django.shortcuts import render, redirect
from .models import Document

def upload_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        uploaded_file = request.FILES['document']
        title = request.POST.get('title', uploaded_file.name)
        
        Document.objects.create(
            title=title,
            file=uploaded_file
        )
        return redirect('document_list') 
        
    return render(request, 'core/upload.html')