import os
import tempfile
import urllib.request
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.response import Response

from ingestion.pipeline import ingestion_pipeline


@api_view(['POST'])
@parser_classes([MultiPartParser, JSONParser])
def ingest_view(request):
    """
    Ingest a document via file upload or URL.
    POST body (multipart/form-data):
      - file: Upload file (pdf/docx/txt/csv/image)
      - url: String URL to ingest
      - type: Optional, force source type (pdf, docx, csv, text, web, image)
    """
    source = None
    source_type = request.data.get('type')

    # Handle file upload
    if 'file' in request.FILES:
        uploaded = request.FILES['file']
        # Save temporarily to disk
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            source = tmp.name

    # Handle URL ingestion
    elif 'url' in request.data:
        source = request.data['url']

    else:
        return Response(
            {'error': 'Provide either "file" or "url" in the request.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Run ingestion pipeline
    result = ingestion_pipeline.process_source(source, source_type)

    # If we wrote a temp file, clean up
    if os.path.isfile(source) and source.startswith(tempfile.gettempdir()):
        try:
            os.remove(source)
        except Exception:
            pass

    if result.get('success'):
        return Response(result, status=status.HTTP_200_OK)
    else:
        return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
