import io
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.http import MediaIoBaseDownload

def check_hw (service, file_ids,folder_id):
    for fid in file_ids:
        # you can get the meta data to see what is the file type or just download it
        file_metadata = service.files().get(fileId=fid, fields='mimeType, name').execute()
        request = service.files().get_media(fileId=fid)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
    
    # here is some proccesing of the files

    file_metadata = {
        'name': 'We_should_think_of_naming_system.txt',
        'parents': folder_id
        #'mimeType': 'text/plain' depedns on how we create the files 
    }
       
    
    # Execute the request combining metadata and content
    file = service.files().create(
        body=file_metadata,
        media_body=None,
        fields='id'
    ).execute()
