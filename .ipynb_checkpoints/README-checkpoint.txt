This is the final code for the Trustly challenge. Here goes a quick list:


- To run locally the best idea is to create an separeted env and use requirements.txt to guarantee all dependencies are installed in your system. After doing  it
    - Run: uvicorn app:app --reload to start a instance (nohup python3 -m uvicorn app:app --reload --host 0.0.0.0 if you need you terminal clean)
    - Go to http://127.0.0.1:8000/docs to see API documentation and an interface to interact with json payload
    
- To execute the same as a Docker container:
    - Run: docker build . -t sklearn_fastapi_docker
    - Start the instance with: docker run -p 80:80 sklearn_fastapi_docker  
    - Go to localhost/docs to see the Swagger API documentation provided by FastAPI.
    
There are two json files in the directory with different sizes.

A suggestion for a curly call would be:
curl -X 'POST' \
  'http://127.0.0.1:8000/uploadfiles/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@input_full.json;type=application/json'