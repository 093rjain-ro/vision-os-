import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        try:
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.wfile.write(f"<html><body><h1>Vision OS Landing Page</h1><p>Error loading HTML: {e}</p></body></html>".encode('utf-8'))
