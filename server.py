import os
import datetime
import http.server
import socketserver
import signal
import argparse
import mimetypes
from urllib.parse import urlparse
import sys
import json
import re
#import nerf_api
ROOT_DIR = './server'

def guess_mimetype(path):
    #default.
    mime_type = mimetypes.guess_type(path)[0]
    if path.endswith('.ply'):
        mime_type = 'text/plain'
    return mime_type


class ServerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        #discuss, as path always start with "/"
        if self.path=='/':
            self.path = "/index.html"
        #response api
        try:
            file_path = urlparse(self.path).path
            if(file_path=='/'):
                file_path = "/index.html"
            mime_type = guess_mimetype(file_path)
            if mime_type:
                file_path = file_path[1:] if file_path[0]=='/' else file_path
                with open(os.path.join(ROOT_DIR,file_path),'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type',mime_type)
                    self.end_headers()
                    self.wfile.write(f.read())
        except IOError:
            self.send_error(404,'File Not Fould {}'.format(self.path))
    
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
    def do_POST(self):
        try:
            api_path = urlparse(self.path).path
            data_string = self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.end_headers()
            data = json.loads(data_string.decode())
            print(data)
            #api_name = api_path.split('/')[-1]
            #ret = getattr(nerf_api,api_name)(data)
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        except IOError:
            self.send_error(404,'Unknown API {}'.format(self.path))
 
    #silent
    def log_message(self,format,*args):
        return



def parse_args():
    file_path = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--port',
        dest='port',
        default=16006,
        type=str
    )
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()

    socketserver.ThreadingTCPServer.allow_reuse_address = True

    server = socketserver.ThreadingTCPServer(('',args.port),ServerHandler)
    server.daemon_threads = True

    def signal_handler(signal,frame):
        print('Shutting down NeRFServer(Ctrl+C Pressed)')
        try:
            if(server):
                server.server_close()
        finally:
            exit(0)

    signal.signal(signal.SIGINT,signal_handler)
    print('starting NeRFServer on port:',args.port)

    try:
        while True:
            sys.stdout.flush()
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()