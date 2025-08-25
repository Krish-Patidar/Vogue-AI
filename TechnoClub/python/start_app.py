# start_app.py
from visual_search_app import app
import socket

def check_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0
    except:
        return False

def find_available_port(start_ports=[8000, 8080, 3000, 5001, 8888, 9000]):
    for port in start_ports:
        if check_port('127.0.0.1', port):
            return port
    return None

if __name__ == '__main__':
    port = find_available_port()
    if port:
        print(f"🚀 Starting Visual Search Application on port {port}...")
        app.run(debug=True, host='127.0.0.1', port=port)
    else:
        print("❌ Could not find an available port — please run visual_search_app.py manually.")
