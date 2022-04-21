from flask import Flask 
from flask_socketio import SocketIO, send
import logging
import logging

log = logging.getLogger('werkzeug')
log.disabled = True
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_to_Enc_messages'
socketio = SocketIO(app, cors_allowed_origins='*')

@socketio.on('message')
def handleMessage(msg):
	send(msg, broadcast=True)

if __name__ == '__main__':
	socketio.run(app)