from flask import Flask, render_template, Response
from flask import  send_file

import cv2
import os
import logging
app = Flask(__name__)
Frame_No=1
log = logging.getLogger('werkzeug')
log.disabled = True
all_ids_set=set(["3","5"])
print(all_ids_set)


def gen_frames(id): 
    while True:
        try:
            all_ids_set.add(id)
            filename_to_be_send=f'{id}-{str(Frame_No)}.jpg'
            filename_to_be_send=os.path.join(r"D:\BEFINAL\datastream\videoserver",filename_to_be_send)

            image = cv2.imread(filename_to_be_send)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        except:pass


@app.route('/video_feed/<id>')
def video_feed(id):
    return Response(gen_frames(id), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/req_frame/<id>' , methods=['GET'])
def req_frame(id):
    global Frame_No
    if id!=Frame_No:

        for i in all_ids_set:
            pass
    Frame_No=id
    return Response({"frame_no":Frame_No})


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/tfjs/<path>')
def send_js(path):
    print(path)
    return send_file(r"D:\BEFINAL\VideoStreamingServer\static\tfjs"+"\\" + path)
if __name__ == '__main__':
    app.run(debug=True,port=5100)
