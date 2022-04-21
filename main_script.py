from queue import Empty
from subprocess import Popen
import time
import os
import subprocess
import moviepy
import moviepy.editor

l =  ["close_prox","come_close","intrested","notintrested","talking","walk_by_looking","walk_by_not_looking","walk_by_very"]
for i in [ "testset"]:
    video_name=i
    video = moviepy.editor.VideoFileClip(f"D:\BEFINAL\seperatedatastreams\yolov3_deepsort\data\\video\\{video_name}.mp4")
    video_duration = int(video.duration)
    print(video_duration*1000)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    video_server= Popen(['python', 'app.py'],cwd="D:\BEFINAL\VideoStreamingServer")
    socket_server = Popen(['python', 'flask_script.py'],cwd="D:\BEFINAL\Flask_Socket")
    alu = Popen(['python', 'seperatedatastreams\yolov3_deepsort\object_tracker.py',f"--video=D:\BEFINAL\seperatedatastreams\yolov3_deepsort\data\\video\\{video_name}.mp4"]) 
    time.sleep(10)
    os.system("start http://localhost:5100/")
    deep=Popen(['python', 'script.py'], cwd="D:\BEFINAL\deep-model") 
    orient= Popen(['python', 'head_pose_estimation_image.py'], cwd="D:\BEFINAL\head-pose-angle-detector") 
    emo =Popen(['python', 'emotion_driver.py'] ) 
    time.sleep(video_duration*8+10)
    print("started killing")
    alu.kill()
    video_server.kill()
    socket_server.kill()
    deep.kill()
    orient.kill()
    emo.kill()
    print("killed")
    # os.system("taskkill /F /IM chrome.exe")
