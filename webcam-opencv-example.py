import cv2
import urllib3
import numpy as np
import requests
host = "192.168.99.128:8081"
userName = "admin"
passWord  = "admin"
hoststr = 'http://'+userName+":"+passWord+'@' + host + '/video'
# stream=urllib3.urlopen(hoststr)
http = urllib3.PoolManager()
response = requests.get(hoststr, stream=True)
bytes=bytes()
while True:
    bytes+=response.raw.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        cv2.imshow('i',i)
        if cv2.waitKey(1) ==27:
            exit(0)    