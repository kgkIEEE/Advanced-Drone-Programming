from djitellopy import tello
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector(minDetectionCon=0.5)

# cap = cv2.VideoCapture(0)
# _, img = cap.read()
hi, wi, = 480, 640
# print(hi, wi)
#                   P   I  D
xPID = cvzone.PID([0.22, 0, 0.1], wi // 2)
yPID = cvzone.PID([0.27, 0, 0.1], hi // 2, axis=1)
zPID = cvzone.PID([0.005, 0, 0.003], 12000,limit=[-20,15])

myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()
me.takeoff()
me.move_up(80)

while True:
    # _, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (640, 480))
    img, bboxs = detector.findFaces(img, draw=True)

    xVal = 0
    yVal = 0
    zVal = 0

    if bboxs:
        cx, cy = bboxs[0]['center']
        x, y, w, h = bboxs[0]['bbox']
        area = w * h

        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        zVal = int(zPID.update(area))
        # print(zVal)
        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        img = xPID.draw(img, [cx, cy])
        img = yPID.draw(img, [cx, cy])
        # imgStacked = cvzone.stackImages([img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)
        imgStacked = cvzone.stackImages([img], 1, 0.75)
        # Display Area
        #cv2.putText(imgStacked, str(area), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    else:
        imgStacked = cvzone.stackImages([img], 1, 0.75)

    me.send_rc_control(0, -zVal, -yVal, xVal)
    #me.send_rc_control(0, -zVal, 0, 0)
    cv2.imshow("Image Stacked", imgStacked)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        me.land()
        break
cv2.destroyAllWindows()
