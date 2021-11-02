from djitellopy import tello
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

# cap = cv2.VideoCapture(0)
detectorHand = HandDetector(maxHands=1, detectionCon=0.9)
detectorFace = FaceDetector()
gesture = ""

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
    img = detectorHand.findHands(img)
    lmList, bboxInfo = detectorHand.findPosition(img)
    img, bboxs = detectorFace.findFaces(img, draw=True)

    if bboxs:
        x, y, w, h = bboxs[0]["bbox"]
        bboxRegion = x - 175 - 25, y - 75, 175, h + 75
        cvzone.cornerRect(img, bboxRegion, rt=0, t=10, colorC=(0, 0, 255))

        if lmList and detectorHand.handType() == "Right":

            handCenter = bboxInfo["center"]
            #       x < cx < x+w
            inside = bboxRegion[0] < handCenter[0] < bboxRegion[0] + bboxRegion[2] and \
                     bboxRegion[1] < handCenter[1] < bboxRegion[1] + bboxRegion[3]

            if inside:
                cvzone.cornerRect(img, bboxRegion, rt=0, t=10, colorC=(0, 255, 0))

                fingers = detectorHand.fingersUp()
                # print(fingers)

                if fingers == [1, 1, 1, 1, 1]:
                    gesture = "  Stop"
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = "  UP"
                    me.move_up(20)
                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "Flip"
                    me.flip_left()
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = " Down"
                    me.move_down(20)
                elif fingers == [0, 0, 0, 0, 1]:
                    gesture = "  Left"
                    me.move_left(40)
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "  Right"
                    me.move_right(40)

                cv2.rectangle(img, (bboxRegion[0], bboxRegion[1] + bboxRegion[3] + 10),
                              (bboxRegion[0] + bboxRegion[2], bboxRegion[1] + bboxRegion[3] + 60),
                              (0, 255, 0), cv2.FILLED)

                cv2.putText(img, f'{gesture}',
                            (bboxRegion[0] + 10, bboxRegion[1] + bboxRegion[3] + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        me.land()
        break
cv2.destroyAllWindows()
