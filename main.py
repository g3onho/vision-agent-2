import cv2
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound
from PyQt5.QtCore import QTimer
import os

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("교통약자 보호")
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton("표지판 등록", self)
        roadButton = QPushButton("도로 영상 불러옴", self)
        # recognitionButton = QPushButton("인식", self)
        quitButton = QPushButton("나가기", self)
        self.label = QLabel("환영합니다!", self)

        signButton.setGeometry(10, 10, 150, 30)
        roadButton.setGeometry(170, 10, 200, 30)
        # recognitionButton.setGeometry(380, 10, 100, 30)
        quitButton.setGeometry(520, 10, 100, 30)
        self.label.setGeometry(20, 40, 600, 100)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        # recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [
            [resource_path("child.png"), "어린이"],
            [resource_path("elder.png"), "노인"],
            [resource_path("disabled.png"), "장애인"],
        ]  # 표지판 모델 영상
        self.signImgs = []  # 표지판 모델 영상 저장

    def signFunction(self):
        self.label.clear()
        self.label.setText("교통약자 번호판을 등록합니다.")

        for fname, _ in self.signFiles:
            img = cv2.imread(fname)
            if img is not None:
                self.signImgs.append(img)
                cv2.imshow(fname, img)
            else:
                print(f"이미지를 로드할 수 없습니다: {fname}")

        #1초 뒤에 창 닫기        
        QTimer.singleShot(1000, cv2.destroyAllWindows)

    def roadFunction(self):
        if not self.signImgs:
            self.label.setText("먼저 번호판을 등록하세요.")
        else:
            fname, _ = QFileDialog.getOpenFileName(self, "파일 읽기", "./", "All Files (*.*);;Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.avi)")
            if not fname:
                return  # 파일 선택이 취소되었을 경우 종료
            
            print(f"선택한 파일 경로: {fname}")  # 선택한 파일 경로를 출력
            
            file_ext = fname.split('.')[-1].lower()  # 파일 확장자 확인
            if file_ext in ['png', 'jpg', 'jpeg']:  # 이미지 파일 처리
                self.roadImg = cv2.imread(fname)
                if self.roadImg is None:
                    self.label.setText("이미지를 불러올 수 없습니다.")
                    return
                self.recognitionFunction()
            elif file_ext in ['mp4', 'avi']:  # 동영상 파일 처리
                cap = cv2.VideoCapture(fname)
                if not cap.isOpened():
                    self.label.setText("동영상을 불러올 수 없습니다.")
                    return

                frame_interval = 5
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # 동영상이 끝나면 종료
                    
                    if frame_count % frame_interval == 0:
                        self.roadImg = frame
                        self.recognitionFunction()  # 프레임마다 표지판 인식

                    cv2.imshow("Video Scene", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌러 종료
                        break

                    frame_count += 1    

                cap.release()
                cv2.destroyAllWindows()
            else:
                self.label.setText("지원하지 않는 파일 형식입니다.")


    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText("먼저 도로 영상을 입력하세요.")
        else:
            sift = cv2.SIFT_create()

            KD = []  # 여러 표지판 영상의 키포인트와 기술자 저장
            for img in self.signImgs:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))

            grayRoad = cv2.cvtColor(self.roadImg, cv2.COLOR_BGR2GRAY)  # 명암으로 변환
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)  # 키포인트와 기술자 추출

            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
            GM = []  # 여러 표지판 영상의 good match를 저장
            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, 2)
                T = 0.7
                good_match = []
                for nearest1, nearest2 in knn_match:
                    if (nearest1.distance / nearest2.distance) < T:
                        good_match.append(nearest1)
                GM.append(good_match)

            best = GM.index(max(GM, key=len))  # 매칭 쌍 개수가 최대인 번호판 찾기

            if len(GM[best]) < 4:  # 최선의 번호판이 매칭 쌍 4개 미만이면 실패
                self.label.setText("표지판이 없습니다.")
            else:  # 성공(호모그래피 찾아 영상에 표시)
                sign_kp = KD[best][0]
                good_match = GM[best]

                points1 = np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                points2 = np.float32([road_kp[gm.trainIdx].pt for gm in good_match])

                H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

                h1, w1 = (
                    self.signImgs[best].shape[0],
                    self.signImgs[best].shape[1],
                )  # 번호판 영상의 크기
                h2, w2 = self.roadImg.shape[0], self.roadImg.shape[1]  # 도로 영상의 크기

               # box1 정의 (4개의 점을 포함하는 형태)
                box1 = np.float32(
                    [[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]
                ).reshape(4, 1, 2)
                
                # H가 유효한지 확인
                if H is None:
                    self.label.setText("호모그래피를 찾을 수 없습니다.")
                    return  # 호모그래피가 유효하지 않은 경우 종료
                box2 = cv2.perspectiveTransform(box1, H)

                self.roadImg = cv2.polylines(
                    self.roadImg, [np.int32(box2)], True, (0, 255, 0), 4
                )

                img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                cv2.drawMatches(
                    self.signImgs[best],
                    sign_kp,
                    self.roadImg,
                    road_kp,
                    good_match,
                    img_match,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.imshow("Matches and Homography", img_match)

                self.label.setText(self.signFiles[best][1] + " 보호구역입니다. 30km로 서행하세요.")
                winsound.Beep(3000, 500)

    def quitFunction(self):
        cv2.destroyAllWindows()
        self.close()

if __name__=="__main__":
    app = QApplication(sys.argv)
    win = TrafficWeak()
    win.show()
    app.exec_()