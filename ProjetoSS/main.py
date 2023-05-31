# Importando as bibliotecas necessárias
import cv2 as cv
import methods
# A main utilizada para chamar os métodos da aplicação

camera = cv.VideoCapture(0)  # URL da Webcam da máquina
# camera = cv.VideoCapture('http://192.168.18.170:4747/video')  # URL da Camera do telefone
ret1, img1 = camera.read()
methods.selecionar_area_para_monitoramento(img1)
methods.monitorar_area(camera)
methods.resgatar_data_hora(img1)
camera.release()
cv.destroyAllWindows()
