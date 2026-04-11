#import das biblotecas
import cv2 #opencv -> lib responsavel pelo gerencimento de dispositivos vis. comp.
from ultralytics import YOLO #lib responsavel pelo reconhecimento facial / ob

#Passo 1: carragamento do modelo
print("Carregando o modelo...")
model =YOLO('yolov8n.pt') #yolov8n é uma versão nano, mais leve/rápida

#2. Abrir uma conexão com webcam
cap = cv2.VideoCapture(0)
# número 0 reoresenta uma webcam integrada ao computador
# número 1 representa uma webcan conectada via USB (via fisica)
# caso a via seja remota, o enderenço de ip deve ser informado

#Verifica se a camera abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir a camera")
    exit()

print("Iniciando a detecção.Pressione 'q' para sair")

#Passo 3: Iniciar a leitura das detecções 
while True:
    sucesso,frame = cap.read() #ler os frames (imagens) da camera

    if sucesso: #realizar a detecção (inference)
       results = model(frame, conf=0.5) #queremos detecções com 50% ou mais de certeza
       anotaded_frame = results[0]. plot() #criar caixa visual na imagem
       cv2.imshow("Visão Computacional - YOLOv8", anotaded_frame)

       if cv2.waitKey(1) & 0xFF == ord('q'): #pressionar q para sair
            break

    else:
        break
#limpeza
cap.release()
cv2.destroyAllWindows()


