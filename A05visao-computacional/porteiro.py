#import das libs
import cv2
from deepface import DeepFace
import time

#passo 1: carregar a identidade (Cadastramento)
imagem_referencia = "face_id.jpg"
print("Carregando identidade do morador.")

#Pré-analise da imagem, pra garantir que foto de referencia é válida
try:
    DeepFace.represent(img_path= imagem_referencia, model_name="VGG-Face")
    print("Identidade carregada com sucesso.")
except:
    print("Erro! Não encontrei o arquivo ou não há rosto nele.")
    exit()

#Iniciar a camera 
cap = cv2. VideoCapture(0) #o número 0 indica que a camera está intergrada ao computador
print ("Sistema de portaria ativo.")

while True:
    ret, frame = cap.read() #ret retorna true se a foto foi tirada, frame recebe a iamgem
    if not ret: break

    frame_small = cv2.resize(frame,(0,0), fx= 0.5, fy=0.5)

    #desenhar um retangulo pra indicar a área de leitura
    height, width, _ = frame.shape
    cv2.rectangle(frame,(100,100), (width-100, height-100), (255,0,0),2)
    #tamanho, com espessura da linha 

    #Verificação da imagem com o rosto detectado
    cv2.putText(frame, "Pressione V para verificaro acesso", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('v'):
        print("Verificando identidade")
        try:
            resultado = DeepFace.verify(
                img1_path = frame, # quem está´na camera
                img2_path = imagem_referencia, #foto capturada
                model_name= "VGG_Face",
                enforce_detection = False #tente detectar o rosto, mas se não for possível não trave o programa

            )

            #se resultado é verdadeiro, (acesso liberado)
            if resultado ['verified']:
                print(">>>>ACESSO LIBERADO!>>>>>")
                cv2.rectangle(frame,(0,0), (width, height), (0,255,0), 2)
                cv2.imshow("Portaria", frame)
                cv2. waitKey(2000) #pausa por 2s para mostrar a borda verde 
            else:
                print(">>>>ACESSO NEGADO!>>>>>")
                cv2.rectangle(frame,(0,0),(width, height), (0,0,255), 2)
                cv2.imshow("Portaria", frame)
                cv2.waitKey(2000)

        except Exception as e:
            print(f"Erro na leiura: {e}")

    cv2.imshow("Portaria", frame)

    if key & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()