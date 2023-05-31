# Importando as bibliotecas necessárias
import statistics
from datetime import datetime
from time import time
import cv2 as cv
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
from scipy.stats import norm
from PIL import Image, ImageFilter
from matplotlib.image import imread


# Identificar o evento do mouse em algum local da imagem
# RefPt é a variável de ponto de referência
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # Se o botão esquerdo do mouse for pressionado, o corte é dado como verdadeiro
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # Se o botão esquerdo do mouse não for pressionado, é encerrada a operação de corte, e exibe a área de referência
    elif event == cv.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv.rectangle(img3, refPt[0], refPt[1], (0, 0, 255), 2)
        cv.imshow("Escolha a regiao a ser monitorada e aperte C", img3)

# Aplicando o filtro passa baixas na imagem recortada em escala de cinza
# E subtraindo pela imagem com a passa baixa
def filtros_genericos(roi):
    blur = cv.blur(roi, (5, 5))
    sub = cv.subtract(roi, blur)
    return sub

# Aplicando o limiar na imagem com os filtros genéricos e contando os pixels brancos da imagem limiarizada
def filtros_imagem1(roi):
    ret, thresh1 = cv.threshold(filtros_genericos(roi), 15, 255, cv.THRESH_BINARY)
    area = np.sum(thresh1 == 255)
    return area

# Aplicando o laplaciano na imagem com os filtros genéricos,
# Aplicando o limiar no laplaciano e contando os pixels brancos da imagem resultante
def filtros_imagem2(roi):
    laplacian = cv.Laplacian(filtros_genericos(roi), cv.CV_64F)
    ret, thresh2 = cv.threshold(laplacian, 15, 255, cv.THRESH_BINARY)
    perimetro = np.sum(thresh2 == 255)
    return perimetro

def margem_erro(iteracoes, desvioPadrao):
    alfa = 0.05
    z = norm.ppf(1-alfa/2,0,1)
    sx = desvioPadrao/((iteracoes)**(0.5))
    return z*sx

def fft(roi):

    f = imageio.imread(roi, pilmode='L')

    # mostrando a imagem
    colormap = 'jet'  # mapa de cores

    fig = plt.figure(figsize=(12, 5))

    ax_img = fig.add_subplot(131)
    ax_img.set_title('Imagem Monitorada')
    ax_img.imshow(f, cmap=colormap)

    # Calculando a FFT 2D
    F = fp.fft2(f)
    Fm = np.absolute(F)
    Fm /= Fm.max()  # normalizando para obter amplitudes em [0, 1]
    Fm = np.log(Fm)  # logaritmo para conseguir enxargar a FFT!

    # mostrando a  |FFT|
    ax_fft = fig.add_subplot(132)
    ax_fft.set_title('log |FFT|')
    ax_fft.imshow(Fm, cmap=colormap)  # vmax é o valor máximo a ser plotado

    # fazendo o shift para obter a |FFT| como deve ser
    Fm = fp.fftshift(Fm)
    ax_fftshift = fig.add_subplot(133)
    ax_fftshift.set_title('FFT com shift')
    ax_fftshift.imshow(Fm, cmap=colormap, vmax=.2)

    plt.tight_layout()
    plt.show()

def fft2(path):
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams.update({'font.size': 18})

    A = cv.imread(path)
    B = np.mean(A, -1);  # Converte para escala de cinza

    # Computando a fft2 do gatinho
    D = np.fft.fft2(B)

    plt.rcParams['figure.figsize'] = [15, 10]

    # Criação de um subplot 3d para a visualização da imagem em termos de intensidade de pixel
    fig = plt.figure()

    # Vale dizer que esta segunda parte é desnecessária se for executar o código em um notebook,
    # nele o plot é interativo
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(1, np.shape(B)[1] + 1), np.arange(1, np.shape(B)[0] + 1))
    ax.plot_surface(X[0::10, 0::10], Y[0::10, 0::10], B[0::10, 0::10], cmap='inferno', edgecolor='none')
    ax.set_title('Plot de superfície')
    ax.mouse_init()
    ax.view_init(200, 270)  # Visão superior
    plt.show()


# Método de seleção da área de monitoramento
def selecionar_area_para_monitoramento(img1):
    # Modificando acesso da imagem 1, atribuindo valor a uma nova variável, para que todos os metodos tenham acesso
    global img3
    img3 = img1
    refPt = []
    cropping = False
    clone = img1.copy()
    cv.namedWindow("Escolha a regiao a ser monitorada e aperte C")
    cv.setMouseCallback("Escolha a regiao a ser monitorada e aperte C", click_and_crop)

    # O loop é executado até que a tecla 'q' seja pressionada
    while True:
        # Exibe a tela de referência para a escolha da área a ser selecionada
        cv.imshow("Escolha a regiao a ser monitorada e aperte C", img1)
        key = cv.waitKey(1) & 0xFF

        # A região de recorte pode ser selecionada novamente se a tecla 'r' for pressionada
        if key == ord("r"):
            img1 = clone.copy()

        # Se a tecla 'c' for pressionada, o programa é inicia o monitoramento da área selecionada
        elif key == ord("c"):

            break

    # Fecha a janela de seleção
    cv.destroyAllWindows()


# Método para iniciar o monitoramento da área
def monitorar_area(camera):
    # O loop é verdadeiro
    emLoop = True
    tempoAEsperar = 20 #segundos

    # Criação das variáveis que recebem os números de pixel da imagem 1 e 2 respectivamente
    x = 0
    y = 0
    valorArea = 0
    valorPerimetro = 0
    controle = 0

    #Criando as variáveis responsáveis pelo cálculo da margem de erro
    listaPerimetro = []
    listaArea = []


    start = time()
    #  Roda o loo´até chegar no tempo a espera
    while (time()-start < tempoAEsperar):
        # Inicia a camera
        ret2, img2 = camera.read()
        # Cria uma variável temporária para guardar a imagem da câmera
        temp2 = img2.copy()
        # Cria uma variável para guardar a imagem recortada para o monitoramento
        roi = temp2[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        # Coordenadas do Início e Fim da área selecionada
        start_point = (refPt[0][0], refPt[0][1])
        end_point = (refPt[1][0], refPt[1][1])
        # Define a cor do retângulo sendo vermelho
        color = (0, 0, 255)
        image = cv.rectangle(img2, start_point, end_point, color, 2)
        # Roi temporário
        temp = roi.copy()
        # Conversão do roi temporário para escala de cinza
        gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        if (x == 0 and y == 0):
            x = filtros_imagem1(gray)  # Valor do perímetro
            y = filtros_imagem2(gray)  # Valor da área
        if (ret2 != True):
            print("Algum quadro nao foi recebido.")
            break
        cv.imshow('TREINANDO', img2)
        # Soma os valores do perimetro e da area para que que seja tracada a media apos o fim do loop
        valorPerimetro += filtros_imagem1(gray)
        valorArea += filtros_imagem2(gray)
        controle += 1
        cv.imwrite('imagemRoi.jpg', roi)
        # Adiciona a uma lista os valores da imagem a cada monitoramento para que seja descoberto o desvio padrao
        listaPerimetro.insert(controle,filtros_imagem1(gray))
        listaArea.insert(controle, filtros_imagem2(gray))

        #print(filtros_imagem1(gray))  # Debug para printar o valor da área
        #print(filtros_imagem2(gray))  # Debug para printar valor do perímetro

        print("\n")


        # Monitorar a área a cada 500 milissegundos
        cv.waitKey(500)

    # Pega os valores da área e do perímetro e traca uma margem de erro e desvio padrao
    # que serao usados para definir a entrada do invasor na imagem
    desvioPadraoPerimetro = statistics.stdev(listaPerimetro)
    desvioPadraoArea = statistics.stdev(listaArea)

    xMedPerimetro = valorPerimetro / controle
    xMedArea = valorArea/controle

    print("\n")

    while (emLoop):
        # Inicia a camera
        ret2, img2 = camera.read()
        # Cria uma variável temporária para guardar a imagem da câmera
        temp2 = img2.copy()
        # Cria uma variável para guardar a imagem recortada para o monitoramento
        roi = temp2[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        # Coordenadas do Início e Fim da área selecionada
        start_point = (refPt[0][0], refPt[0][1])
        end_point = (refPt[1][0], refPt[1][1])
        # Define a cor do retângulo sendo vermelho
        color = (0, 0, 255)
        image = cv.rectangle(img2, start_point, end_point, color, 2)
        # Roi temporário
        temp = roi.copy()
        # Conversão do roi temporário para escala de cinza
        gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        if (x == 0 and y == 0):
            x = filtros_imagem1(gray)  # Valor do perímetro
            y = filtros_imagem2(gray)  # Valor da área
        if (ret2 != True):
            print("Algum quadro nao foi recebido.")
            break
        cv.imshow('REC', img2)

        #cv.imshow('SubtracaoDoBlur', filtros_genericos(roi))
        #cv.imshow('Laplaciano+Limiar', cv.Laplacian(filtros_genericos(roi), cv.CV_64F))
        #cv.imshow('FiltroBlur', cv.blur(roi, (5, 5), cv.CV_64F))

        #print(filtros_imagem1(gray))  # Debug para printar o valor da área
        #print(filtros_imagem2(gray))  # Debug para printar valor do perímetro

        print("\n")

        # Condicional para a ativação do alarme
        if ((filtros_imagem1(gray) > xMedPerimetro + (desvioPadraoPerimetro*2) and filtros_imagem2(gray) > xMedArea + (desvioPadraoArea*2))
                or (filtros_imagem1(gray) < xMedPerimetro - (desvioPadraoPerimetro*2) and filtros_imagem2(gray) < xMedArea - (desvioPadraoArea*2))):
            # Capturar a imagem caso a condicional seja satisfeita

            cv.imwrite("invasor.jpg", roi)
            # Sai do loop
            emLoop = False

        # Chamada do método para imprimir os textos na imagem no momento da invasão
        resgatar_data_hora(roi)

        # Monitorar a área a cada 250 milissegundos
        cv.waitKey(250)

    # Fecha todas as janelas
    cv.destroyAllWindows()
    # Exibe a imagem do invasor
    cv.imshow('Captura', roi)
    cv.waitKey(5000)
    imagem1 = "invasor.jpg"
    imagem2 = "imagemRoi.jpg"
    fft(imagem1)
    fft(imagem2)
    fft2(imagem1)
    fft2(imagem2)


# Método para registrar que houve invasão em um determinado momento
def resgatar_data_hora(roi):
    data_e_hora_atuais = datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y - %H:%M')
    text = "Data e hora da invasao:"
    # Adicionando os textos em suas devidas posições. X Sempre começando em 10 e Y pulando linha de 30 em 30
    # O tamanho da fonte é 0.75 e o texto é exibido na cor verde (para melhor leitura)
    cv.putText(roi, 'ALERTA INVASAO', (10, 30), cv.FONT_HERSHEY_SIMPLEX, (3/4), (0, 255, 0))
    cv.putText(roi, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, (3/4), (0, 255, 0))
    cv.putText(roi, data_e_hora_em_texto, (10, 90), cv.FONT_HERSHEY_SIMPLEX, (3/4), (0, 255, 0))

