import glob                                    # para buscar archivos por patrón 
import random                                  # para elegir imágenes al azar
from pathlib import Path                       # para manejar rutas de archivos cómodamente
import cv2                                      # OpenCV, para leer/dibujar sobre imágenes
import matplotlib.pyplot as plt                 # para mostrar las imágenes en pantalla


def explore_dataset(images_dir, labels_dir=None, num_images=5, class_names=None):
    images_dir = Path(images_dir)                                  # convierte la ruta de texto a un objeto Path
    if labels_dir is None:                                          # si no se especificó carpeta de etiquetas
        labels_dir = Path(str(images_dir).replace("images", "labels"))  # la deduce cambiando 'images' por 'labels'
    else:
        labels_dir = Path(labels_dir)                               # si se especificó, la usa tal cual

    image_paths = sorted(glob.glob(str(images_dir / "*.jpg"))) + sorted(glob.glob(str(images_dir / "*.png")))  # busca todas las imágenes .jpg y .png en la carpeta
    if not image_paths:                                             # si la lista quedó vacía
        print(f"No se encontraron imágenes en: {images_dir}")       # avisa que no hay imágenes
        return                                                      # y termina la función aquí

    muestra = random.sample(image_paths, min(num_images, len(image_paths)))
    #  elige al azar num_images imágenes (o menos, si no hay suficientes)

    fig, axes = plt.subplots(1, len(muestra), figsize=(5 * len(muestra), 5))
    #  crea una fila de "lienzos" (uno por imagen), con ancho proporcional a cuántas haya
    if len(muestra) == 1:                                           # si solo hay una imagen
        axes = [axes]                                               # matplotlib no devuelve una lista, así que se fuerza a que sea una

    for ax, img_path in zip(axes, muestra):                         # recorre cada lienzo junto con su imagen correspondiente
        img = cv2.imread(img_path)                                  # carga la imagen desde disco
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                   # OpenCV usa orden de colores BGR, se convierte a RGB para verse bien
        h, w = img.shape[:2]                                        # obtiene alto y ancho de la imagen en píxeles

        label_path = labels_dir / (Path(img_path).stem + ".txt")    # construye la ruta del archivo .txt de etiquetas (mismo nombre que la imagen)
        if label_path.exists():                                     # si existe ese archivo de etiquetas
            with open(label_path, "r") as f:                        # lo abre para leer
                for linea in f.readlines():                         # recorre cada línea (cada línea = un objeto etiquetado)
                    partes = linea.strip().split()                  # separa la línea en sus 5 valores (clase, x, y, ancho, alto)
                    if len(partes) != 5:                            # si la línea no tiene exactamente 5 valores
                        continue                                    # la ignora (línea corrupta o vacía)
                    cls_id, xc, yc, bw, bh = partes                  # desempaqueta los 5 valores en variables con nombre
                    cls_id = int(cls_id)                             # el id de clase es un entero
                    xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                    #  las coordenadas en el .txt están normalizadas (0 a 1), aquí se convierten a píxeles reales
                    x1, y1 = int(xc - bw / 2), int(yc - bh / 2)      # calcula la esquina superior-izquierda de la caja
                    x2, y2 = int(xc + bw / 2), int(yc + bh / 2)      # calcula la esquina inferior-derecha de la caja
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    #  dibuja el rectángulo de la caja sobre la imagen, en rojo, con grosor 2
                    etiqueta = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
                    # si se pasó un diccionario de nombres de clase, busca el nombre; si no, usa el número
                    cv2.putText(img, etiqueta, (x1, max(y1 - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    #  escribe el texto de la etiqueta justo arriba de la caja

        ax.imshow(img)                                              # muestra la imagen (ya con cajas dibujadas) en su lienzo
        ax.set_title(Path(img_path).name, fontsize=9)               # le pone como título el nombre del archivo
        ax.axis("off")                                              # oculta los ejes de coordenadas (no son útiles aquí)

    plt.tight_layout()                                              # ajusta espacios para que no se encimen los lienzos
    plt.show()                                                      # muestra la figura completa en pantalla


def show_predictions(model, source, num_images=5, conf=0.25):
    source_path = Path(source)                                      # convierte la ruta de entrada a un objeto Path

    if source_path.is_dir():                                        # si la ruta es una carpeta (no una sola imagen)
        candidatos = sorted(glob.glob(str(source_path / "*.jpg")))  + sorted(glob.glob(str(source_path / "*.png")))   # busca todas las .jpg y .png dentro
        if not candidatos:                                          # si no encontró ninguna imagen
            print(f"No se encontraron imágenes en: {source_path}")  # avisa
            return                                                  # y termina
        imagenes = random.sample(candidatos, min(num_images, len(candidatos)))
        #  elige al azar num_images imágenes de la carpeta
    else:
        imagenes = [str(source_path)]                               # si es un solo archivo, lo mete en una lista de un elemento

    resultados = model.predict(source=imagenes, conf=conf, save=False, verbose=False)
    #  le pide al modelo YOLO que detecte objetos en esas imágenes
    #   conf = umbral mínimo de confianza para considerar una detección válida
    #   save=False porque no queremos que las guarde en disco, solo verlas aquí

    fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados), 5))
    #  crea un lienzo por cada imagen procesada
    if len(resultados) == 1:                                        # si solo hubo un resultado
        axes = [axes]                                               # lo convierte en lista para poder iterar igual

    for ax, r in zip(axes, resultados):                             # recorre cada lienzo junto con su resultado de predicción
        img_anotada = r.plot()                                      # YOLO genera la imagen ya con las cajas predichas dibujadas
        img_anotada = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)   # convierte de BGR a RGB para mostrarla correctamente
        ax.imshow(img_anotada)                                      # la muestra en su lienzo
        ax.set_title(Path(r.path).name, fontsize=9)                 # título con el nombre del archivo original
        ax.axis("off")                                              # sin ejes de coordenadas

    plt.tight_layout()                                              # ajusta espacios entre lienzos
    plt.show()                                                      # muestra todo en pantalla


def show_training_curves(results_dir="runs/detect/train"):
    ruta = Path(results_dir) / "results.png"                        # ruta a la imagen de curvas que genera Ultralytics
    if not ruta.exists():                                           # si ese archivo no existe todavía
        print(f"No se encontró results.png en: {results_dir}")      # avisa
        return                                                      # y termina

    img = cv2.imread(str(ruta))                                     # carga la imagen de las curvas
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # convierte de BGR a RGB
    plt.figure(figsize=(12, 8))                                     # crea un lienzo grande (las curvas tienen varios gráficos)
    plt.imshow(img)                                                 # muestra la imagen
    plt.axis("off")                                                 # sin ejes
    plt.title("Curvas de entrenamiento")                            # título de la ventana
    plt.show()                                                      # la despliega en pantalla


def show_confusion_matrix(results_dir="runs/detect/train"):
    ruta = Path(results_dir) / "confusion_matrix.png"                # ruta a la matriz de confusión generada por Ultralytics
    if not ruta.exists():                                           # si no existe ese archivo
        print(f"No se encontró confusion_matrix.png en: {results_dir}")  # avisa
        return                                                      # y termina

    img = cv2.imread(str(ruta))                                     # carga la imagen de la matriz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # convierte de BGR a RGB
    plt.figure(figsize=(8, 8))                                      # crea un lienzo cuadrado
    plt.imshow(img)                                                 # muestra la imagen
    plt.axis("off")                                                 # sin ejes
    plt.title("Matriz de confusión")                                # título
    plt.show()                                                      # la despliega en pantalla