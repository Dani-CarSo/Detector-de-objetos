
import shutil # Para copiar archivos de pesos
from pathlib import Path   # Para manejar rutas de archivos de manera más robusta

from ultralytics import YOLO # Para cargar y entrenar modelos YOLOv8


def get_model(base_model="yolov8n.pt"): # Cargar el modelo base
   
    print(f" Cargando modelo base: {base_model}") # Verificar que el modelo base exista
    return YOLO(base_model) # Cargar el modelo base (puede ser yolov8n.pt, yolov8s.pt, etc.)


def get_dataset_config(yaml_path="configs/coco128.yaml"): # Verificar que el archivo de configuración del dataset exista
  
    path = Path(yaml_path) # Convertir a Path para manejo de archivos
    if not path.exists(): # Verificar que el archivo de configuración del dataset exista
        raise FileNotFoundError( # Si no se encuentra el archivo, lanzar un error con un mensaje claro
            f"No se encontró el archivo de configuración del dataset: {path}" # Mensaje de error claro indicando la ruta del archivo que no se encontró
        )
    print(f" Usando configuración de dataset: {path}") # Verificar que se está utilizando la configuración de dataset correcta
    return str(path) # Devolver la ruta del archivo de configuración del dataset como cadena (str) para que pueda ser utilizada por el modelo


def train(model, data_yaml, epochs=50, imgsz=640, batch=16, device=0, # Entrenar el modelo con los parámetros especificados
          project="runs/detect", name="train", patience=20): # Parámetros de entrenamiento: número de épocas, tamaño de imagen, tamaño de lote, dispositivo (CPU o GPU), directorio del proyecto, nombre del experimento y paciencia para detener el entrenamiento si no hay mejora
   
    print(
        f" Entrenando modelo\n" # Imprimir los parámetros de entrenamiento para verificar que se están utilizando los valores correctos
        f"   epochs : {epochs}\n" # Número de épocas para entrenar
        f"   imgsz  : {imgsz}\n" # Tamaño de imagen para el entrenamiento
        f"   batch  : {batch}\n" # Tamaño de lote para el entrenamiento
        f"   device : {device}\n" # Dispositivo para el entrenamiento (0 para GPU, -1 para CPU)
    )
    results = model.train( # Iniciar el entrenamiento del modelo con los parámetros especificados
        data=data_yaml, # Ruta al archivo de configuración del dataset
        epochs=epochs, # Número de épocas para entrenar
        imgsz=imgsz, # Tamaño de imagen para el entrenamiento
        batch=batch, # Tamaño de lote para el entrenamiento
        device=device, # Dispositivo para el entrenamiento (0 para GPU, -1 para CPU)
        project=project, # Directorio del proyecto donde se guardarán los resultados del entrenamiento
        name=name, # Nombre del experimento para organizar los resultados del entrenamiento
        patience=patience, # Paciencia para detener el entrenamiento si no hay mejora en las métricas durante este número de épocas
    )
    print("Entrenamiento finalizado.")  # Imprimir mensaje indicando que el entrenamiento ha finalizado
    print(f"   Resultados guardados en: {model.trainer.save_dir}") # Verificar que los resultados del entrenamiento se han guardado en el directorio correcto
    return results # Devolver los resultados del entrenamiento para su posterior análisis o evaluación


def get_best_weights_path(model): # Obtener la ruta de los mejores pesos después del entrenamiento
   
    save_dir = Path(model.trainer.save_dir) # Convertir a Path para manejar archivos de manera más robusta
    pesos = save_dir / "weights" / "best.pt" # Construir la ruta completa a los mejores pesos (best.pt) dentro del directorio de resultados del entrenamiento
    if not pesos.exists(): # Verificar que los mejores pesos existen en la ruta especificada
        raise FileNotFoundError(f"No se encontraron pesos en: {pesos}") # Si no se encuentran los pesos, lanzar un error con un mensaje claro indicando la ruta donde se esperaba encontrar los pesos
    return str(pesos) # Devolver la ruta de los mejores pesos como cadena (str) para que pueda ser utilizada por otras funciones, como la de evaluación o guardado del modelo


def evaluate(model, data_yaml=None): # Evaluar el modelo utilizando el conjunto de validación especificado en el archivo de configuración del dataset (data_yaml) o utilizando el conjunto de validación predeterminado si no se proporciona un archivo de configuración
  
    print(" Evaluando modelo...") # Imprimir mensaje indicando que se está evaluando el modelo
    if data_yaml: # Si se proporciona un archivo de configuración del dataset, utilizarlo para la evaluación
        metrics = model.val(data=data_yaml) # Evaluar el modelo utilizando el conjunto de validación especificado en el archivo de configuración del dataset (data_yaml)
    else: # Si no se proporciona un archivo de configuración del dataset, utilizar el conjunto de validación predeterminado para la evaluación
        metrics = model.val() # Evaluar el modelo utilizando el conjunto de validación predeterminado (puede ser el conjunto de validación del dataset utilizado para el entrenamiento o un conjunto de validación específico para la evaluación)

    print(f"   mAP50-95        : {metrics.box.map:.4f}") # Imprimir las métricas de evaluación para verificar el rendimiento del modelo
    print(f"   mAP50           : {metrics.box.map50:.4f}") # Imprimir la métrica de precisión media a un umbral de IoU del 50% para verificar el rendimiento del modelo en detección de objetos con un umbral de IoU específico
    print(f"   Precisión media : {metrics.box.mp:.4f}") # Imprimir la métrica de precisión media para verificar el rendimiento del modelo en términos de precisión general en la detección de objetos
    print(f"   Recall medio    : {metrics.box.mr:.4f}") # Imprimir la métrica de recall medio para verificar el rendimiento del modelo en términos de recall general en la detección de objetos
    return metrics # Devolver las métricas de evaluación para su posterior análisis o comparación con otros modelos o configuraciones de entrenamiento


def save_model(weights_path, destino="mejor_modelo_yolo.pt"): # Guardar el modelo utilizando la ruta de los pesos obtenida después del entrenamiento y especificando un destino para el archivo del modelo guardado
   
    origen = Path(weights_path) # Convertir a Path para manejar archivos de manera más robusta
    if not origen.exists(): # Verificar que el archivo de pesos existe antes de intentar guardarlo
        raise FileNotFoundError(f"No se encontraron los pesos en: {origen}") # Si no se encuentran los pesos, lanzar un error con un mensaje claro indicando la ruta donde se esperaba encontrar los pesos

    shutil.copy(origen, destino) # Copiar los pesos desde la ruta de origen (weights_path) a la ruta de destino (destino) para guardar el modelo con un nombre específico o en una ubicación deseada
    print(f" Modelo guardado en: {destino}") 
    return destino # Devolver la ruta del modelo guardado para su posterior uso o referencia


def load_model(weights_path): # Cargar el modelo utilizando la ruta de los pesos obtenida después del entrenamiento
   
    path = Path(weights_path) # Convertir a Path para manejar archivos de manera más robusta
    if not path.exists():   # Verificar que el archivo de pesos existe antes de intentar cargarlo
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {path}") # Verificar que el archivo de pesos existe antes de intentar cargarlo

    print(f" Cargando modelo desde: {path}") # Verificar que se está cargando el modelo desde la ruta correcta
    return YOLO(str(path)) # Cargar el modelo utilizando la ruta de los pesos como cadena (str) para que pueda ser utilizada por la función de carga del modelo YOLOv8
