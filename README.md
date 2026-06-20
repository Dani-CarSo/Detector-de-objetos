Daniela Carbajal Sotomayor
# PROYECTO DE VISIÓN ARTIFICIAL
# Propuesta de aplicación real: agricultura y ganadería

Este mismo modelo, entrenado con un dataset propio en lugar de COCO128,
puede convertirse en una herramienta de apoyo para tareas de campo que
hoy se hacen mayormente a simple vista. A continuación se describe una
propuesta concreta de cómo llevarlo a producción.

# Problema que resuelve
En explotaciones agrícolas y ganaderas, gran parte del monitoreo
(contar cabezas de ganado, detectar animales enfermos o heridos,
identificar frutos listos para cosecha, detectar plagas o malezas en un
cultivo) se realiza de forma manual, recorriendo el terreno o revisando
video a simple vista. Esto consume tiempo, depende de la disponibilidad
de personal capacitado, y es propenso a errores por cansancio o
condiciones de poca visibilidad.

# Cómo se aplicaría el modelo
Captura de imágenes: Cámaras fijas en corrales o invernaderos,
cámaras montadas en drones que recorren el terreno, o cámaras
trampa en zonas de pastoreo, capturan imágenes o video de forma
continua.
Detección automática: El modelo YOLO, entrenado con fotos
reales del tipo de animal o cultivo específico, procesa cada imagen
y genera una caja por cada objeto relevante detectado (un animal,
una fruta madura, una zona con plaga visible), junto con su
ubicación y nivel de confianza.
Conteo y alertas: A partir de las detecciones, un sistema
simple puede generar un conteo automático (cuántos animales hay en
el corral, cuántos frutos están listos), o disparar una alerta si
detecta algo anómalo (un animal separado del grupo, una zona con
alta concentración de plaga).
Panel de seguimiento: Los resultados se acumulan en el tiempo,
permitiendo ver tendencias (crecimiento del rebaño, avance de una
plaga, ritmo de maduración de un cultivo) sin que nadie tenga que
recorrer el campo todos los días para registrarlo a mano.


# Qué se necesitaría para llevarlo a este punto
Dataset propio: Fotos reales del entorno específico (el tipo de
ganado, el cultivo, las condiciones de luz del lugar), etiquetadas
con cajas igual que se hizo aquí con COCO128. Esto es lo que más
impacta la precisión final, más que el número de épocas de
entrenamiento.
Más cómputo y más épocas: A diferencia de la prueba rápida de
este repositorio (3 épocas, modelo nano), una versión de producción
necesitaría entrenar muchas más épocas, posiblemente con un modelo
YOLO de mayor tamaño (yolov8s o yolov8m), para alcanzar una
precisión confiable en campo.
Hardware en el lugar de uso: Dependiendo de si el procesamiento
se hace en una cámara con poco cómputo (modelo exportado a un formato
liviano) o se envían las imágenes a un servidor, cambia qué tan
rápido se entera el productor de lo que el modelo detectó.


# Beneficio esperado
Reducir el tiempo que el personal dedica a tareas repetitivas de
conteo e inspección visual, detectar problemas (un animal enfermo, una
plaga temprana) antes de que se vuelvan más costosos de resolver, y
tener un registro objetivo y constante del estado del campo, en lugar
de depender únicamente de recorridos manuales esporádicos.

# Limitaciones a considerar
Un modelo de detección no reemplaza el criterio de un experto: puede
fallar en condiciones de poca luz, niebla, oclusión (animales o frutos
parcialmente tapados), o ante variaciones que no estaban representadas
en su dataset de entrenamiento. La propuesta más realista es usarlo
como apoyo que reduce la carga de trabajo manual, con revisión humana
periódica de los casos marcados como dudosos.

