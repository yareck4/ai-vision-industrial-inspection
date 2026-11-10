# 👁️ Sistema de inspección industrial mediante visión artificial e IA

Este repositorio contiene un **sistema de control de calidad basado en visión artificial** de alta velocidad, diseñado para líneas de fabricación y ensamblaje automatizadas. Utilizando **OpenCV** para el procesamiento de imágenes de bajo nivel y **YOLOv8 (You Only Look Once)** para la inferencia mediante aprendizaje profundo (*deep learning*) en tiempo real, la aplicación define una ventana de escaneo activa para clasificar los componentes en estados **OK (conformes)** o **NG (no conformes/defectuosos)**, basándose en parámetros geométricos de los objetos.

---

## 🛠️ Características técnicas y optimización

*   **Filtrado por región de interés (ROI):** En lugar de ejecutar la inferencia de aprendizaje profundo sobre toda la imagen captada por la cámara, el sistema recorta una sección específica («zona de cinta virtual»). Esto limita el consumo de ciclos de CPU/GPU, reduce la latencia computacional y optimiza la tasa de fotogramas por segundo (FPS).
*   **Ajuste del umbral de confianza:** Se han integrado controles de umbral (`conf=0.65`) que obligan a la red neuronal a descartar predicciones con baja fiabilidad (inferiores al 65 % de certeza). Este filtrado estructural ayuda a mitigar el ruido ambiental, las variaciones en la iluminación y los falsos positivos, como aquellos provocados por la presencia de manos de los operarios.
*   **Diseño para computación en el borde (*Edge Computing*):** Gracias al uso de una arquitectura ligera (`yolov8n.pt`), el sistema es totalmente portátil y puede implementarse directamente en nodos de computación industrial, como dispositivos NVIDIA Jetson o placas Raspberry Pi. ---

## 📂 Estructura del proyecto

```text
├── .venv/                   # Entorno aislado que contiene el núcleo de Ultralytics
├── inspeccion_vision.py     # Script principal de visión artificial (IA)
└── README.md                # Documentación del sistema y registros
```

---

## 🚀 Guía de configuración y ejecución

### 1. Inicializar dependencias
Dentro de tu entorno virtual activo, instala el paquete de visión artificial:
```bash
.\.venv\Scripts\pip install opencv-python ultralytics
```

### 2. Ejecutar el proceso de inspección
```bash
python "inspeccion_vision.py"
```

### 3. Controles para el operador
*   Coloca la mano abierta o una superficie plana dentro del área de inspección: El sistema indicará **`COMPONENT VALID [OK]`** (indicador verde).
*   Introduce un objeto cerrado (como un teléfono móvil o una taza): El sistema indicará **`DEFECTIVE COMPONENT [NG]`** (alerta roja).
*   Presiona la tecla **`q`** en la ventana de visualización de video para finalizar correctamente la captura de datos.

---

## 🧠 Análisis técnico: Ajustes críticos

### Problema: Falsos positivos y detección de objetos no deseados
*   **Síntoma:** Cuando el área de inspección estaba vacía, o cuando la mano de un operador entraba en los bordes, el cuadro delimitador (*bounding box*) parpadeaba de forma errática, clasificando incorrectamente pliegues de la piel o sombras como herramientas o piezas.
*   **Diagnóstico:** La configuración base de los pesos de YOLO utiliza un umbral de confianza del 25 % (`conf=0.25`). Aunque es adecuado para sistemas de videovigilancia de áreas amplias, resulta demasiado inestable para las tolerancias de precisión requeridas en líneas de fabricación a corta distancia.
*   **Resolución:** Se ajustaron las tolerancias del sistema mediante un filtrado más estricto en la función de inferencia: `model(frame, conf=0.4)`. Esto aumentó el umbral de activación de los pesos, obligando a la red a registrar un cambio de estado solo cuando la matriz de probabilidad coincide con los objetivos definidos; esto eliminó por completo el ruido generado por elementos no relevantes, como los dedos del operador.