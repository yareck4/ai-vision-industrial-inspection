import cv2
from ultralytics import YOLO

def main():
    # 1. Cargar el modelo YOLOv8 Nano (Es ligero, rápido y se descarga solo en el primer uso)
    print("🧠 Cargando Red Neuronal YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    # Inicializar la cámara
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("❌ Error: No se puede abrir la webcam.")
        return

    print("🎥 Sistema de Inspección Industrial por IA Activo. Presiona 'q' para salir.")

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        # Coordenadas de la "Zona de Inspección" (Banda Virtual)
        x1_zona, y1_zona = 150, 100
        x2_zona, y2_zona = 490, 380

        # Recortar la imagen para que la IA SOLO inspeccione lo que pasa dentro del cuadro amarillo
        zona_inspeccion = frame[y1_zona:y2_zona, x1_zona:x2_zona]

        # Pasar el recorte por la Red Neuronal (YOLO)
        # verbose=False para no saturar la terminal con logs de la IA
        resultados = model(zona_inspeccion,conf=0.4, verbose=False)

        # Estados iniciales por defecto de la línea de producción
        estado_inspeccion = "ESPERANDO PIEZA"
        color_borde = (0, 255, 255) 

        # Procesar las detecciones de la IA
        for resultado in resultados:
            for box in resultado.boxes:
                # Obtener el nombre del objeto detectado (ej. 'person', 'cell phone', 'cup')
                id_clase = int(box.cls[0])
                nombre_objeto = model.names[id_clase]

                precision = float(box.conf[0]) # Confiabilidad de la detección
                
                # Obtener coordenadas de la detección relativa a la zona
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                
                # Mapear las coordenadas de vuelta a la pantalla global para dibujar el recuadro del objeto
                cv2.rectangle(frame, (bx1 + x1_zona, by1 + y1_zona), 
                              (bx2 + x1_zona, by2 + y1_zona), (255, 0, 0), 1)

                # LÓGICA DE CONTROL DE CALIDAD INDUSTRIAL
                if nombre_objeto in ["cell phone", "cup", "bottle"]:
                    # Si detecta un objeto cerrado/invasivo -> PIEZA DEFECTUOSA (No Conforme / NG)
                    estado_inspeccion = "🚨 PIEZA DEFECTUOSA [NG]"
                    color_borde = (0, 0, 255) # Rojo 
                    break
                elif nombre_objeto in ["person", "scissors", "book"]:
                    # Si detecta la mano abierta o superficie limpia -> PIEZA CORRECTA (OK)
                    estado_inspeccion = "✅ PIEZA CORRECTA [OK]"
                    color_borde = (0, 255, 0) # Verde

        # Interfaz del Operador (HMI en pantalla)
        # Dibujar la caja de la banda virtual con el color del estado actual
        cv2.rectangle(frame, (x1_zona, y1_zona), (x2_zona, y2_zona), color_borde, 3)
        
        # Desplegar el veredicto del control de calidad
        cv2.putText(frame, estado_inspeccion, (150, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_borde, 2)

        # Mostrar ventana
        cv2.imshow("AI Vision Inspection System - RoboticsLab", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
