import cv2
import time
import os
from ultralytics import YOLO

def main():
    print("🧠 Loading YOLOv8 Neural Network...")
    model = YOLO("yolov8n.pt")
    
    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Habilitar la cámara web para inspección en tiempo real
    if not camara.isOpened():
        print("❌ Error: Cannot open webcam stream.")
        return

    print("🎥 AI Industrial Inspection System Active. Press 'q' to exit.")

    # Variables para el reporte final de control de calidad
    total_inspeccionado = 0
    total_defectuosos = 0
    historial_defectos = [] # Lista para almacenar los registros de defectos detectados
    try:
        while True:
            ret, frame = camara.read()
            if not ret:
                break

            x1_zona, y1_zona = 150, 100
            x2_zona, y2_zona = 490, 380 # Definir las coordenadas de la zona de inspección
            zona_inspeccion = frame[y1_zona:y2_zona, x1_zona:x2_zona]

            # Inferencia con tu calibración optimizada a 40%
            resultados = model(zona_inspeccion, conf=0.40, verbose=False)

            estado_inspeccion = "AWAITING COMPONENT"
            color_borde = (0, 255, 255) 

            for resultado in resultados:
                for box in resultado.boxes:
                    id_clase = int(box.cls)
                    nombre_objeto = model.names[id_clase]# Obtener el nombre del objeto detectado
                    precision = float(box.conf)
                    
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (bx1 + x1_zona, by1 + y1_zona), 
                                (bx2 + x1_zona, by2 + y1_zona), (255, 0, 0), 1) # Dibujar el cuadro delimitador del objeto detectado

                    if nombre_objeto in ["cell phone", "cup", "bottle"]: # Lista de objetos defectuosos
                        estado_inspeccion = f"🚨 DEFECTIVE COMPONENT [NG] ({precision:.2%})"
                        color_borde = (0, 0, 255) 
                        
                        # Registrar métricas del defecto
                        total_defectuosos += 1
                        marca_tiempo = time.strftime('%H:%M:%S')
                        historial_defectos.append(f"[{marca_tiempo}] Defect: {nombre_objeto} ({precision:.2%})")
                        break
                        
                    elif nombre_objeto in ["person", "scissors", "book"]: # Lista de objetos conformes
                        estado_inspeccion = f"✅ COMPONENT VALID [OK] ({precision:.2%})"
                        color_borde = (0, 255, 0) 
                        total_inspeccionado += 1

            cv2.rectangle(frame, (x1_zona, y1_zona), (x2_zona, y2_zona), color_borde, 3)
            cv2.putText(frame, estado_inspeccion, (150, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_borde, 2)

            nombre_ventana = "AI Vision Inspection System - RoboticsLab"
            cv2.imshow(nombre_ventana, frame)

            tecla = cv2.waitKey(1) & 0xFF
            ventana_cerrada = cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1
            if tecla == ord('q') or ventana_cerrada:
                break
    except KeyboardInterrupt:
        print("\n🛑 Ejecución interrumpida manualmente.")

    finally:
        # ESTO SIEMPRE SE EJECUTA, INCLUSO SI HAY UN ERROR O PRESIONAS CTRL+C
        camara.release()
        cv2.destroyAllWindows()
        print("📷 Cámara liberada correctamente.")

    # GENERACIÓN AUTOMÁTICA DEL REPORTE INDUSTRIAL AL SALIR
    carpeta_destino = "audit_logs" # Nombre de la carpeta donde se guardarán los reportes
    os.makedirs(carpeta_destino, exist_ok=True) # Crea la carpeta automáticamente si no existe
    
    fecha_hora = time.strftime('%Y-%m-%d_%H-%M-%S') # Marca de tiempo para no sobrescribir archivos
    ruta_archivo = os.path.join(carpeta_destino, f"quality_audit_{fecha_hora}.txt")

    print(f"\n💾 Generando reporte de auditoría de calidad en '{ruta_archivo}'...")
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("🏭 ROBOTICSLAB - AI VISION QUALITY AUDIT REPORT\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')} | Time: {time.strftime('%H:%M:%S')}\n")
        f.write("==================================================\n\n")
        f.write(f"📊 Total Conforming Parts [OK]: {total_inspeccionado}\n")
        f.write(f"❌ Total Non-Conforming Parts [NG]: {total_defectuosos}\n")
        if total_inspeccionado + total_defectuosos > 0:
            yield_rate = (total_inspeccionado / (total_inspeccionado + total_defectuosos)) * 100
            f.write(f"📈 Line Yield Rate: {yield_rate:.2f}%\n\n")
        f.write("📋 Defect Log Details:\n")
        f.write("--------------------------------------------------\n")
        if not historial_defectos:
            f.write("No defects detected during this run.\n")
        else:
            for log in historial_defectos:
                f.write(f"{log}\n")
        f.write("\n==================================================\n")
    
    print(f"✅ Reporte guardado con éxito en '{ruta_archivo}'.")

if __name__ == "__main__":
    main()
