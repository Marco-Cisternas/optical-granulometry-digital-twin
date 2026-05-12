import os

# --- CONFIGURACIÓN ---
# Carpeta donde están tus 50,000 archivos originales (ajusta si es necesario)
input_folder = "sand_atlas_meshes/50000_generated_grains"

# Carpeta donde se guardarán los archivos escalados
output_folder = "sand_atlas_meshes/50000_generated_grains_scaled"

# Factor de escala calculado anteriormente
scale_factor = 207.9277
# ---------------------

def scale_obj_file(input_path, output_path, factor):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('v '):
                # Es una línea de vértice: "v x y z"
                parts = line.strip().split()
                try:
                    # Parsear coordenadas
                    x = float(parts[1]) * factor
                    y = float(parts[2]) * factor
                    z = float(parts[3]) * factor
                    # Escribir con 8 decimales para mantener precisión
                    f_out.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
                except (ValueError, IndexError):
                    # Si la línea está mal formada, la copiamos tal cual por seguridad
                    f_out.write(line)
            else:
                # Copiar cualquier otra línea (caras, normales, comentarios) sin cambios
                f_out.write(line)

def main():
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Carpeta creada: {output_folder}")

    # Listar archivos .obj
    files = [f for f in os.listdir(input_folder) if f.endswith('.obj')]
    total_files = len(files)
    
    print(f"Iniciando el escalado de {total_files} archivos...")
    print(f"Factor de escala: {scale_factor}")

    for i, filename in enumerate(files):
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)
        
        scale_obj_file(in_path, out_path, scale_factor)

        # Mostrar progreso cada 1000 archivos
        if (i + 1) % 1000 == 0:
            print(f"Procesados: {i + 1}/{total_files}")

    print("¡Proceso terminado exitosamente!")

if __name__ == "__main__":
    main()
