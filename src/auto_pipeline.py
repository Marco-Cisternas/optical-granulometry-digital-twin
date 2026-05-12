import os
import subprocess
import glob
import time

# --- CONFIGURACIÓN ---
# 1. Ruta a tu ejecutable de Blender (AJUSTAR A TU PC)
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" 

# 2. Rutas de carpetas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "ensemble_dataset")     # Donde están los JSONs
TEMP_OBJ_DIR = os.path.join(BASE_DIR, "temp_objs")         # Donde guardaremos OBJs temporales
OUTPUT_RENDER_DIR = os.path.join(BASE_DIR, "ensemble_renders") # Donde saldrán las imágenes

# 3. Configuración de Render
PRESET = "8K"  # Calidad final
VIEW_MM = 50.0
ROT_X = 90.0   # Rotación si tus OBJs salen "parados" en vez de acostados

def run_command(cmd):
    """Ejecuta un comando de sistema y espera a que termine"""
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando comando: {e}")
        exit(1)

def main():
    # Crear carpetas si no existen
    os.makedirs(TEMP_OBJ_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RENDER_DIR, exist_ok=True)

    # Buscar todos los archivos run_X.json
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "run_*.json")))
    
    if not json_files:
        print(f"❌ No se encontraron archivos JSON en {INPUT_DIR}")
        return

    print(f"🚀 Iniciando Pipeline Ensemble para {len(json_files)} simulaciones...\n")

    for i, json_path in enumerate(json_files):
        filename = os.path.basename(json_path)
        name_no_ext = os.path.splitext(filename)[0] # ej: "run_0"
        
        print(f"--------------------------------------------------")
        print(f"🔄 Procesando {filename} ({i+1}/{len(json_files)})")
        print(f"--------------------------------------------------")

        # Rutas dinámicas
        obj_path = os.path.join(TEMP_OBJ_DIR, f"{name_no_ext}.obj")
        img_name = f"{name_no_ext}.png"

        # --- FASE 1: EXPORTAR A OBJ (Usando Python del sistema) ---
        print(f"   🔨 Generando OBJ...")
        report_path = os.path.join(TEMP_OBJ_DIR, f"{name_no_ext}_export_report.json")
        inbounds_json = os.path.join(TEMP_OBJ_DIR, f"{name_no_ext}_inbounds.json")

        cmd_export = [
          "python", "export_to_blender.py",
          "--json", json_path,
          "--out-obj", obj_path,
          "--out-report", report_path,
          "--out-json", inbounds_json,
          "--mesh-root", "./sand_atlas_meshes",
          "--clip-mm", str(VIEW_MM),
          "--strict-mesh"
        ]
        # Convertir lista a string para subprocess
        run_command(" ".join(cmd_export))

        # --- FASE 2: RENDERIZAR EN BLENDER ---
        print(f"   📸 Renderizando en Blender ({PRESET})...")
        cmd_render = [
            f'"{BLENDER_EXE}"',
            "--background",
            "--python", "render_in_blender.py",
            "--", # Separador de argumentos de Blender
            "--input_obj", f'"{obj_path}"',
            "--out_dir", f'"{OUTPUT_RENDER_DIR}"',
            "--out_name", img_name,
            "--preset", PRESET,
            "--view_width_mm", str(VIEW_MM),
            "--rot_x_deg", str(ROT_X)
        ]
        run_command(" ".join(cmd_render))
        
        print(f"   ✅ Listo: {img_name}")

    print("\n🎉 ¡PIPELINE FINALIZADO EXITOSAMENTE!")
    print(f"📂 Imágenes guardadas en: {OUTPUT_RENDER_DIR}")

if __name__ == "__main__":
    main()
