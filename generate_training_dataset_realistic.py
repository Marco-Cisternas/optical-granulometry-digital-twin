import bpy
import os
import math
import random
import glob
from mathutils import Vector

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\mcist\OneDrive\Escritorio\tesis"
MESH_DIR_ROOT = os.path.join(BASE_DIR, "sand_atlas_meshes")

# Carpetas de Salida
OUTPUT_IMAGES = os.path.join(BASE_DIR, "dataset", "training", "images")
OUTPUT_MASKS = os.path.join(BASE_DIR, "dataset", "training", "masks")
OUTPUT_INSTANCES = os.path.join(BASE_DIR, "dataset", "training", "instances") # <--- NUEVO

RES_X = 512
RES_Y = 512
SAMPLES = 64
# Ajusta esto según necesites (pon 0 para procesar todos)
LIMIT_SAMPLES = 2000  

# Colores tierra para la imagen realista
EARTH_COLORS = [
    (0.2, 0.15, 0.1, 1), (0.4, 0.3, 0.2, 1), (0.6, 0.5, 0.4, 1),
    (0.5, 0.5, 0.55, 1), (0.7, 0.65, 0.6, 1), (0.1, 0.1, 0.12, 1)
]

def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in bpy.data.meshes:
        if b.users == 0: bpy.data.meshes.remove(b)
    for b in bpy.data.materials:
        if b.users == 0: bpy.data.materials.remove(b)

def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.cycles.samples = SAMPLES
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    
    # Intentar usar GPU
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices: 
            if 'RTX' in device.name or 'GTX' in device.name: device.use = True
    except: pass

def center_and_frame_object(obj):
    """Centra y ajusta la cámara al objeto."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0,0,0)
    
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim <= 0: return False

    cam_dist = max_dim * 5.0
    if bpy.context.scene.camera:
        bpy.data.objects.remove(bpy.context.scene.camera, do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, cam_dist), rotation=(0, 0, 0))
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max_dim * 1.3
    cam.data.clip_start = max_dim * 0.01
    cam.data.clip_end = cam_dist * 5.0
    bpy.context.scene.camera = cam
    return True

def create_realistic_material(base_color):
    mat = bpy.data.materials.new(name="Mat_Realistic")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = 1.0
    
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 20.0 
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.5
    
    mat.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    mat.node_tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def create_mask_material():
    """Material Blanco Puro (Binario)."""
    mat = bpy.data.materials.new(name="Mat_Mask")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emis = nodes.new('ShaderNodeEmission')
    emis.inputs['Color'].default_value = (1, 1, 1, 1)
    mat.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    return mat

def create_instance_material():
    """Material Grises Aleatorios (Instancia)."""
    mat = bpy.data.materials.new(name="Mat_Instance")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emis = nodes.new('ShaderNodeEmission')
    
    # Geometría Random -> Color
    geo = nodes.new('ShaderNodeNewGeometry')
    
    mat.node_tree.links.new(geo.outputs['Random Per Island'], emis.inputs['Color'])
    mat.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    return mat

def render_pass(output_path, transparent_bg):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    scene.render.film_transparent = transparent_bg 
    bpy.ops.render.render(write_still=True)

def main():
    # Crear carpetas si no existen
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_MASKS, exist_ok=True)
    os.makedirs(OUTPUT_INSTANCES, exist_ok=True)
    
    setup_render_settings()

    print(f"📂 Buscando modelos en {MESH_DIR_ROOT}...")
    generated = glob.glob(os.path.join(MESH_DIR_ROOT, "50000_generated_grains", "*.obj"))
    # originals = glob.glob(os.path.join(MESH_DIR_ROOT, "1551_original_grains", "*.obj")) # Descomentar si tienes la carpeta
    
    full_list = generated # + originals
    random.shuffle(full_list)
    
    # Limitar cantidad
    if LIMIT_SAMPLES > 0:
        full_list = full_list[:LIMIT_SAMPLES]

    print(f"✅ Procesando {len(full_list)} modelos...")

    for i, mesh_path in enumerate(full_list):
        # Generar ID único
        parent_dir = os.path.basename(os.path.dirname(mesh_path))
        filename = os.path.splitext(os.path.basename(mesh_path))[0]
        unique_id = f"{parent_dir}_{filename}"
        
        print(f"[{i+1}/{len(full_list)}] {unique_id}")
        
        reset_scene()
        
        try:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        except:
            try: bpy.ops.import_scene.obj(filepath=mesh_path)
            except: continue
            
        if not bpy.context.selected_objects: continue
        obj = bpy.context.selected_objects[0]
        
        # Rotación aleatoria
        obj.rotation_euler = (random.uniform(0,3.14), random.uniform(0,3.14), random.uniform(0,3.14))
        
        if not center_and_frame_object(obj): continue

        # Luz dinámica simple
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.object
        sun.data.energy = 5.0
        sun.rotation_euler = (math.radians(45), math.radians(30), 0)

        # ----------------------------------------------------
        # 1. RENDER IMAGEN REALISTA (RGB)
        # ----------------------------------------------------
        col = random.choice(EARTH_COLORS)
        mat_real = create_realistic_material(col)
        if obj.data.materials: obj.data.materials[0] = mat_real
        else: obj.data.materials.append(mat_real)
        
        render_pass(os.path.join(OUTPUT_IMAGES, f"{unique_id}.png"), True)
        
        # Apagar luces y fondo negro para máscaras
        for o in bpy.data.objects:
            if o.type == 'LIGHT': o.hide_render = True
        
        if not bpy.context.scene.world: 
            bpy.context.scene.world = bpy.data.worlds.new("World_Mask")
        if bpy.context.scene.world.node_tree:
             bg = bpy.context.scene.world.node_tree.nodes.get("Background")
             if bg: bg.inputs[0].default_value = (0,0,0,1)

        # ----------------------------------------------------
        # 2. RENDER MÁSCARA BINARIA (BLANCO)
        # ----------------------------------------------------
        mat_mask = create_mask_material()
        obj.data.materials[0] = mat_mask
        render_pass(os.path.join(OUTPUT_MASKS, f"{unique_id}.png"), False)

        # ----------------------------------------------------
        # 3. RENDER MÁSCARA DE INSTANCIA (GRISES)
        # ----------------------------------------------------
        mat_inst = create_instance_material()
        obj.data.materials[0] = mat_inst
        # Importante: fondo negro, objeto gris aleatorio
        render_pass(os.path.join(OUTPUT_INSTANCES, f"{unique_id}.png"), False)

    print("✅ ¡Generación completada! Verifica la carpeta /instances/")

if __name__ == "__main__":
    main()