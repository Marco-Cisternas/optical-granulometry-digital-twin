import bpy
import os
import random
import math
import glob
import numpy as np
from mathutils import Vector

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\mcist\OneDrive\Escritorio\tesis"
MESH_DIR = os.path.join(BASE_DIR, "sand_atlas_meshes")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "training_hq")

# --- PARÁMETROS DE GENERACIÓN ---
NUM_IMAGES = 200          # Cantidad de imágenes (alta calidad toma más tiempo)
VIEW_WIDTH = 0.3          # 30cm (Zoom para que se vea ultra nítido, simula recorte de 8K)

# --- CALIDAD VISUAL ---
RES_X = 2048              # 4 veces más resolución que antes
RES_Y = 2048
SAMPLES = 64              # Suficiente para training con denoiser

# Escala Física: 1 unidad Blender = 1 Metro
SCALE_UNIT = 0.001 

def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in bpy.data.meshes:
        if b.users == 0: bpy.data.meshes.remove(b)
    for b in bpy.data.materials:
        if b.users == 0: bpy.data.materials.remove(b)

def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.cycles.samples = SAMPLES
    
    # IGUALAR AL RENDER FINAL
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        for d in prefs.devices: d.use = True
    except: pass

# --- MATERIALES (Copia Exacta de render_in_blender.py) ---
def create_materials():
    # 1. Mat_Sand (Idéntico)
    m_sand = bpy.data.materials.new(name="Mat_Sand")
    m_sand.use_nodes = True
    nodes = m_sand.node_tree.nodes; nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial'); out.location = (400,0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0,0)
    
    bsdf.inputs['Roughness'].default_value = 1.0 
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    geo = nodes.new('ShaderNodeNewGeometry'); geo.location = (-800, 200)
    ramp = nodes.new('ShaderNodeValToRGB'); ramp.location = (-500, 200)
    ramp.color_ramp.interpolation = 'CONSTANT'
    
    els = ramp.color_ramp.elements
    els.remove(els[0])
    def add_col(pos, col): 
        e = els.new(pos); e.color = col
    
    add_col(0.0, (0.2, 0.15, 0.1, 1))
    add_col(0.16, (0.4, 0.3, 0.2, 1))
    add_col(0.33, (0.6, 0.5, 0.4, 1))
    add_col(0.50, (0.5, 0.5, 0.55, 1))
    add_col(0.66, (0.7, 0.65, 0.6, 1))
    add_col(0.83, (0.1, 0.1, 0.12, 1))
    
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 1500.0
    bump = nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.8
    
    m_sand.node_tree.links.new(geo.outputs['Random Per Island'], ramp.inputs['Fac'])
    m_sand.node_tree.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    m_sand.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    m_sand.node_tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    m_sand.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    # 2. Mat_Mask (Blanco Puro)
    m_mask = bpy.data.materials.new(name="Mat_Mask")
    m_mask.use_nodes = True
    nodes = m_mask.node_tree.nodes; nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emis = nodes.new('ShaderNodeEmission')
    emis.inputs['Color'].default_value = (1,1,1,1)
    emis.inputs['Strength'].default_value = 10.0
    m_mask.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    
    # 3. Mat_Instance (Grises Aleatorios)
    m_inst = bpy.data.materials.new(name="Mat_Inst")
    m_inst.use_nodes = True
    nodes = m_inst.node_tree.nodes; nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emis = nodes.new('ShaderNodeEmission')
    info = nodes.new('ShaderNodeObjectInfo') # Random por objeto
    m_inst.node_tree.links.new(info.outputs['Random'], emis.inputs['Color'])
    m_inst.node_tree.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    
    # 4. Mat_Floor (Gris 0.5)
    m_floor = bpy.data.materials.new(name="Mat_Floor")
    m_floor.use_nodes = True
    nodes = m_floor.node_tree.nodes; nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    m_floor.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    return m_sand, m_mask, m_inst, m_floor

def get_mesh_files():
    paths = []
    # Buscar en subcarpetas para tener variedad (50000 y 1551)
    for root, dirs, files in os.walk(MESH_DIR):
        for file in files:
            if file.lower().endswith(".obj"):
                paths.append(os.path.join(root, file))
    print(f"📂 Biblioteca cargada: {len(paths)} modelos.")
    return paths

def normalize_and_scale_phi(obj, phi_target):
    """Escala el objeto a un tamaño Phi específico."""
    # 1. Centrar
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # 2. Medir tamaño actual
    dims = obj.dimensions
    current_size = max(dims.x, dims.y, dims.z)
    if current_size <= 0: return
    
    # 3. Calcular tamaño objetivo en metros
    # Fórmula: Diametro(mm) = 2^(-Phi)
    # Ejemplo: Phi -1 = 2mm, Phi 4 = 0.0625mm
    target_mm = 2.0 ** (-phi_target)
    target_m = target_mm * SCALE_UNIT # Convertir a metros Blender
    
    # 4. Aplicar escala
    factor = target_m / current_size
    obj.scale = (factor, factor, factor)
    bpy.ops.object.transform_apply(scale=True)

def scatter_particles_phi_distribution(files, collection, m_sand):
    """Genera una mezcla realista de tamaños."""
    loaded_objs = []
    
    # Definición de capas para variedad (Cantidad, Rango Phi)
    # Phi más alto = Partícula más pequeña
    layers = [
        {"count": 250, "phi_min": 3.0, "phi_max": 4.0}, # Polvo fino (0.06 - 0.12 mm)
        {"count": 100, "phi_min": 1.0, "phi_max": 2.0}, # Arena media (0.25 - 0.5 mm)
        {"count": 15,  "phi_min": -1.0, "phi_max": 0.0} # Gravilla (1.0 - 2.0 mm)
    ]
    
    area_limit = VIEW_WIDTH / 2.0 * 1.2 # Esparcir un poco más allá de la cámara
    
    for layer in layers:
        # Seleccionar archivos al azar
        batch = random.choices(files, k=layer["count"])
        
        for fpath in batch:
            try:
                bpy.ops.wm.obj_import(filepath=fpath)
            except:
                try: bpy.ops.import_scene.obj(filepath=fpath)
                except: continue
            
            obj = bpy.context.selected_objects[0]
            
            # Asignar material visual
            if obj.data.materials: obj.data.materials[0] = m_sand
            else: obj.data.materials.append(m_sand)
            
            # ESCALA PHI ALEATORIA
            phi = random.uniform(layer["phi_min"], layer["phi_max"])
            normalize_and_scale_phi(obj, phi)
            
            # Posición
            x = random.uniform(-area_limit, area_limit)
            y = random.uniform(-area_limit, area_limit)
            # Z jitter leve para apilamiento visual
            z = random.uniform(0, 0.003) 
            
            obj.location = (x, y, z)
            obj.rotation_euler = (random.uniform(0,3.14), random.uniform(0,3.14), random.uniform(0,3.14))
            
            # Gestión de colección
            try: collection.objects.link(obj)
            except: pass
            try: bpy.context.scene.collection.objects.unlink(obj)
            except: pass
            
            loaded_objs.append(obj)
            
    return loaded_objs

def setup_camera():
    if bpy.context.scene.camera:
        bpy.data.objects.remove(bpy.context.scene.camera, do_unlink=True)
        
    bpy.ops.object.camera_add(location=(0, 0, 1.0))
    cam = bpy.context.object
    cam.name = "Camera_Train"
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = VIEW_WIDTH # 0.3m (Zoom macro para ver detalle)
    bpy.context.scene.camera = cam

def setup_lights():
    # Luz Principal (Igual a render_in_blender.py)
    bpy.ops.object.light_add(type='SUN', location=(5,5,10))
    l = bpy.context.object
    l.data.energy = 3.5
    l.rotation_euler = (math.radians(15), 0, math.radians(30))
    l.data.angle = 0.5 # Sombras suaves realistas
    
    # Luz Relleno
    bpy.ops.object.light_add(type='SUN', location=(-5,-5,5))
    l2 = bpy.context.object
    l2.data.energy = 1.5
    l2.rotation_euler = (math.radians(-45), 0, math.radians(200))
    
    return [l, l2]

def render_loop():
    # Carpetas
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "instances"), exist_ok=True)
    
    reset_scene()
    setup_render_engine()
    
    files = get_mesh_files()
    if not files: return
    
    m_sand, m_mask, m_inst, m_floor = create_materials()
    setup_camera()
    lights = setup_lights()
    
    # Piso
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0,0,-0.0001))
    floor = bpy.context.object
    floor.data.materials.append(m_floor)
    
    coll = bpy.data.collections.new("Particles")
    bpy.context.scene.collection.children.link(coll)
    
    print(f"🚀 Generando {NUM_IMAGES} imágenes HQ con distribución Phi...")
    
    for i in range(NUM_IMAGES):
        # Limpiar
        for o in coll.objects: bpy.data.objects.remove(o, do_unlink=True)
        
        # Esparcir con Phi Scale
        objs = scatter_particles_phi_distribution(files, coll, m_sand)
        
        # --- 1. IMAGEN (RGB) ---
        # Entorno Gris
        if not bpy.context.scene.world: bpy.context.scene.world = bpy.data.worlds.new("W")
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1) # Gris 0.5
        bg.inputs[1].default_value = 0.5 # Fuerza 0.5
        
        floor.hide_render = False
        for l in lights: l.hide_render = False
        for o in objs: 
            if o.data.materials: o.data.materials[0] = m_sand
            else: o.data.materials.append(m_sand)
            
        bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, "images", f"train_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        
        # --- 2. MÁSCARA (Binaria) ---
        bg.inputs[0].default_value = (0,0,0,1) # Negro
        bg.inputs[1].default_value = 0.0
        floor.hide_render = True
        for l in lights: l.hide_render = True
        for o in objs: o.data.materials[0] = m_mask
        
        bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, "masks", f"train_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        
        # --- 3. INSTANCIAS (Grises) ---
        for o in objs: o.data.materials[0] = m_inst
        bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, "instances", f"train_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        
        print(f"✅ Dataset HQ: {i+1}/{NUM_IMAGES}")

if __name__ == "__main__":
    render_loop()
