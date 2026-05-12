import bpy
import os
import sys
import math
import json
import argparse
from mathutils import Vector

# --- CONFIGURACIÓN POR DEFECTO ---
# (Se sobrescriben si lo llama el script automático, pero sirven de fallback)
SAMPLES_DEFAULT = 256
PRESETS = {
    "HD": 1024,
    "2K": 2048,
    "4K": 4096,
    "8K": 8192
}

def parse_args():
    """Captura los argumentos enviados por auto_pipeline.py"""
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_obj", required=True, help="Ruta al OBJ")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--out_name", default="render.png")
    ap.add_argument("--preset", default="8K", choices=list(PRESETS.keys()))
    ap.add_argument("--view_width_mm", type=float, default=50.0) # Referencial
    ap.add_argument("--samples", type=int, default=SAMPLES_DEFAULT)
    # Argumentos extra para compatibilidad (aunque tu código usa lógica propia)
    ap.add_argument("--rot_x_deg", type=float, default=0.0)
    ap.add_argument("--eps_z", type=float, default=0.0)
    ap.add_argument("--with_floor", action="store_true")
    
    return ap.parse_args(argv)

def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Limpieza profunda
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.worlds:
        if block.users == 0: bpy.data.worlds.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)

def setup_render_engine(res, samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    
    # Standard para igualar al dataset
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0 
    
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        for d in prefs.devices: d.use = True
    except: pass

def setup_world_grey():
    # TU LÓGICA ORIGINAL (CORRECTA)
    if not bpy.context.scene.world:
        world = bpy.data.worlds.new("World_Render")
        bpy.context.scene.world = world
    else: 
        world = bpy.context.scene.world
        
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    
    bg.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bg.inputs['Strength'].default_value = 0.5 
    
    out = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])

def create_sand_material():
    # TU MATERIAL ORIGINAL
    mat = bpy.data.materials.new(name="Mat_Sand")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (400,0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0,0)
    
    bsdf.inputs['Roughness'].default_value = 1.0 
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    geo = nodes.new('ShaderNodeNewGeometry'); geo.location = (-800, 200)
    ramp = nodes.new('ShaderNodeValToRGB'); ramp.location = (-500, 200)
    ramp.color_ramp.interpolation = 'CONSTANT'
    
    elements = ramp.color_ramp.elements
    elements.remove(elements[0])
    def add_color(pos, col):
        e = elements.new(pos)
        e.color = col
        
    elements[0].position = 0.0
    elements[0].color = (0.2, 0.15, 0.1, 1)
    add_color(0.16, (0.4, 0.3, 0.2, 1))
    add_color(0.33, (0.6, 0.5, 0.4, 1))
    add_color(0.50, (0.5, 0.5, 0.55, 1))
    add_color(0.66, (0.7, 0.65, 0.6, 1))
    add_color(0.83, (0.1, 0.1, 0.12, 1))

    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 1500.0 
    noise.inputs['Detail'].default_value = 5.0
    
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.8 
    
    mat.node_tree.links.new(geo.outputs['Random Per Island'], ramp.inputs['Fac'])
    mat.node_tree.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    mat.node_tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def create_floor_material():
    mat = bpy.data.materials.new(name="Mat_Floor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def create_floor_plane(material):
    bpy.ops.mesh.primitive_plane_add(size=1000, location=(0, 0, -0.001))
    floor = bpy.context.object
    floor.name = "Floor_Solid"
    if floor.data.materials: floor.data.materials[0] = material
    else: floor.data.materials.append(material)
    return floor

def import_and_fix_mesh(filepath, material):
    if not os.path.exists(filepath): 
        print(f"❌ ERROR: No existe {filepath}")
        return None
        
    try: bpy.ops.wm.obj_import(filepath=filepath)
    except: bpy.ops.import_scene.obj(filepath=filepath)
    
    objs = bpy.context.selected_objects
    if not objs: return None
    
    # Unir si hay múltiples mallas
    if len(objs) > 1:
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        
    obj = bpy.context.active_object
    if obj.data.materials: obj.data.materials[0] = material
    else: obj.data.materials.append(material)
    
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0,0,0)
    
    # TU ROTACIÓN ORIGINAL (0,0,0) - Mantenemos esto
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (0, 0, 0)
    
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    # Levantar del suelo
    z_height = obj.dimensions.z
    obj.location = (0, 0, z_height / 2.0)
    
    return obj


def setup_camera(target_obj, view_width_mm: float):
    if not target_obj:
        return 50.0

    # Cámara ortográfica top
    cam_z = target_obj.dimensions.z + 10.0
    bpy.ops.object.camera_add(location=(0, 0, cam_z), rotation=(0, 0, 0))
    cam = bpy.context.object
    cam.name = "Camera_Top"
    cam.data.type = 'ORTHO'

    # 50 mm -> 0.05 m (Blender trabaja “en metros” si unit scale=1)
    cam.data.ortho_scale = view_width_mm
    cam.data.clip_end = 2000.0

    bpy.context.scene.camera = cam
    return cam.data.ortho_scale


def setup_lights():
    lights = {}
    def add_sun(name, loc, rot, energy):
        bpy.ops.object.light_add(type='SUN', location=loc)
        l = bpy.context.object
        l.name = name
        l.data.energy = energy
        l.data.angle = 0.5 
        l.rotation_euler = rot
        return l
        
    lights['main'] = add_sun("Sun_Main", (5,5,10), (math.radians(15), 0, math.radians(30)), 3.5)
    lights['fill'] = add_sun("Sun_Fill", (-5,-5,5), (math.radians(-45), 0, math.radians(200)), 1.5)
    return lights

def main():
    # 1. PARSEAR ARGUMENTOS (Lo nuevo necesario para la automatización)
    args = parse_args()
    res = PRESETS[args.preset]
    
    # 2. SETUP DE ESCENA (Tu código original)
    reset_scene()
    setup_render_engine(res, args.samples)
    setup_world_grey() # <- Arreglado con tu versión
    
    mat_sand = create_sand_material()
    mat_floor = create_floor_material()
    
    # 3. IMPORTAR (Usando la ruta dinámica)
    print(f"🔄 Importando: {args.input_obj}")
    obj = import_and_fix_mesh(args.input_obj, mat_sand)
    
    if obj:
        create_floor_plane(mat_floor)
        ortho_scale = setup_camera(obj, args.view_width_mm)
        setup_lights()
        
        # 4. RENDERIZAR
        out_full_path = os.path.join(args.out_dir, args.out_name)
        bpy.context.scene.render.filepath = out_full_path
        print(f"📸 Renderizando en: {out_full_path}")
        bpy.ops.render.render(write_still=True)
        
        # 5. GUARDAR METADATA (Vital para la U-Net)
        # Calculamos mm_per_px basado en tu cámara
        # ortho_scale en Blender es el ancho de visión en metros (si unit scale=1)
        # Pero tu script usa unit scale 0.001? No, tu script usa scale 0.001 en el OBJ path original pero aqui no lo veo aplicado al importar.
        # Asumiremos que ortho_scale está en unidades de Blender.
        # Si ortho_scale = 0.05 (50mm), y res = 8192
        # mm_per_px = (ortho_scale * 1000) / res
        
        # OJO: Si el OBJ viene en metros, ortho_scale es metros.
        # Tu cámara se ajusta al objeto.
        
        mm_width = ortho_scale * 1000.0 # Convertir a mm
        mm_per_px = args.view_width_mm / res
        meta = {
          "mm_per_px": mm_per_px,
          "ortho_scale": ortho_scale,
          "preset": args.preset,
          "input_obj": args.input_obj,
          "view_width_mm": args.view_width_mm,
          "res": res
        }
        
        json_path = out_full_path.replace(".png", ".json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)
            
        print(f"✅ Metadata guardada: {json_path}")
    else:
        print("❌ Error crítico: No se pudo cargar el objeto.")

if __name__ == "__main__":
    main()
