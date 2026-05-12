#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import os
import glob
import random
import json
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
SCALE_FACTOR = 1  # Mantener 1 si tus OBJs ya están en la escala correcta o si ajustas dinámicamente
MESH_DIR = "./sand_atlas_meshes"
OUTPUT_DIR = "./ensemble_dataset"  # Carpeta nueva para los 5 JSONs

# Rutas de búsqueda
PATHS_TO_SEARCH = [
    os.path.join(MESH_DIR, "1551_original_grains", "*.obj"),
    os.path.join(MESH_DIR, "50000_generated_grains", "*.obj")
]

NUM_PARTICLES = 4000
NUM_RUNS = 5  # Cantidad de "sacudidas" (Escenarios)

REAL_CONTAINER_SIZE = 50.0
CONTAINER_SIZE = REAL_CONTAINER_SIZE * SCALE_FACTOR
SPAWN_AREA = (CONTAINER_SIZE / 2.0) * 0.90  # Zona de caída segura

DROP_HEIGHT = 150.0
SIMULATION_STEPS = 4000

# Límites de tamaño (para normalización)
MIN_SIZE_SIM = 0.05 * SCALE_FACTOR
MAX_SIZE_SIM = 2.0 * SCALE_FACTOR

# Límites estrictos para filtrado final
LIMIT_XY = CONTAINER_SIZE / 2.0  # 25.0
LIMIT_Z_BOTTOM = 0.0             # El suelo es 0

# --- FUNCIONES ---

def get_normalization_scale(mesh_path, target_size_sim):
    """Mide el objeto y devuelve la escala para alcanzar el target_size."""
    # Usamos una instancia temporal de PyBullet ligera solo para medir
    # Nota: Hacemos esto 'lazy' o dentro de la simulación principal para no abrir/cerrar p.connect muchas veces
    # Para eficiencia, asumiremos que p está conectado cuando se llame a esta función.
    
    col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[1, 1, 1])
    # Crear cuerpo muy lejos
    body = p.createMultiBody(0, col, -1, [0, 0, -5000])
    min_aabb, max_aabb = p.getAABB(body)
    p.removeBody(body) # Limpiar

    dims = [max_aabb[i] - min_aabb[i] for i in range(3)]
    current_max = max(dims)

    if current_max <= 0: return 1.0
    return target_size_sim / current_max

def create_environment(container_size):
    """Crea suelo y muros contenedores."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9800) # Gravedad fuerte (unidades mm/s^2 aprox si escala=1)
    
    # Suelo
    p.loadURDF("plane.urdf", [0, 0, 0]) 

    # Muros invisibles o visibles (usamos cajas)
    wall_th = 5.0
    h_wall = 300.0
    s = container_size / 2.0
    
    # Posiciones: N, S, E, O
    walls = [
        ([0, s + wall_th/2, h_wall/2], [s + wall_th, wall_th/2, h_wall/2]), # Norte
        ([0, -s - wall_th/2, h_wall/2], [s + wall_th, wall_th/2, h_wall/2]), # Sur
        ([s + wall_th/2, 0, h_wall/2], [wall_th/2, s, h_wall/2]), # Este
        ([-s - wall_th/2, 0, h_wall/2], [wall_th/2, s, h_wall/2])  # Oeste
    ]

    for pos, half_extents in walls:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        p.createMultiBody(0, col, -1, pos)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. BÚSQUEDA DE ARCHIVOS
    print(f"📂 Buscando modelos en {MESH_DIR}...")
    mesh_files = []
    for pattern in PATHS_TO_SEARCH:
        found = glob.glob(pattern)
        mesh_files.extend(found)
    
    mesh_files = sorted(list(set(mesh_files)))
    if not mesh_files:
        print("❌ ERROR: No se encontraron archivos .obj")
        return
    print(f"   -> Total mallas disponibles: {len(mesh_files)}")

    # 2. SELECCIÓN DE LA MUESTRA (Pre-generación para consistencia)
    print("💎 Seleccionando las 4000 partículas 'Patrón' (se usarán en todas las simulaciones)...")
    
    # Iniciamos pybullet brevemente para calcular escalas (medir AABB)
    p.connect(p.DIRECT)
    particles_blueprint = []
    
    for _ in tqdm(range(NUM_PARTICLES), desc="Midiendo y seleccionando"):
        chosen_mesh = random.choice(mesh_files)
        target_size = random.uniform(MIN_SIZE_SIM, MAX_SIZE_SIM)
        
        try:
            scale_factor = get_normalization_scale(chosen_mesh, target_size)
            particles_blueprint.append({
                "mesh_path": chosen_mesh,
                "scale": scale_factor,
                "target_size_debug": target_size
            })
        except:
            pass # Si falla una medición, saltamos
            
    p.disconnect()
    print(f"✅ Muestra Patrón lista: {len(particles_blueprint)} partículas definidas.")

    # 3. BUCLE DE SIMULACIONES (Ensemble)
    for run_idx in range(NUM_RUNS):
        print(f"\n🎬 INICIANDO RUN {run_idx + 1}/{NUM_RUNS} ...")
        
        # Reiniciar motor físico
        p.connect(p.DIRECT)
        create_environment(CONTAINER_SIZE)
        
        spawned_ids = [] # Lista de tuplas (pybullet_id, blueprint_data)

        # Lluvia de partículas (Usando el blueprint)
        print("   🌧️ Generando caída...")
        for p_data in particles_blueprint:
            # Posición aleatoria (El 'sacudido')
            x = random.uniform(-SPAWN_AREA, SPAWN_AREA)
            y = random.uniform(-SPAWN_AREA, SPAWN_AREA)
            z = DROP_HEIGHT + random.uniform(0, 200) # Dispersión vertical
            
            orn = p.getQuaternionFromEuler([
                random.uniform(0, 2*np.pi),
                random.uniform(0, 2*np.pi),
                random.uniform(0, 2*np.pi)
            ])
            
            try:
                col = p.createCollisionShape(p.GEOM_MESH, 
                                           fileName=p_data["mesh_path"], 
                                           meshScale=[p_data["scale"]]*3)
                
                body_id = p.createMultiBody(1.0, col, -1, [x, y, z], orn)
                
                # Fricción para comportamiento de arena
                p.changeDynamics(body_id, -1, lateralFriction=0.8, rollingFriction=0.1, restitution=0.1)
                
                spawned_ids.append((body_id, p_data))
                
            except:
                pass

        # Asentamiento
        print("   ⏳ Asentando (Physics Step)...")
        for _ in tqdm(range(SIMULATION_STEPS), desc=f"Simulando Run {run_idx}", leave=False):
            p.stepSimulation()

        # 4. EXTRACCIÓN Y FILTRADO (Z > 0 y Dentro de Muros)
        final_data = []
        rejected_z = 0
        rejected_xy = 0
        
        print("   🧹 Filtrando y guardando...")
        for bid, bdata in spawned_ids:
            pos, orn = p.getBasePositionAndOrientation(bid)
            x, y, z = pos
            
            # FILTRO 1: Debajo del suelo (Tunneling)
            if z <= LIMIT_Z_BOTTOM:
                rejected_z += 1
                continue
                
            # FILTRO 2: Fuera del area 50x50 (Saltó el muro)
            if abs(x) > LIMIT_XY or abs(y) > LIMIT_XY:
                rejected_xy += 1
                continue
            
            final_data.append({
                "mesh_path": bdata["mesh_path"],
                "position": pos,
                "orientation": orn,
                "scale": bdata["scale"]
            })

        # Guardar JSON del Run actual
        out_file = os.path.join(OUTPUT_DIR, f"run_{run_idx}.json")
        with open(out_file, "w") as f:
            json.dump(final_data, f, indent=2)
            
        print(f"   ✅ Run {run_idx} guardado: {len(final_data)} partículas válidas.")
        print(f"      (Rechazadas: Z={rejected_z}, Fuera={rejected_xy})")
        
        p.disconnect() # Cerrar para limpiar memoria antes del siguiente Run

    print("\n🎉 ¡Generación de Ensemble completa!")

if __name__ == "__main__":
    main()