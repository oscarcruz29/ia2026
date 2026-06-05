# Juego con IA: Bala + Salto + Agachar + Random Forest

## Descripción general

Juego 2D desarrollado en **Python** con **Pygame** que combina mecánicas clásicas de acción lateral con un sistema de inteligencia artificial basado en **Random Forest** (scikit-learn). El jugador controla un mono que debe esquivar balas disparadas por una nave enemiga, ya sea saltando o agachándose. El objetivo principal es recolectar datos de juego manual para entrenar un modelo de clasificación que luego controle al personaje de forma autónoma.

---

## Mecánicas del juego

- **Salto**: el jugador presiona `ESPACIO` para saltar y esquivar balas a nivel del suelo o a media altura.
- **Agacharse**: presionando `↓` el jugador reduce su hitbox verticalmente para esquivar balas altas.
- **Balas**: se disparan desde la nave enemiga con velocidad y altura aleatorias (6 niveles: muy baja, baja, media-baja, media-alta, alta, máxima).
- **Colisión**: al ser golpeado, el estado del juego se reinicia sin borrar los datos de entrenamiento acumulados.

---

## Sistema de IA

### Características de entrada

El modelo recibe tres variables por frame:

| Feature | Descripción |
|---|---|
| `velocidad_bala` | Velocidad horizontal de la bala (negativa, avanza hacia el jugador) |
| `distancia` | Distancia horizontal entre la bala y el jugador |
| `altura_bala` | Nivel normalizado de altura de la bala (0.0 a 1.0) |

### Clases de salida

| Código | Acción |
|---|---|
| `0` | NADA (no hacer nada) |
| `1` | SALTAR |
| `2` | AGACHAR |

### Modelo

```
RandomForestClassifier(
    n_estimators = 50,
    max_depth    = 6,
    random_state = 42
)
```

Se usa un split 80/20 con estratificación para evaluar el *accuracy* en test.

### Recolección de datos

- Los datos se capturan **solo en modo manual** mientras la bala está activa.
- Para evitar el sesgo hacia la clase `NADA`, se aplica un **submuestreo aleatorio** (retención del ~38 % de muestras NADA).
- Se requieren mínimo **80 muestras** para permitir el entrenamiento.
- Las clases con menos del 10 % de representación se eliminan antes de entrenar, salvo que sean las únicas clases de acción disponibles.

---

## Modos de juego

| Modo | Tecla | Descripción |
|---|---|---|
| Manual | `M` | El jugador controla al mono; se recolectan datos de entrenamiento |
| Auto | `A` | El modelo entrenado controla al mono automáticamente |
| Entrenar | `T` | Entrena el Random Forest con los datos acumulados |
| Exportar CSV | `C` | Guarda los datos recolectados en `datos_modelo.csv` |
| Fullscreen | `F` | Alterna entre ventana y pantalla completa |
| Menú / Pausa | `ESC` / `P` | Regresa al menú principal |
| Salir | `Q` | Cierra el juego |

---

## Arquitectura del código

```
Juego
├── _apply_resolution()       # Escalado dinámico de assets y geometría
├── _cargar_assets()          # Carga de sprites y fondos con fallback
├── disparar_bala()           # Lógica de disparo con altura y velocidad aleatoria
├── iniciar_salto()           # Inicia la física de salto
├── manejar_salto()           # Actualiza posición vertical por gravedad
├── iniciar_agachar()         # Reduce hitbox del jugador
├── terminar_agachar()        # Restaura hitbox al tamaño normal
├── registrar_decision_manual()  # Captura y almacena muestras de entrenamiento
├── entrenar_modelo()         # Entrena y evalúa el RandomForestClassifier
├── decision_auto_accion()    # Infiere la acción óptima en modo automático
└── loop()                    # Bucle principal del juego (45 FPS)
```

---

## Dependencias

```
pygame
scikit-learn
```

Instalación:

```bash
pip install pygame scikit-learn
```

---

## Archivos relevantes

```
proyecto/
├── main.py                  # Código principal del juego
├── datos_modelo.csv         # Dataset exportado (generado al presionar C)
└── assets/
    ├── sprites/
    │   ├── mono_frame_1.png
    │   ├── mono_frame_2.png
    │   ├── mono_frame_3.png
    │   ├── mono_frame_4.png
    │   └── purple_ball.png
    └── game/
        ├── fondo2.png
        └── ufo.png
```

---

## Flujo de uso recomendado

1. Ejecutar el juego: `python main.py`
2. Seleccionar **M** para jugar en modo manual.
3. Jugar varias rondas esquivando balas con `ESPACIO` y `↓`.
4. Regresar al menú (`ESC`) y presionar **T** para entrenar el modelo.
5. Presionar **A** para ver al mono jugar solo con el modelo entrenado.
6. Opcionalmente presionar **C** para exportar los datos a CSV.
