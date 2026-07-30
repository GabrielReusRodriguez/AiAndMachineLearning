# Objetivo

Este proyecto es un recopilatorio de ejemplos de aplicaciones relacionadas con Machine Learning y AI.

# Stack tecnológico

- Python 3, con entorno virtual `venv` para ejecutar y testear el proyecto.
- Scripts Bash cuando haga falta automatizar tareas.
- Google Colab para notebooks.
- Google Coral TPU por USB (opcional) para inferencia.

# Convenciones de código

| Elemento | Convención | Ejemplo |
|----------|------------|---------|
| Variables | camelCase | `nombreUsuario` |
| Funciones | camelCase | `obtenerDatos()` |
| Clases | PascalCase | `GestorUsuarios` |
| Constantes | UPPER_SNAKE | `MAX_INTENTS` |

## Estilo

- Todo código Python ha de cumplir la guía de estilos PEP 8.
- Indentación de 2 espacios
- Línea máxima de 100 caracteres
- Comillas: dobles para strings, simples para caracteres

# Estructura de carpetas

- `src/`
  - `AI/` — proyectos de Inteligencia Artificial
  - `ML/` — proyectos de Machine Learning
- `docs/` — documentación de cada ejemplo
- `tests/` — tests automatizados

# Flujo de trabajo

Para implementar una nueva feature:

1. Actualiza el repositorio
2. Crear rama `feature/nombre-descripcion`
3. Implementar el código
4. Hacer commit atómico
5. Crear Pull Request
6. Esperar code review

Para corregir un error:

1. Actualiza el repositorio
2. Crear rama `fix/nombre-descripcion`
3. Implementar el código
4. Hacer commit atómico
5. Crear Pull Request
6. Esperar code review

Antes de hacer commit es obligatorio:

- [ ] Ejecutar linters / revisión de estilo (PEP 8)
- [ ] Pasar los tests: `MPLBACKEND=Agg pytest tests/ -v`
- [ ] Actualizar la documentación en `docs/` y, si aplica, el índice de `README.md`
