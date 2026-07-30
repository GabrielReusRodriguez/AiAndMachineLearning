# Objetivo

Este proyecto es un recopilatorio de ejemplos de aplicaciones relacionadasc on Machine Learning y AI.

# Stack Tecnológico

## Stack Tecnológico

- Se utiliza Python 3, se utilizará un entorno virtual venv de python para ejecutar y testear el proyecto.
- Scripts Bash
- Se utiliza Google Colab.
- Se puede utilizar Google Coral TPU por USB para inferencia.

## Convenciones de código.

| Elemento | Convención | Ejemplo |

|----------|------------|--------|

| Variables     | camelCase | nombreUsuario |
| Funciones     | camelCase | obtenerDatos()|   
| Clases        | PascalCase    | GestorUsuarios  |
| Constantes    | UPPER_SNAKE   | MAX_INTENTS   |

Todo código pyhton ha de cumplir la guía de estilos PEP 8.

### Estilo

El estilo del código del  proyecto será:
- Identación de 2 espacios
- Línea máxima de 100 carácteres
- Comillas: Dobles para Strings, simples para carácteres

# Estructura de carpetas

El proyecto tendrá la siguientes carpetas:
- src/
    -- AI/ para proyectos de Inteligencia Artificial
    -- ML/ para proyectos de Machine Learning
- docs/
- tests/

# Flujo de trabajo

Para implementar una nueva feature se seguirán los pasos:
    1 Crear rama "feature/nombre-descripcion"
    2 Implementar el código
    3 Hacer commit atómico
    4 Crear Pull Request
    5 Esperar code review

Antes de hacer commit es obligatorio:
    - [] Ejecutar linters: `npm run lint`
    - [] Pasar los tests: `npm test`
    - [] Actualizar la documentación