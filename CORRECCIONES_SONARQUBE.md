# Correcciones de seguridad y calidad (SonarQube Cloud)

Proyecto: MI-PROYECTO-FINAL · Análisis de SonarQube Cloud sobre la rama `master`.

## Tabla de vulnerabilidades corregidas

| # | Archivo | Tipo de vulnerabilidad | Gravedad | Regla | Corrección aplicada |
|---|---------|------------------------|----------|-------|---------------------|
| 1 | `app.py` (`predict_batch`) | Path Traversal: el nombre del archivo subido se unía directamente a la ruta de guardado | Security / Blocker | pythonsecurity:S2083 | Se usa `secure_filename()` de `werkzeug.utils` antes de construir la ruta |
| 2 | `static/style.css` (líneas 172, 178, 184) | Función CSS desconocida `darken()`: el borde no se aplica | Reliability / Blocker | css:S9448 | Se reemplaza por colores de borde fijos (`#c3e6cb`, `#f5c6cb`, `#bee5eb`) |
| 3 | `app.py` (línea 13) | Clave secreta escrita en el código (`'fgr-secret-key'`) | Security | Hotspot | La clave se lee de la variable de entorno `SECRET_KEY`; si no existe se genera aleatoria |
| 4 | `app.py` (línea 16) | Protección CSRF ausente (hotspot "Make sure disabling CSRF protection is safe here") | Security / High | python:S4502 | Se activa `CSRFProtect(app)` y se añade el token `csrf_token` a todos los formularios POST |
| 5 | `templates/predict_batch.html` | `<select>`, `<input type="file">` sin `id` ni asociación con su `<label>` | Reliability / Medium | Web:InputWithoutLabelCheck | Se añaden `id` y `for` coincidentes |
| 6 | `templates/predict_individual.html` | Los 30 campos numéricos y el `<select>` sin `id` ni asociación con su `<label>` | Reliability / Medium | Web:InputWithoutLabelCheck | Se añade `id="Cn"` a cada campo y `for="Cn"` a cada `<label>` |

Error adicional detectado en la revisión: `predict_individual.html` tenía una letra "s" suelta después del label de C2. Se eliminó.

## Antes y después

### 1. Path Traversal (`app.py`)

Antes:
```python
if file.filename.endswith('.xlsx'):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
```

Después:
```python
filename = secure_filename(file.filename or '')
if filename.lower().endswith('.xlsx'):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
```

### 2. CSS con función inválida (`static/style.css`)

Antes:
```css
border: 1px solid darken(var(--alert-success-bg), 10%);
```

Después:
```css
border: 1px solid #c3e6cb;
```

### 3 y 4. Clave secreta y CSRF (`app.py`)

Antes:
```python
app.secret_key = 'fgr-secret-key'
```

Después:
```python
from flask_wtf.csrf import CSRFProtect
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)
CSRFProtect(app)
```

En cada formulario POST se añade:
```html
<input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
```

Dependencia nueva en `requirements.txt`: `Flask-WTF`.

### 5 y 6. Accesibilidad de formularios

Antes:
```html
<label>C1 - Edad de la madre (15 a 50 años):</label>
<input type="number" name="C1" required>
```

Después:
```html
<label for="C1">C1 - Edad de la madre (15 a 50 años):</label>
<input type="number" id="C1" name="C1" required>
```

## Verificación local

- La aplicación arranca y las rutas `/`, `/training`, `/predict_individual` y `/predict_batch` responden 200.
- Un POST sin token CSRF responde 400, por lo que la protección está activa.
- `secure_filename('../../evil.xlsx')` devuelve `evil.xlsx`, así que la ruta ya no puede salir de `data/uploads`.

## Notas para ejecutar

- Antes de arrancar en producción define la variable `SECRET_KEY`. Si no la defines, la app genera una clave aleatoria al iniciar; sirve para desarrollo, pero las sesiones no sobreviven a un reinicio.
- Instala dependencias con `pip install -r requirements.txt`.

## Ajustes posteriores al primer envío

- **Cabeceras de seguridad restauradas** en `app.py` (`add_security_headers`). La primera versión las había quitado por error. Se comprueba que la respuesta incluye `X-XSS-Protection` y `X-Content-Type-Options`.
- **Flujo de trabajo de SonarCloud** (`.github/workflows/sonarcloud.yml`) ejecutado en `ubuntu-latest` con la acción oficial fijada a una versión concreta. Comprueba que exista `SONAR_TOKEN` antes de iniciar el análisis y permite reejecutarlo manualmente desde GitHub Actions.
- **`sonar-project.properties`** con la clave del proyecto (`cape-dlmc_MI-PROYECTO-FINAL`) y la organización (`cape-dlmc`).
- En SonarCloud, el análisis automático debe estar **desactivado** (Administration > Analysis Method), porque el análisis se hace desde GitHub Actions.

## Requisito para reactivar el análisis de SonarCloud

La ejecución de GitHub Actions del 6 de noviembre de 2026 terminó antes de analizar el código: el registro muestra `SONAR_TOKEN` vacío y el error `Not authorized or organization not found`. Esto es un problema de autenticación/configuración, no un resultado de calidad del código.

1. En SonarCloud, abre la cuenta con permiso **Execute Analysis** para `cape-dlmc_MI-PROYECTO-FINAL` y genera un token de usuario en **My Account > Security**. No lo guardes en el repositorio ni lo compartas en el chat.
2. En GitHub, abre `cape-dlmc/MI-PROYECTO-FINAL` > **Settings > Secrets and variables > Actions > New repository secret**. Usa el nombre exacto `SONAR_TOKEN` y pega ahí el token.
3. Confirma que el proyecto de SonarCloud pertenece a la organización `cape-dlmc`, que su project key es `cape-dlmc_MI-PROYECTO-FINAL`, y que **Automatic Analysis** está desactivado.
4. Después de guardar el secreto, abre **Actions > SonarCloud > Run workflow** para `master`, o haz un nuevo push. El paso de validación ahora informa claramente si falta el secreto; el análisis solo puede completarse con un token válido.
