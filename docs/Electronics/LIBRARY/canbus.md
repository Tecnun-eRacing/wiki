# Análisis Técnico Avanzado del Protocolo CAN

## Especificaciones Físicas del Protocolo CAN

### Características Eléctricas
- **Nivel de Señal**: Diferencial de voltaje entre CAN_H y CAN_L
  - Estado dominante (lógico 0): CAN_H ≈ 3.5V, CAN_L ≈ 1.5V
  - Estado recesivo (lógico 1): CAN_H ≈ 2.5V, CAN_L ≈ 2.5V
- **Rango de Voltaje**: Típicamente 0-5V
- **Impedancia Característica**: 120 Ω (resistencia de terminación)

### Niveles de Velocidad de Transmisión
| Velocidad | Distancia Máxima | Descripción |
|-----------|------------------|-------------|
| 1 Mbps | Hasta 40 metros | Alta velocidad, corta distancia |
| 500 Kbps | Hasta 100 metros | Configuración estándar en automoción |
| 250 Kbps | Hasta 250 metros | Aplicaciones industriales medianas |
| 125 Kbps | Hasta 500 metros | Redes de largo alcance |
| 10 Kbps | Hasta 1000 metros | Redes de muy baja velocidad |

## Estructura Detallada del Protocolo CAN

### Formato de Trama CAN (ISO 11898-1)

#### Trama de Datos (Data Frame)
1. **Inicio (SOF - Start of Frame)**
   - Bit dominante que indica el inicio de transmisión
   - Sincronización de todos los nodos

2. **Identificador**
   - CAN 2.0A (11 bits)
   - CAN 2.0B (29 bits extendidos)
   - Define prioridad del mensaje
   - Método de identificación única

3. **Bits de Control**
   - **IDE (Identifier Extension)**: Indica formato estándar o extendido
   - **RTR (Remote Transmission Request)**: Diferencia entre trama de datos y remota
   - **DLC (Data Length Code)**: Longitud de datos (0-8 bytes)

4. **Campo de Datos**
   - Máximo 8 bytes de información
   - Transmisión bit a bit, comenzando por el bit más significativo

5. **CRC (Cyclic Redundancy Check)**
   - Polinomio generador: x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
   - 15 bits de verificación de error
   - Detección de hasta 5 bits de error consecutivos

6. **Campo de Reconocimiento (ACK)**
   - Bit de ACK dominante confirma recepción correcta
   - Transmitido por cualquier nodo receptor

7. **Delimitadores y Fin de Trama**
   - Bits recesivos que marcan fin de transmisión

### Mecanismo de Arbitraje

#### Principio de Resolución de Conflictos
- Implementa resolución bit a bit
- Nodos monitorizan el bus durante la transmisión
- Si un nodo escribe un bit dominante mientras otro escribe recesivo, prevalece el bit dominante
- El identificador más bajo (más bits dominantes) gana el arbitraje

### Control de Errores

#### Tipos de Errores
1. **Error de Bit**
   - Detección de inconsistencia entre bit transmitido y observado

2. **Error de Stuffing**
   - Verifica la inserción correcta de bits de relleno
   - Máximo 5 bits consecutivos idénticos

3. **Error de CRC**
   - Fallo en la verificación del checksum

4. **Error de Form**
   - Violación de formato de trama esperado

5. **Error de Acknowledge**
   - Ausencia de confirmación de recepción

#### Máquina de Estados de Error
- **Error Activo**: Transmite tramas de error
- **Error Pasivo**: Espera recuperación del bus
- **Bus Off**: Desconexión temporal del bus tras múltiples errores

### Mecanismos de Sincronización

#### Bit Timing
- **Segmentación del Bit**:
  - Sampling Point
  - Synchronization Segment
  - Propagation Time Compensation

#### Técnicas de Sincronización
- Hard Synchronization
- Resynchronization
- Bit Stuffing

## Implementación en Microcontroladores

### Módulos CAN Típicos
- Registro de máscara de filtrado
- Búfer de transmisión y recepción
- Controlador de interrupciones
- Generador de velocidad de baudios

### Ejemplo de Pseudocódigo de Transmisión

```c
void can_transmit(uint32_t id, uint8_t* data, uint8_t length) {
    // Configurar identificador
    CAN_TxMsg.id = id;
    CAN_TxMsg.length = length;
    
    // Copiar datos
    for(uint8_t i = 0; i < length; i++) {
        CAN_TxMsg.data[i] = data[i];
    }
    
    // Iniciar transmisión
    CAN_SendMessage(&CAN_TxMsg);
}
```

## Consideraciones Avanzadas

### Problemas de Rendimiento
- Latencia variable
- Overhead de arbitraje
- Limitaciones de ancho de banda

### Extensiones Modernas
- CAN FD (Flexible Data-Rate)
- CAN XL
- Mejoras en velocidad y longitud de datos