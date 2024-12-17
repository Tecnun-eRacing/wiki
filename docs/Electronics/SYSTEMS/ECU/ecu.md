# TeR_ECU
<kicanvas-embed src="../TeR_ECU.kicad_pcb" controls="basic"></kicanvas-embed>
La ECU (Electronics Control Unit) es la centralita principal del TeR su trabajo 
es leer leer todos los sensores conectados a ella a traves del bus CAN y decidir
como debe moverse el powertrain en función de estos.

## Decisiones de diseño
Se ha utilizado un micro STM32F405 dado su soporte para 2 CAN 2.0 y 
su potencia de computo elevada pudiendo llegar a los 180Mhz  para correr
algoritmos de control como el torque vectoring.

La ECU es una placa multiproposito ya que implementa puertos de entrada y salida que 
pueden ser utilizados para funciones auxiliares como el control de servos o la lectura 
de ciertos sensores.

Implementa:

- STM32F405VGTx Microcontroller (Cortex™-M4 Core@168mHz with FPU)
- USB For Diagnosis Operation
- 2x CAN 2.0 For its communication with the Powertrain CAN and main sensors CAN
- NEO M9N GPS modules from u-blox, for determining cars position (Posibility to drive an active Antenna)
- 9DOF IMU consisting in Accelerometer, Gyroscope and Magnetometer for accurate posting and torque algorithms
- 4 Digital Inputs (0V-24V Range)
- 4 PWM Outputs(3.3V), for servo control, including actuators
- 4 Analog Inputs (0V-3.3V) Possibility of configurable Input divider
- 4 Digital Outputs(0V-24V) High side mosfet drivers
- 2 WS2812 RGB Led Channels using integrated SPI for FS-Spain LightShow Acceleration


#