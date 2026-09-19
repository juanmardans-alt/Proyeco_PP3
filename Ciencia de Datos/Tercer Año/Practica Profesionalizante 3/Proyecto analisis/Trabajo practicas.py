import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carga de microdatos (Ejemplo: 2do Trimestre 2020 - Pandemia)
# Se asume que las bases ya están descargadas en el entorno
df_hogar = pd.read_csv('usu_hogar_T220.txt', sep=';')
df_indiv = pd.read_csv('usu_individual_T220.txt', sep=';')

# 2. Identificación de la Población Objetivo en base individual
# Criterio IFE: Desocupados (ESTADO=2) o Informales (PP07H=2) o Cuentapropistas (CAT_OCUP=2)
condicion_ife = (
    (df_indiv['ESTADO'] == 2) | 
    ((df_indiv['ESTADO'] == 1) & (df_indiv['PP07H'] == 2)) |
    ((df_indiv['ESTADO'] == 1) & (df_indiv['CAT_OCUP'] == 2))
)
df_indiv['es_poblacion_objetivo'] = np.where(condicion_ife, 1, 0)

# Agrupamos por hogar para saber si el hogar tiene al menos un destinatario IFE
hogares_objetivo = df_indiv.groupby('CODUSU')['es_poblacion_objetivo'].max().reset_index()

# 3. Merge con la base de hogares
df = pd.merge(df_hogar, hogares_objetivo, on='CODUSU', how='inner')

# 4. Definición del Umbral (Línea de Pobreza / CBT)
# Valor hipotético de la CBT por adulto equivalente para el período analizado
CBT_ADULTO_EQ = 14300  
df['CBT_hogar'] = df['IX_TOT'] * CBT_ADULTO_EQ # IX_TOT o la variable de adultos equivalentes (adequi)

# 5. Filtrar hogares objetivo que están bajo la línea de pobreza
df_target = df[(df['es_poblacion_objetivo'] == 1) & (df['ITF'] < df['CBT_hogar'])].copy()

# 6. Cálculo de la Brecha de Ingresos
df_target['brecha_ingresos'] = df_target['CBT_hogar'] - df_target['ITF']

# 7. Resultados: Montos y Eficiencia Fiscal
# Cálculo del Monto Fijo Óptimo (Promedio ponderado)
monto_optimo_fijo = np.average(df_target['brecha_ingresos'], weights=df_target['PONDIH'])

# Cálculo del Costo Fiscal bajo Transferencias Variables (Focalización perfecta pura)
costo_fiscal_perfecto = (df_target['brecha_ingresos'] * df_target['PONDIH']).sum()

# Cálculo del Costo Fiscal si se diera el Monto Fijo Óptimo a todos
costo_fiscal_fijo = monto_optimo_fijo * df_target['PONDIH'].sum()

print(f"Monto Fijo Óptimo Promedio: ${monto_optimo_fijo:,.2f}")
print(f"Costo Fiscal Total (Transferencia Variable Exacta): ${costo_fiscal_perfecto:,.2f}")
print(f"Costo Fiscal Total (Transferencia Fija Promedio): ${costo_fiscal_fijo:,.2f}")

# 8. Visualización Analítica: Brecha de Ingresos vs IFE
print("\nGenerando gráfico de eficiencia...")

plt.figure(figsize=(10, 6))

# Filtramos valores extremos muy altos solo para que el gráfico se vea más claro (ej. brechas > 150k)
brechas_grafico = df_target[df_target['brecha_ingresos'] < 150000]['brecha_ingresos']

# Creamos un histograma para ver cuántos hogares caen en cada nivel de brecha
plt.hist(brechas_grafico, bins=40, color='#4C72B0', edgecolor='black', alpha=0.7)

# Trazamos la línea del IFE Real que dio el Gobierno (Inversión idéntica para todos)
plt.axvline(10000, color='red', linestyle='--', linewidth=2.5, 
            label='IFE Real Otorgado ($10.000)')

# Trazamos la línea del Monto Óptimo que calculamos nosotros
plt.axvline(monto_optimo_fijo, color='green', linestyle='--', linewidth=2.5, 
            label=f'Monto Óptimo Promedio (${monto_optimo_fijo:,.0f})')

# Personalizamos etiquetas y diseño
plt.title('Distribución de la Brecha de Ingresos vs. Montos de Transferencia (IFE)', fontsize=13, fontweight='bold')
plt.xlabel('Dinero exacto que le falta al hogar para salir de la pobreza ($)', fontsize=11)
plt.ylabel('Frecuencia (Cantidad de hogares en la muestra)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Sombreamos las áreas de ineficiencia para el IFE Real de 10.000
plt.axvspan(0, 10000, color='red', alpha=0.1, label='Sobre-cobertura (Recibieron más de lo que necesitaban)')
plt.axvspan(10000, 150000, color='orange', alpha=0.1, label='Sub-cobertura (Recibieron el IFE pero siguieron siendo pobres)')

# Guardamos el gráfico como imagen en tu escritorio
ruta_imagen = r'C:\Users\Usuario\Desktop\Grafico_Eficiencia_IFE.png'
plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')

print(f"¡Listo! El gráfico se guardó exitosamente en: {ruta_imagen}")

# Mostramos el gráfico en pantalla
plt.show()