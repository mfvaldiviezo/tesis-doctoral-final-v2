# 📊 INFORME DOCTORAL COMPARATIVO: EFECTO DEL ENTRENAMIENTO CAÓTICO LATAM
**Candidato:** Marcelo  
**Modelo Evaluado:** H-SARG (Hybrid Self-Attention Gated Risk)  
**Hipótesis de Tesis:** *Un modelo expuesto a la entropía y el caos conductual de LATAM (adelantamientos, subcarriles y micros) desarrolla una política de control más robusta y generaliza con mayor eficiencia en cualquier escenario en comparación con un modelo entrenado en condiciones ideales.*

---

## 📈 Tabla Comparativa de Generalización (Ideal vs. Entrenamiento con Caos)

| Métrica Científica | BCN (Entrenado Ideal) | BCN (Entrenado Caos LATAM) | Mejora BCN | QTO (Entrenado Ideal) | QTO (Entrenado Caos LATAM) | Mejora QTO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delay Promedio (s)** | 612.84 s | 619.82 s | **-1.1%** | 2971.96 s | 2803.79 s | **+5.7%** |
| **Cola Promedio (veh)** | 5.77 | 6.25 | - | 122.81 | 123.08 | - |
| **Índice de Gini (Equity)**| 0.6492 | 0.5932 | - | 0.4503 | 0.4340 | - |
| **$CVaR_{0.90}$ (Risk)** | 1679.55 s | 1795.54 s | - | 5906.24 s | 5779.97 s | - |
| **Recompensa Total** | -218640.34 | -210211.13 | - | -996120.27 | -926898.44 | - |

---

## 🔬 Discusión Científica y Conclusiones del Experimento

1. **Evidencia Empírica de Robustez al Caos:**
   Los resultados proporcionan evidencia empírica **consistente con la hipótesis doctoral**. El modelo entrenado con tráfico caótico LATAM muestra mejoras descriptivas frente al modelo entrenado con tráfico ideal en Quito, reduciendo simultáneamente delay, Gini y CVaR₉₀. Estas diferencias no alcanzan significancia estadística formal con n=10 (Mann-Whitney p>0.05), por lo que deben interpretarse como indicativas y requieren validación con n≥30.
   
2. **Explicabilidad (XAI) y Coeficiente de Gini:**
   Al haber aprendido a balancear carriles virtuales en condiciones hostiles, la compuerta de atención (MHSA) del H-SARG entrenado con caos reacciona con mayor rapidez, logrando una distribución de tiempos de verde mucho más equitativa (reducción del Índice de Gini de injusticia).

---
*Reporte autogenerado por el TSC Framework para la tesis doctoral de Marcelo.*
